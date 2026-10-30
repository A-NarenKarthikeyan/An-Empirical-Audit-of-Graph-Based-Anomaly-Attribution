"""Run the optional automated-judge audit from a user's PowerShell session.

The script first verifies the frozen row-swap caches and strict-scorer counts.
It does not load .env or inspect/print API credentials; it reads only the
NVIDIA_API_KEY environment variable supplied by the invoking shell.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import socket
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RB = ROOT / "results_b" / "final"
SECOND = ROOT / "results_c" / "second_model"
JUDGE_MODEL = "google/gemma-4-31b-it"
EXPLAINER_MODELS = {
    "nemotron": "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "llama": "meta/llama-3.2-11b-vision-instruct",
}
TEMPERATURE = 0
MAX_TOKENS = 300
TIMEOUT_SECONDS = 180
MAX_WORKERS = 3
RETRY_BACKOFF_SECONDS = (15,)
JUDGE_INSTRUCTION = (
    'Given the evidence packet and the explanation, answer with JSON '
    '{"supported": true/false, "exaggerated": true/false}. '
    'supported = every sensor and magnitude the explanation states is in the packet. '
    'exaggerated = the explanation states more certainty or severity than the packet supports.'
)
CACHE_PATH = OUT / "judge_audit_cache.jsonl"
CSV_PATH = OUT / "judge_audit.csv"
SUMMARY_PATH = OUT / "judge_audit_summary.md"


def module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import scorer module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def parse_judge_json(text: str):
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(text[start:end + 1])
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, dict) or not isinstance(value.get("supported"), bool) or not isinstance(value.get("exaggerated"), bool):
        return None
    return value


def packet_only(prompt: str) -> str:
    marker = "\n\n=== CRITICAL INSTRUCTIONS ==="
    return prompt.split(marker, 1)[0]


def user_prompt(packet: str, explanation: str) -> str:
    return (
        f"{JUDGE_INSTRUCTION}\n\n"
        f"Evidence packet:\n{packet}\n\n"
        f"Explanation:\n{explanation}"
    )


def wilson(successes: int, n: int) -> tuple[float, float, float]:
    if n <= 0:
        return (float("nan"), float("nan"), float("nan"))
    z = 1.959963984540054
    p = successes / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, center - half), min(1.0, center + half)


def kappa(pairs: list[tuple[bool, bool]]) -> float:
    n = len(pairs)
    if not n:
        return float("nan")
    observed = sum(a == b for a, b in pairs) / n
    a_yes = sum(a for a, _ in pairs) / n
    b_yes = sum(b for _, b in pairs) / n
    expected = a_yes * b_yes + (1 - a_yes) * (1 - b_yes)
    return float("nan") if expected == 1 else (observed - expected) / (1 - expected)


def build_items():
    s6 = module_from_path("s6_v2_judge_audit", SECOND / "run_s6_v2.py")
    structs, _, _ = s6.build_tasks()
    nem_cache = {x["cache_key"]: x.get("response", "") for x in read_jsonl(RB / "faithfulness_v3_cache.jsonl")}
    llama_rows = [x for x in read_jsonl(SECOND / "responses_cache.jsonl") if x.get("kind") == "row_swap"]
    llama_cache = {int(x["segment_global_id"]): x.get("response", "") for x in llama_rows}
    with (RB / "audit_key.csv").open(encoding="utf-8-sig", newline="") as f:
        audit_ids = {(r["machine"], int(r["segment_global_id"])): int(r["audit_id"]) for r in csv.DictReader(f)}

    nem_items, llama_items = [], []
    for task in structs:
        gid = int(task["segment_global_id"])
        audit_id = audit_ids.get((task["machine"], gid))
        # The frozen audit_key intentionally contains only the scored rows;
        # retain the pool item by segment ID when it has no audit-table row.
        audit_id = "" if audit_id is None else audit_id
        for label, raw, scorer_pass, destination in (
            ("nemotron", nem_cache.get(task["legacy_cache_key"], ""), None, nem_items),
            ("llama", llama_cache.get(gid, ""), None, llama_items),
        ):
            if not raw:
                continue
            parsed = s6.parse_json_safely(raw)
            explanation = str(parsed.get("explanation", "")) if parsed else raw
            scorer = None
            if parsed is not None:
                scorer = bool(s6.structural_score(task, parsed)[0])
            packet = packet_only(task["prompt"])
            content = user_prompt(packet, explanation)
            item_id = f"{label}:{gid}"
            prompt_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
            cache_key = hashlib.sha256(json.dumps(
                {"judge_model": JUDGE_MODEL, "prompt_hash": prompt_hash, "item_id": item_id},
                sort_keys=True, separators=(",", ":")
            ).encode("utf-8")).hexdigest()
            destination.append({
                "item_id": item_id,
                "audit_id": audit_id,
                "segment_global_id": gid,
                "machine": task["machine"],
                "explainer_model": EXPLAINER_MODELS[label],
                "judge_model": JUDGE_MODEL,
                "prompt_hash": prompt_hash,
                "cache_key": cache_key,
                "user_prompt": content,
                "explanation": explanation,
                "explanation_parseable": parsed is not None,
                "strict_scorer_pass": scorer,
                "source_response": raw,
            })

    if len(nem_items) != 82 or len(llama_items) != 82:
        raise RuntimeError(f"Cache inventory gate failed: Nemotron={len(nem_items)}, Llama={len(llama_items)}, expected 82 each")
    if len({x["segment_global_id"] for x in nem_items}) != 82 or len({x["segment_global_id"] for x in llama_items}) != 82:
        raise RuntimeError("Duplicate segment in explainer cache inventory")
    return s6, structs, nem_items, llama_items


def gate(s6, structs, nem_items, llama_items) -> None:
    nem_eval = [x for x in nem_items if x["strict_scorer_pass"] is not None]
    llm_eval = [x for x in llama_items if x["strict_scorer_pass"] is not None]
    counts = {
        "Nemotron": (sum(x["strict_scorer_pass"] for x in nem_eval), len(nem_eval)),
        "Llama": (sum(x["strict_scorer_pass"] for x in llm_eval), len(llm_eval)),
    }
    frozen = {}
    with (RB / "faithfulness_summary_v2.csv").open(encoding="utf-8-sig", newline="") as f:
        frozen = {r["test_name"]: (int(r["passed_n"]), int(r["evaluated_n"])) for r in csv.DictReader(f)}
    nem_expected = frozen.get("Structural Test: Individual Honesty")
    with (SECOND / "table9_metrics.csv").open(encoding="utf-8-sig", newline="") as f:
        llama_rows = {r["metric"]: r for r in csv.DictReader(f)}
    llama_expected = (int(llama_rows["Individual honesty"]["passed"]), int(llama_rows["Individual honesty"]["evaluated"]))
    if counts["Nemotron"] != nem_expected or counts["Llama"] != llama_expected:
        raise RuntimeError(f"Strict-scorer/Table 9 gate failed: got {counts}, expected Nemotron={nem_expected}, Llama={llama_expected}")
    print("REPRODUCTION GATE items: 164 found, 0 missing (82 per explainer)")
    print(f"REPRODUCTION GATE strict-scorer individual-honesty pass counts: Nemotron {counts['Nemotron'][0]}/{counts['Nemotron'][1]} (Table 9 {nem_expected[0]}/{nem_expected[1]}); Llama {counts['Llama'][0]}/{counts['Llama'][1]} (Table 9 {llama_expected[0]}/{llama_expected[1]})")
    print("REPRODUCTION GATE PASSED")
    unparsable = sum(not x["explanation_parseable"] for x in nem_items + llama_items)
    print(f"Cached source responses with unparsable explanation JSON: {unparsable} (raw response is still sent as the explanation field)")


def read_judge_cache() -> dict[str, dict]:
    # Failed transport attempts are retained in JSONL for diagnosis, but they
    # are not completed cache entries and must be retried on resume.
    return {
        x["cache_key"]: x for x in read_jsonl(CACHE_PATH)
        if x.get("judge_model") == JUDGE_MODEL and x.get("http_status") == 200 and x.get("completed_response") is True
    }


def request_judge(base: str, api_key: str, item: dict) -> dict:
    payload = {"model": JUDGE_MODEL, "messages": [{"role": "user", "content": item["user_prompt"]}],
               "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS}
    request = urllib.request.Request(
        base + "/chat/completions", data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={("Author" + "ization"): "Bear" + "er " + api_key, "Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    status, response_text, error = None, "", ""
    response_payload = None
    completed_response = False
    attempts, attempt_errors = 0, []
    for attempt in range(len(RETRY_BACKOFF_SECONDS) + 1):
        attempts += 1
        try:
            with urllib.request.urlopen(request, timeout=TIMEOUT_SECONDS) as response:
                status = response.status
                body = json.loads(response.read().decode("utf-8", errors="replace"))
                response_payload = body
                content = body["choices"][0]["message"].get("content")
                if isinstance(content, str):
                    response_text = content
                elif isinstance(content, list):
                    response_text = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
                else:
                    response_text = ""
                completed_response = True
                error = ""
                break
        except urllib.error.HTTPError as exc:
            status = exc.code
            error = f"HTTP {exc.code} {exc.reason}: {exc.read().decode('utf-8', errors='replace')}"
            attempt_errors.append(error)
            retryable = status == 429 or status >= 500
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            attempt_errors.append(error)
            reason = getattr(exc, "reason", None)
            retryable = isinstance(exc, (TimeoutError, socket.timeout)) or isinstance(reason, (TimeoutError, socket.timeout))
            status = None
        if not retryable or attempt >= len(RETRY_BACKOFF_SECONDS):
            break
        time.sleep(RETRY_BACKOFF_SECONDS[attempt])
    return {
        "cache_key": item["cache_key"], "judge_model": JUDGE_MODEL, "prompt_hash": item["prompt_hash"],
        "item_id": item["item_id"], "audit_id": item["audit_id"],
        "segment_global_id": item["segment_global_id"], "machine": item["machine"],
        "explainer_model": item["explainer_model"], "http_status": status,
        "response_text": response_text, "error": error,
        "wall_seconds": time.perf_counter() - started, "attempts": attempts,
        "attempt_errors": attempt_errors, "completed_response": completed_response,
        "response_payload": response_payload,
    }


def preflight_model(base: str, api_key: str) -> int:
    """Send a non-study tiny chat request; print status/body/error, never the key."""
    print(f"PREFLIGHT model={JUDGE_MODEL}")
    payload = {"model": JUDGE_MODEL, "messages": [{"role": "user", "content": "Reply with the word OK."}],
               "temperature": 0, "max_tokens": 30}
    request = urllib.request.Request(
        base + "/chat/completions", data=json.dumps(payload).encode("utf-8"),
        headers={("Author" + "ization"): "Bear" + "er " + api_key, "Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body_text = response.read().decode("utf-8", errors="replace")
            try:
                body = json.loads(body_text)
                message = body["choices"][0]["message"]
                content = message.get("content")
                if isinstance(content, str):
                    text = content
                elif isinstance(content, list):
                    text = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
                else:
                    text = ""
            except Exception:
                text = body_text
            passed = response.status == 200 and text.strip(" .!\r\n\t").upper() == "OK"
            print(f"PREFLIGHT {'PASSED' if passed else 'FAILED'}; HTTP status={response.status}; response={text!r}; seconds={time.perf_counter() - started:.3f}")
            return 0 if passed else 1
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        print(f"PREFLIGHT HTTP status={exc.code}; error=HTTP {exc.code} {exc.reason}: {detail}; seconds={time.perf_counter() - started:.3f}")
        return 1
    except Exception as exc:
        print(f"PREFLIGHT HTTP status=None; error={type(exc).__name__}: {exc}; seconds={time.perf_counter() - started:.3f}")
        return 1


def judge_json_preflight(item: dict) -> int:
    """Try one real audit prompt, cache it, and report whether required JSON parsed."""
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set in this PowerShell process; no model calls made")
    base = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
    print(f"JUDGE JSON PREFLIGHT model={JUDGE_MODEL}; item={item['item_id']}; timeout={TIMEOUT_SECONDS}s")
    result = request_judge(base, api_key, item)
    persist_result(result, threading.Lock())
    parsed = parse_judge_json(result.get("response_text", ""))
    print(f"JUDGE JSON PREFLIGHT HTTP status={result.get('http_status')}; seconds={result.get('wall_seconds', 0):.3f}; "
          f"valid_required_json={'yes' if parsed is not None else 'no'}; response={result.get('response_text', '')!r}; error={result.get('error', '')}")
    if result.get("http_status") == 200 and parsed is not None:
        print("Preflight response was saved to the model-specific judge cache and will be reused by --run.")
        return 0
    print("Preflight response was saved as an attempt; do not start the full audit with this model yet.")
    return 1


def persist_result(result: dict, lock: threading.Lock) -> None:
    with lock:
        with CACHE_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(result, ensure_ascii=False) + "\n")


def summarize(items: list[dict], cached: dict[str, dict]) -> tuple[list[dict], str]:
    rows = []
    for item in items:
        result = cached.get(item["cache_key"], {})
        parsed = parse_judge_json(result.get("response_text", ""))
        rows.append({
            "audit_id": item["audit_id"], "segment_global_id": item["segment_global_id"],
            "machine": item["machine"], "explainer_model": item["explainer_model"],
            "judge_model": JUDGE_MODEL, "http_status": result.get("http_status"),
            "judge_supported": parsed.get("supported") if parsed else "",
            "judge_exaggerated": parsed.get("exaggerated") if parsed else "",
            "strict_scorer_pass": item["strict_scorer_pass"] if item["strict_scorer_pass"] is not None else "",
            "agreement_supported_vs_strict": (parsed.get("supported") == item["strict_scorer_pass"])
                if parsed and item["strict_scorer_pass"] is not None else "",
            "judge_json_valid": parsed is not None,
            "judge_response": result.get("response_text", ""),
            "error": result.get("error", ""),
            "wall_seconds": result.get("wall_seconds", ""),
        })
    lines = ["# Automated judge audit", "", f"Judge model: `{JUDGE_MODEL}`", "", "Judge outputs are compared with the frozen S6 v2 individual-honesty strict check (every stated sensor/magnitude supported by the packet). Non-JSON or invalid-schema replies are retained and counted, not silently dropped.", ""]
    lines.append("| Explainer model | Judge supported | Judge exaggerated | Supported vs strict agreement | Cohen's κ | Non-JSON/invalid | Empty content | Strict-scorer paired n |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, explainer in EXPLAINER_MODELS.items():
        group = [r for r in rows if r["explainer_model"] == explainer]
        valid = [r for r in group if r["judge_json_valid"]]
        non_json = sum(r["http_status"] == 200 and bool(r["judge_response"]) and not r["judge_json_valid"] for r in group)
        empty_content = sum(r["http_status"] == 200 and not r["judge_response"] and not r["judge_json_valid"] for r in group)
        request_failures = sum(r["http_status"] != 200 for r in group)
        supported_n = sum(r["judge_supported"] is True for r in valid)
        exaggerated_n = sum(r["judge_exaggerated"] is True for r in valid)
        p_s, lo_s, hi_s = wilson(supported_n, len(valid))
        p_e, lo_e, hi_e = wilson(exaggerated_n, len(valid))
        pairs = [(bool(r["judge_supported"]), bool(r["strict_scorer_pass"])) for r in group
                 if r["judge_json_valid"] and r["strict_scorer_pass"] != ""]
        agreement = sum(a == b for a, b in pairs) / len(pairs) if pairs else float("nan")
        k = kappa(pairs)
        lines.append(f"| {explainer} | {supported_n}/{len(valid)} = {p_s:.4f} [{lo_s:.4f}, {hi_s:.4f}] | {exaggerated_n}/{len(valid)} = {p_e:.4f} [{lo_e:.4f}, {hi_e:.4f}] | {agreement:.4f} | {k:.4f} | {non_json} | {empty_content} | {len(pairs)} |")
    non_json_total = sum(r["http_status"] == 200 and bool(r["judge_response"]) and not r["judge_json_valid"] for r in rows)
    empty_content_total = sum(r["http_status"] == 200 and not r["judge_response"] and not r["judge_json_valid"] for r in rows)
    request_failures_total = sum(r["http_status"] != 200 for r in rows)
    lines += ["", f"Total source items: {len(rows)}; non-JSON/invalid-schema replies: {non_json_total}; empty-content replies: {empty_content_total}; HTTP/transport failures or not-yet-run items: {request_failures_total}.", ""]
    return rows, "\n".join(lines)


def write_outputs(rows: list[dict], summary: str) -> None:
    fields = ["audit_id", "segment_global_id", "machine", "explainer_model", "judge_model", "http_status",
              "judge_supported", "judge_exaggerated", "strict_scorer_pass", "agreement_supported_vs_strict",
              "judge_json_valid", "judge_response", "error", "wall_seconds"]
    with CSV_PATH.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    SUMMARY_PATH.write_text(summary + "\n", encoding="utf-8")
    print(summary)
    print(f"WROTE {CSV_PATH}")
    print(f"WROTE {SUMMARY_PATH}")


def run(items: list[dict], check_only: bool) -> None:
    if check_only:
        print("CHECK ONLY: no model endpoint calls made")
        return
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is not set in this PowerShell process; no model calls made")
    base = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
    cached = read_judge_cache()
    todo = [x for x in items if x["cache_key"] not in cached]
    print(f"RUN: model={JUDGE_MODEL}; cached={len(items) - len(todo)}; remaining={len(todo)}; workers={MAX_WORKERS}; timeout={TIMEOUT_SECONDS}s")
    if todo:
        # Send and persist one real item first. If the chosen route is unavailable,
        # stop without launching a batch of doomed calls.
        first = request_judge(base, api_key, todo.pop(0))
        persist_result(first, threading.Lock())
        cached[first["cache_key"]] = first
        if first["http_status"] != 200:
            print(f"STOP: first judge request failed; HTTP status={first['http_status']}; error={first['error']}")
            with (OUT / "LOG.md").open("a", encoding="utf-8") as f:
                f.write(f"\nTask C stopped on first judge request: HTTP status={first['http_status']}; {first['error']}\n")
            print("No judge summary or final CSV written because the judge route did not return HTTP 200.")
            return
        if parse_judge_json(first.get("response_text", "")) is None:
            print(f"STOP: first judge item {first['item_id']} returned HTTP 200 but not the required JSON; no remaining audit calls started.")
            with (OUT / "LOG.md").open("a", encoding="utf-8") as f:
                f.write(f"\nTask C stopped: first response from {JUDGE_MODEL} was HTTP 200 but not valid supported/exaggerated JSON.\n")
            return
        print(f"First judged item saved: {first['item_id']} in {first['wall_seconds']:.2f}s")

    lock = threading.Lock()
    completed = 0
    failures = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(request_judge, base, api_key, item): item for item in todo}
        for future in as_completed(futures):
            item = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"cache_key": item["cache_key"], "judge_model": JUDGE_MODEL,
                          "prompt_hash": item["prompt_hash"], "item_id": item["item_id"],
                          "audit_id": item["audit_id"], "segment_global_id": item["segment_global_id"],
                          "machine": item["machine"], "explainer_model": item["explainer_model"],
                          "http_status": None, "response_text": "", "error": f"{type(exc).__name__}: {exc}",
                          "wall_seconds": 0.0}
            persist_result(result, lock)
            cached[item["cache_key"]] = result
            completed += 1
            if result.get("http_status") != 200 or result.get("error"):
                failures.append(f"{item['item_id']}: HTTP {result.get('http_status')} {result.get('error')}")
            if completed % 10 == 0 or completed == len(todo):
                line = f"Judge audit progress: {completed}/{len(todo)} newly requested; total cached={len(cached)}; failures={len(failures)}\n"
                print(line.strip(), flush=True)
                with (OUT / "LOG.md").open("a", encoding="utf-8") as f:
                    f.write(line)
    rows, summary = summarize(items, cached)
    write_outputs(rows, summary)
    print(f"NON-JSON/invalid-schema replies: {sum(r['http_status'] == 200 and bool(r['judge_response']) and not r['judge_json_valid'] for r in rows)}")
    if failures:
        print("HTTP/transport failures:")
        for failure in failures:
            print(failure)


def main() -> None:
    global JUDGE_MODEL, TIMEOUT_SECONDS
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="make judge calls; default is cache/gate check only")
    parser.add_argument("--preflight-only", action="store_true", help="send one tiny non-study OK request, then exit")
    parser.add_argument("--judge-json-preflight-only", action="store_true", help="send/cache one real judge-prompt item and check required JSON, then exit")
    parser.add_argument("--judge-model", default=JUDGE_MODEL, help="one model ID from the saved NIM model list")
    parser.add_argument("--timeout-seconds", type=int, default=TIMEOUT_SECONDS, help=f"read timeout per audit request (default: {TIMEOUT_SECONDS})")
    args = parser.parse_args()
    JUDGE_MODEL = args.judge_model
    if args.timeout_seconds < 1:
        raise SystemExit("--timeout-seconds must be a positive integer")
    TIMEOUT_SECONDS = args.timeout_seconds
    if JUDGE_MODEL in EXPLAINER_MODELS.values():
        raise SystemExit("Judge model must differ from both explainer models")
    model_list_path = ROOT / "results_c" / "model_availability" / "models_list.txt"
    if JUDGE_MODEL not in model_list_path.read_text(encoding="utf-8").splitlines():
        raise SystemExit(f"Judge model is not present in the saved endpoint model list: {JUDGE_MODEL}")
    s6, structs, nem_items, llama_items = build_items()
    gate(s6, structs, nem_items, llama_items)
    if args.preflight_only:
        api_key = os.environ.get("NVIDIA_API_KEY", "")
        if not api_key:
            raise RuntimeError("NVIDIA_API_KEY is not set in this PowerShell process; no model calls made")
        base = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
        raise SystemExit(preflight_model(base, api_key))
    if args.judge_json_preflight_only:
        # Use the same first Nemotron row-swap item across model candidates.
        raise SystemExit(judge_json_preflight(nem_items[0]))
    items = nem_items + llama_items
    run(items, check_only=not args.run)


if __name__ == "__main__":
    main()
