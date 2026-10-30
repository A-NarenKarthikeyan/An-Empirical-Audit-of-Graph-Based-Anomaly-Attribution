"""Automated judge audit over the frozen 82-segment S6 pools.

Real mode reads NVIDIA_API_KEY and JUDGE_MODEL from the environment only.
--mock-test exercises the inventory, gate, cache, and CSV path without network
access or credentials. Each judge response is persisted immediately.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RB = ROOT / "results_b" / "final"
SECOND = ROOT / "results_c" / "second_model"

NEMOTRON = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
LLAMA = "meta/llama-3.2-11b-vision-instruct"
EXPLAINER_NAMES = {NEMOTRON: "Nemotron", LLAMA: "Llama"}
EXPECTED = {
    "Nemotron": {"segments": 82, "magnitude": (77, 80), "lenient": (74, 75), "strict": (60, 75),
                 "unparseable_ids": {215, 287}},
    "Llama": {"segments": 82, "magnitude": (82, 82), "lenient": (61, 77), "strict": (58, 77),
              "unparseable_ids": set()},
}
FIXED_JUDGE_TEMPLATE = (
    'You are auditing an explanation against its evidence packet. Answer with JSON only: '
    '{{"supported": true/false, "exaggerated": true/false}}. '
    'supported = every sensor and every magnitude the explanation states appears in the packet. '
    'exaggerated = the explanation states more certainty or severity than the packet supports. '
    'EVIDENCE PACKET: {packet} EXPLANATION: {explanation}'
)
ENDPOINT = "https://integrate.api.nvidia.com/v1/chat/completions"
TEMPERATURE = 0
MAX_TOKENS = 300
READ_TIMEOUT_SECONDS = 600
MAX_WORKERS = 3
CACHE_PATH = OUT / "judge_cache.jsonl"
CSV_PATH = OUT / "judge_audit.csv"
FAILURES_PATH = OUT / "judge_failures.jsonl"
LOG_PATH = OUT / "LOG.md"
CSV_FIELDS = [
    "explainer_model", "segment_id", "judge_model", "supported", "exaggerated",
    "parse_ok", "explainer_parse_ok", "magnitude_honesty", "strict_pass", "lenient_pass",
    "http_status", "error",
]


def load_s6_module():
    path = SECOND / "run_s6_v2.py"
    spec = importlib.util.spec_from_file_location("frozen_s6_for_judge", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load frozen scorer module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def bool_or_none(value):
    if value is True or value == "True":
        return True
    if value is False or value == "False":
        return False
    return None


def read_json_lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def cache_key(judge_model: str, prompt: str, item_id: str) -> str:
    # This is the requested identity: sha256(judge_model + judge_prompt + item_id).
    return hashlib.sha256((judge_model + prompt + item_id).encode("utf-8")).hexdigest()


def make_prompt(packet: str, explanation: str) -> str:
    return FIXED_JUDGE_TEMPLATE.format(packet=packet, explanation=explanation)


def inventory_and_gate() -> list[dict]:
    """Rebuild the frozen source inventory and stop unless every supplied gate matches."""
    s6 = load_s6_module()
    struct_tasks, evid_tasks, _ = s6.build_tasks()
    pool = json.loads((RB / "s6_pool_keys.json").read_text(encoding="utf-8"))
    nem_raw = s6.read_jsonl(RB / "faithfulness_v3_cache.jsonl")
    llama_rows = read_json_lines(SECOND / "responses_cache.jsonl")
    llama_raw = {
        (int(row["segment_global_id"]), row.get("kind")): row.get("response", "")
        for row in llama_rows if row.get("model_id") == LLAMA
    }
    with (RB / "faithfulness_results_v2.csv").open(encoding="utf-8-sig", newline="") as f:
        nem_scores = {int(row["segment_global_id"]): row for row in csv.DictReader(f)}

    if len(pool) != 82 or len(struct_tasks) != 82 or len(evid_tasks) != 77:
        raise RuntimeError(f"Frozen pool/task inventory differs: pool={len(pool)}, row_swap={len(struct_tasks)}, perturbed={len(evid_tasks)}")
    if len(nem_scores) != 82:
        raise RuntimeError(f"Nemotron frozen scorer rows differ: expected 82, found {len(nem_scores)}")

    llama_evidence = {}
    llama_evidence_eval = []
    for task in evid_tasks:
        raw = llama_raw.get((int(task["segment_global_id"]), "perturbed"), "")
        parsed = s6.parse_json_safely(raw)
        if parsed is not None:
            lenient, strict = s6.evidential_scores(parsed, int(task["target_sensor"]))
            llama_evidence[int(task["segment_global_id"])] = (bool(strict), bool(lenient))
            llama_evidence_eval.append((bool(lenient), bool(strict)))

    nem_unparseable = set()
    llama_unparseable = set()
    nem_magnitude = []
    llama_magnitude = []
    items = []
    for task in struct_tasks:
        segment_id = int(task["segment_global_id"])
        packet = task["prompt"].split("\n\n=== CRITICAL INSTRUCTIONS ===", 1)[0]
        if not packet:
            raise RuntimeError(f"Empty evidence packet for segment {segment_id}")

        # Nemotron: use its saved response and frozen per-segment scorer labels.
        raw_n = nem_raw.get(task["legacy_cache_key"], "")
        if not raw_n:
            raise RuntimeError(f"Missing Nemotron explanation for segment {segment_id}")
        parsed_n = s6.parse_json_safely(raw_n)
        parse_n = parsed_n is not None and isinstance(parsed_n.get("explanation"), str)
        if not parse_n:
            nem_unparseable.add(segment_id)
            explanation_n = raw_n  # preserve raw response exactly; no repair or dropping
        else:
            explanation_n = parsed_n["explanation"]
        row_n = nem_scores.get(segment_id)
        if row_n is None:
            raise RuntimeError(f"Missing Nemotron scorer row for segment {segment_id}")
        magnitude_n = bool_or_none(row_n.get("individual_honesty"))
        strict_n = bool_or_none(row_n.get("evidential_strict_pass"))
        lenient_n = bool_or_none(row_n.get("evidential_pass"))
        if magnitude_n is not None:
            nem_magnitude.append(magnitude_n)

        # Llama: use the frozen second-model row-swap output and its same scorer.
        raw_l = llama_raw.get((segment_id, "row_swap"), "")
        if not raw_l:
            raise RuntimeError(f"Missing Llama row-swap explanation for segment {segment_id}")
        parsed_l = s6.parse_json_safely(raw_l)
        parse_l = parsed_l is not None and isinstance(parsed_l.get("explanation"), str)
        if not parse_l:
            llama_unparseable.add(segment_id)
            explanation_l = raw_l
            magnitude_l = None
        else:
            explanation_l = parsed_l["explanation"]
            magnitude_l = bool(s6.structural_score(task, parsed_l)[0])
            llama_magnitude.append(magnitude_l)
        strict_l, lenient_l = llama_evidence.get(segment_id, (None, None))

        for explainer_model, explanation, explainer_parse_ok, magnitude, strict, lenient in (
            (NEMOTRON, explanation_n, bool(parse_n), magnitude_n, strict_n, lenient_n),
            (LLAMA, explanation_l, bool(parse_l), magnitude_l, strict_l, lenient_l),
        ):
            item_id = f"{explainer_model}:{segment_id}"
            prompt = make_prompt(packet, explanation)
            items.append({
                "explainer_model": explainer_model,
                "segment_id": segment_id,
                "item_id": item_id,
                "judge_prompt": prompt,
                "magnitude_honesty": magnitude,
                "strict_pass": strict,
                "lenient_pass": lenient,
                "explainer_parse_ok": explainer_parse_ok,
            })

    nem_lenient = [bool_or_none(r.get("evidential_pass")) for r in nem_scores.values()]
    nem_strict = [bool_or_none(r.get("evidential_strict_pass")) for r in nem_scores.values()]
    nem_lenient = [x for x in nem_lenient if x is not None]
    nem_strict = [x for x in nem_strict if x is not None]
    llama_lenient = [x[0] for x in llama_evidence_eval]
    llama_strict = [x[1] for x in llama_evidence_eval]

    actual = {
        "Nemotron": {
            "segments": sum(1 for x in items if x["explainer_model"] == NEMOTRON),
            "explanations": sum(1 for t in struct_tasks if bool(nem_raw.get(t["legacy_cache_key"], ""))),
            "unparseable_ids": nem_unparseable,
            "magnitude": (sum(nem_magnitude), len(nem_magnitude)),
            "lenient": (sum(nem_lenient), len(nem_lenient)),
            "strict": (sum(nem_strict), len(nem_strict)),
        },
        "Llama": {
            "segments": sum(1 for x in items if x["explainer_model"] == LLAMA),
            "explanations": sum(1 for t in struct_tasks if bool(llama_raw.get((int(t["segment_global_id"]), "row_swap"), ""))),
            "unparseable_ids": llama_unparseable,
            "magnitude": (sum(llama_magnitude), len(llama_magnitude)),
            "lenient": (sum(llama_lenient), len(llama_lenient)),
            "strict": (sum(llama_strict), len(llama_strict)),
        },
    }
    print("PHASE 1 REPRODUCTION GATE")
    print(f"Pool: Nemotron {actual['Nemotron']['segments']}/82, Llama {actual['Llama']['segments']}/82; total={len(items)}/164")
    print(f"Raw explanations present: Nemotron {actual['Nemotron']['explanations']}/82; Llama {actual['Llama']['explanations']}/82")
    for name in ("Nemotron", "Llama"):
        for metric in ("magnitude", "lenient", "strict"):
            print(f"{name} {metric}: computed={actual[name][metric][0]}/{actual[name][metric][1]}; expected={EXPECTED[name][metric][0]}/{EXPECTED[name][metric][1]}")
        print(f"{name} unparseable raw responses: computed={sorted(actual[name]['unparseable_ids'])}; expected={sorted(EXPECTED[name]['unparseable_ids'])}")

    errors = []
    if len(items) != 164 or any(actual[n]["segments"] != 82 or actual[n]["explanations"] != 82 for n in EXPECTED):
        errors.append("82-per-model / 164-total / no-missing-explanation gate failed")
    for name, expected in EXPECTED.items():
        for metric in ("magnitude", "lenient", "strict"):
            if actual[name][metric] != expected[metric]:
                errors.append(f"{name} {metric}: expected {expected[metric]}, got {actual[name][metric]}")
        if actual[name]["unparseable_ids"] != expected["unparseable_ids"]:
            errors.append(f"{name} unparseable IDs: expected {sorted(expected['unparseable_ids'])}, got {sorted(actual[name]['unparseable_ids'])}")
    excluded = {int(r["segment_global_id"]) for r in nem_scores.values() if bool_or_none(r.get("individual_honesty")) is None}
    print(f"Nemotron magnitude-honesty excluded IDs: computed={sorted(excluded)}; expected=[215, 287]")
    if excluded != {215, 287}:
        errors.append(f"Nemotron magnitude-honesty excluded IDs differ: {sorted(excluded)}")
    if errors:
        raise RuntimeError("REPRODUCTION GATE FAILED: " + "; ".join(errors))
    print("All Phase 1 gates PASSED.")
    return items


def parse_judge_reply(text: str):
    try:
        value = json.loads(text.strip())
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(value, dict):
        return None
    if not isinstance(value.get("supported"), bool) or not isinstance(value.get("exaggerated"), bool):
        return None
    return {"supported": value["supported"], "exaggerated": value["exaggerated"]}


def redact_secret(text: str, api_key: str) -> str:
    return text.replace(api_key, "[REDACTED]") if api_key else text


def real_client(prompt: str, judge_model: str, api_key: str) -> dict:
    payload = {"model": judge_model, "messages": [{"role": "user", "content": prompt}],
               "temperature": TEMPERATURE, "max_tokens": MAX_TOKENS}
    request = urllib.request.Request(
        ENDPOINT, data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={("Author" + "ization"): "Bear" + "er " + api_key, "Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=READ_TIMEOUT_SECONDS) as response:
            status = response.status
            body_text = response.read().decode("utf-8", errors="replace")
        body = json.loads(body_text)
        content = body["choices"][0]["message"].get("content")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "".join(str(part.get("text", "")) for part in content if isinstance(part, dict))
        else:
            text = ""
        return {"http_status": status, "response_text": redact_secret(text, api_key), "error": "",
                "wall_seconds": time.perf_counter() - started}
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        return {"http_status": exc.code, "response_text": "",
                "error": redact_secret(f"HTTP {exc.code} {exc.reason}: {body}", api_key),
                "wall_seconds": time.perf_counter() - started}
    except Exception as exc:
        return {"http_status": None, "response_text": "",
                "error": redact_secret(f"{type(exc).__name__}: {exc}", api_key),
                "wall_seconds": time.perf_counter() - started}


def read_cache(path: Path) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in read_json_lines(path):
        grouped.setdefault(row["cache_key"], []).append(row)
    return grouped


def append_jsonl(path: Path, row: dict, lock: threading.Lock) -> None:
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()


def terminal_record(records: list[dict]) -> dict | None:
    if not records:
        return None
    latest = records[-1]
    if latest.get("parse_ok") is True:
        return latest
    if latest.get("http_status") != 200:
        return latest
    if latest.get("parse_ok") is False and int(latest.get("attempt", 1)) >= 2:
        return latest
    return None


def make_cache_record(item: dict, judge_model: str, key: str, attempt: int, result: dict) -> dict:
    parsed = parse_judge_reply(result.get("response_text", "")) if result.get("http_status") == 200 else None
    return {
        "cache_key": key,
        "item_id": item["item_id"],
        "judge_model": judge_model,
        "explainer_model": item["explainer_model"],
        "segment_id": item["segment_id"],
        "attempt": attempt,
        "http_status": result.get("http_status"),
        "response_text": result.get("response_text", ""),
        "supported": None if parsed is None else parsed["supported"],
        "exaggerated": None if parsed is None else parsed["exaggerated"],
        "parse_ok": parsed is not None if result.get("http_status") == 200 else None,
        "error": result.get("error", ""),
        "wall_seconds": result.get("wall_seconds"),
    }


def run_audit(items: list[dict], judge_model: str, api_key: str, cache_path: Path,
              csv_path: Path, failures_path: Path, log_path: Path,
              client: Callable[[str, str, str], dict] = real_client) -> None:
    cache_lock = threading.Lock()
    cache = read_cache(cache_path)
    pending = []
    for item in items:
        prompt = item["judge_prompt"]
        key = cache_key(judge_model, prompt, item["item_id"])
        prior = cache.get(key, [])
        done = terminal_record(prior)
        if done is not None:
            continue
        next_attempt = int(prior[-1].get("attempt", 0)) + 1 if prior else 1
        pending.append((item, key, next_attempt))
    print(f"RUN: judge={judge_model}; total_items={len(items)}; cached_terminal={len(items)-len(pending)}; remaining={len(pending)}; workers={MAX_WORKERS}")

    progress_lock = threading.Lock()
    completed = 0

    def process(task):
        item, key, attempt = task
        final = None
        this_attempt = attempt
        while this_attempt <= 2:
            result = client(item["judge_prompt"], judge_model, api_key)
            record = make_cache_record(item, judge_model, key, this_attempt, result)
            append_jsonl(cache_path, record, cache_lock)  # durable immediately after every reply/attempt
            cache.setdefault(key, []).append(record)
            if record["http_status"] != 200:
                append_jsonl(failures_path, {k: record[k] for k in ("cache_key", "item_id", "judge_model", "http_status", "error", "attempt")}, cache_lock)
                final = record
                break
            if record["parse_ok"]:
                final = record
                break
            if this_attempt >= 2:
                append_jsonl(failures_path, {k: record[k] for k in ("cache_key", "item_id", "judge_model", "http_status", "error", "attempt")}, cache_lock)
                final = record
                break
            print(f"Malformed judge JSON for {item['item_id']}; making the single permitted retry.", flush=True)
            this_attempt += 1
        return key, final

    if pending:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(process, task): task for task in pending}
            for future in as_completed(futures):
                key, final = future.result()
                completed += 1
                if final is not None:
                    cache[key] = [*cache.get(key, [])]
                if completed % 10 == 0 or completed == len(pending):
                    line = f"Judge audit progress: {completed}/{len(pending)} new items; terminal cache keys={sum(terminal_record(v) is not None for v in cache.values())}; failures logged={sum(1 for v in cache.values() if terminal_record(v) and terminal_record(v).get('http_status') != 200 or terminal_record(v) and terminal_record(v).get('parse_ok') is False)}"
                    print(line, flush=True)
                    with log_path.open("a", encoding="utf-8") as f:
                        f.write(line + "\n")

    latest_cache = read_cache(cache_path)
    rows = []
    for item in items:
        key = cache_key(judge_model, item["judge_prompt"], item["item_id"])
        record = terminal_record(latest_cache.get(key, []))
        rows.append({
            "explainer_model": item["explainer_model"],
            "segment_id": item["segment_id"],
            "judge_model": judge_model,
            "supported": "" if record is None or record.get("parse_ok") is not True else record["supported"],
            "exaggerated": "" if record is None or record.get("parse_ok") is not True else record["exaggerated"],
            "parse_ok": "" if record is None else record.get("parse_ok"),
            "explainer_parse_ok": item["explainer_parse_ok"],
            "magnitude_honesty": "" if item["magnitude_honesty"] is None else item["magnitude_honesty"],
            "strict_pass": "" if item["strict_pass"] is None else item["strict_pass"],
            "lenient_pass": "" if item["lenient_pass"] is None else item["lenient_pass"],
            "http_status": "" if record is None or record.get("http_status") is None else record["http_status"],
            "error": "" if record is None else record.get("error", ""),
        })
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    unresolved = sum(terminal_record(latest_cache.get(cache_key(judge_model, x["judge_prompt"], x["item_id"]), [])) is None for x in items)
    print(f"WROTE {csv_path}; rows={len(rows)}; unresolved={unresolved}")


def mock_test(items: list[dict]) -> None:
    fixed = '{"supported": true, "exaggerated": false}'
    calls = {"n": 0}

    def mock_client(prompt: str, model: str, key: str) -> dict:
        calls["n"] += 1
        return {"http_status": 200, "response_text": fixed, "error": "", "wall_seconds": 0.0}

    # Keep the mock artifacts under the writable repository; the OS temp
    # directory may be ACL-restricted in the managed desktop sandbox.
    with TemporaryDirectory(prefix="judge_mock_", dir=OUT) as temp:
        temp_dir = Path(temp)
        cache_path = temp_dir / "cache.jsonl"
        csv_path = temp_dir / "judge_audit.csv"
        failures_path = temp_dir / "failures.jsonl"
        log_path = temp_dir / "LOG.md"
        model = "mock/fixed-json"
        run_audit(items, model, "", cache_path, csv_path, failures_path, log_path, client=mock_client)
        if calls["n"] != 164:
            raise RuntimeError(f"Mock first pass expected 164 requests, got {calls['n']}")
        run_audit(items, model, "", cache_path, csv_path, failures_path, log_path, client=mock_client)
        if calls["n"] != 164:
            raise RuntimeError(f"Cache replay duplicated mock calls: expected 164 total, got {calls['n']}")
        with csv_path.open(encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 164 or any(r["parse_ok"] != "True" for r in rows):
            raise RuntimeError(f"Mock CSV check failed: rows={len(rows)}, all_parse_ok={all(r['parse_ok']=='True' for r in rows)}")
        bad_flags = {int(r["segment_id"]) for r in rows if r["explainer_model"] == NEMOTRON and r["explainer_parse_ok"] == "False"}
        if bad_flags != {215, 287}:
            raise RuntimeError(f"Mock CSV explainer parse flags differ: {sorted(bad_flags)}")
        print("MOCK END-TO-END TEST PASSED: fixed JSON parsed; 164 rows written; cache replay made zero duplicate calls; malformed explainer flags retained.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock-test", action="store_true", help="run full offline mock integration test; no credentials or network")
    args = parser.parse_args()
    items = inventory_and_gate()
    if args.mock_test:
        mock_test(items)
        return
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    judge_model = os.environ.get("JUDGE_MODEL", "")
    if not api_key:
        raise SystemExit("NVIDIA_API_KEY is required in this PowerShell process; no API calls made")
    if not judge_model:
        raise SystemExit("JUDGE_MODEL is required in this PowerShell process; no API calls made")
    if judge_model in {NEMOTRON, LLAMA}:
        raise SystemExit("JUDGE_MODEL must differ from both explainer model IDs; no API calls made")
    run_audit(items, judge_model, api_key, CACHE_PATH, CSV_PATH, FAILURES_PATH, LOG_PATH)


if __name__ == "__main__":
    main()
