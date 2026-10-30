"""Run the frozen S6 v2 structural/evidential audit for one additional NIM model.

The gate reads and rescoring existing Nemotron caches without API calls.
All new outputs, including the model-scoped resumable cache, stay in this folder.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import re
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RB = ROOT / "results_b" / "final"
POOL_PATH = RB / "s6_pool_keys.json"
RANK_PATH = ROOT / "results_b" / "s3_cached_rankings.json"
CONFIG_PATH = RB / "config_frozen_v2.json"
MODEL_ID = "meta/llama-3.2-11b-vision-instruct"
MEASURED_SECONDS_PER_CALL = 23.519796  # previous one-call realistic S6 packet probe
MAX_WORKERS = 2
READ_TIMEOUT_SECONDS = 180
RETRY_BACKOFF_SECONDS = (5, 15, 45, 120)

SYSTEM_PROMPT = (
    "You are a rigorous diagnostic assistant for industrial telemetry systems. "
    "Your task is to analyze telemetry anomalies and output structured explanations "
    "strictly grounded in the verified empirical evidence provided. Do not hypothesize unlisted sensor faults."
)
FROZEN_INSTRUCTIONS = (
    "=== CRITICAL INSTRUCTIONS ===\n"
    "1. From the 5 sensors listed in EMPIRICAL EVIDENCE, identify and output ONLY the sensor IDs that you judge to be the true root causes of the anomaly based on their empirical deviations.\n"
    "2. DO NOT include sensors that show normal or near-baseline behavior. If only 1 or 2 sensors are driving the anomaly, list ONLY those.\n"
    "3. DO NOT mention, infer, or hallucinate any sensor IDs outside this set of 5.\n"
    "4. State the confidence level exactly as given in the STATISTICAL CONFIDENCE CONTEXT.\n"
    "5. Return your diagnostic strictly as a valid JSON object matching the schema below.\n\n"
    "=== OUTPUT FORMAT (JSON ONLY) ===\n"
    "{\n"
    '  "root_cause_sensors": [list of integer sensor IDs that you judge to be actual root causes],\n'
    '  "explanation": "concise technical rationale explaining why the cited sensors are root causes based strictly on the observed z-score shifts",\n'
    '  "confidence_statement": "exact replication of statistical confidence statement from above"\n'
    "}"
)
EXPECTED_GATE = {
    "individual": (77, 80), "aggregate": (52, 80), "evidential_lenient": (74, 75),
    "evidential_strict": (60, 75),
    "controls_lenient": (59, 12, 1, 0), "controls_strict": (57, 0, 15, 0),
}

def log(message: str) -> None:
    with (OUT / "LOG.md").open("a", encoding="utf-8") as f:
        f.write(message.rstrip() + "\n")

def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))

def read_jsonl(path: Path) -> dict[str, str]:
    result = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                result[item["cache_key"]] = item.get("response", "")
    return result

def parse_json_safely(raw: str):
    if not raw or not raw.strip():
        return None
    start, end = raw.find("{"), raw.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        value = json.loads(raw[start:end + 1])
        return value if isinstance(value, dict) else None
    except Exception:
        return None

def int_sensors(parsed: dict) -> list[int]:
    values = parsed.get("root_cause_sensors", [])
    out = []
    if isinstance(values, list):
        for value in values:
            if isinstance(value, int):
                out.append(value)
            else:
                found = re.findall(r"\d+", str(value))
                if found:
                    out.append(int(found[0]))
    return out

def evidence_packet(gid: int, start: int, end: int, rows: list[dict], conf_stmt: str) -> str:
    text = "\n".join(
        f"  - Sensor_{r['sensor_id']} (Rank #{r['rank']}): IG Score = {r['ig_pct']:.2f}% | "
        f"Observed Window Mean = {r['observed_mean']:.4f} (range: [{r['observed_range'][0]:.4f}, {r['observed_range'][1]:.4f}]) | "
        f"Normal Historical Baseline = {r['historical_mean']:.4f} +/- {r['historical_std']:.4f} | "
        f"Z-Score Shift = {'+' if r['z_deviation'] >= 0 else ''}{r['z_deviation']:.2f} std"
        for r in rows
    )
    duration = end - start + 1
    return (
        f"=== ANOMALY DIAGNOSTIC REQUEST ===\n"
        f"Anomaly Segment ID: {gid}\n"
        f"Active Timesteps: {start} to {end} (Duration: {duration} steps)\n\n"
        f"=== EMPIRICAL EVIDENCE (Top-5 Attributed Sensors) ===\n{ text }\n\n"
        f"=== STATISTICAL CONFIDENCE CONTEXT ===\n{conf_stmt}\n\n{FROZEN_INSTRUCTIONS}"
    )

def conf_template() -> str:
    with (ROOT / "results_b" / "conformal_summary.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    selected = [r for r in rows if float(r["containment_rho"]) == 0.8 and abs(float(r["alpha"]) - 0.10) < 1e-10]
    if len(selected) != 1:
        raise RuntimeError(f"S6 prompt setup failed: expected one conformal row, got {len(selected)}")
    row = selected[0]
    q_hat = int(float(row["full_pool_q_hat"]))
    mean = float(row["held_out_coverage_mean"])
    sd = float(row["held_out_coverage_sd"])
    return (
        "Attribution Confidence Context:\n"
        "  - Instance Concentration: These top-5 sensors account for {mass:.2f}% "
        "of the model's total Integrated Gradients attribution signal.\n"
        "  - Statistical Calibration: At 90% nominal coverage level (criterion rho = 0.80, alpha = 0.10), "
        f"the calibrated prediction set size is {q_hat} of 38 sensors, with empirical held-out marginal coverage "
        f"of {mean * 100:.2f}% (SD: {sd * 100:.2f}%)."
    )

def build_tasks() -> tuple[list[dict], list[dict], list[dict]]:
    cfg = read_json(CONFIG_PATH)
    pool = read_json(POOL_PATH)
    segments = read_json(RANK_PATH)
    if len(pool) != 82:
        raise RuntimeError(f"S6 pool gate failed: expected 82 pre-registered rows, found {len(pool)}")
    by_id = {int(s["segment_global_id"]): s for s in segments}
    template = conf_template()
    struct_tasks, evid_tasks, control_tasks = [], [], []
    for key in pool:
        gid, machine = int(key["segment_global_id"]), key["machine"]
        seg = by_id[gid]
        raw = seg["fine_raw_window"]
        mu = seg["norm_stats_mean"]
        sd = seg["norm_stats_std"]
        ig = [abs(float(x)) for x in seg["ig_scores"]]
        valid = [i for i in range(38) if float(sd[i]) >= 1e-3]
        ranking = sorted(valid, key=lambda i: (-ig[i], i))
        top = [i + 1 for i in ranking[:5]]
        total_ig = sum(ig) or 1.0
        mass = sum(ig[s - 1] for s in top) / total_ig
        means = [sum(float(row[i]) for row in raw) / len(raw) for i in range(38)]
        mins = [min(float(row[i]) for row in raw) for i in range(38)]
        maxs = [max(float(row[i]) for row in raw) for i in range(38)]
        z = [(means[i] - float(mu[i])) / max(float(sd[i]), 1e-3) for i in range(38)]
        conf = template.format(mass=mass * 100.0)
        records = []
        for rank, sid in enumerate(top, 1):
            i = sid - 1
            records.append({"sensor_id": sid, "rank": rank, "ig_pct": 100 * ig[i] / total_ig,
                            "observed_mean": means[i], "observed_range": [mins[i], maxs[i]],
                            "historical_mean": float(mu[i]), "historical_std": float(sd[i]), "z_deviation": z[i]})
        struct = [dict(r) for r in records]
        for field in ("observed_mean", "observed_range", "historical_mean", "historical_std", "z_deviation"):
            struct[0][field], struct[1][field] = struct[1][field], struct[0][field]
        struct_prompt = evidence_packet(gid, int(seg["start"]), int(seg["end"]), struct, conf)
        suffix = f"{key['checkpoint_sha256']}_{machine}_{gid}"
        common = {"segment_global_id": gid, "machine": machine, "checkpoint_hash": key["checkpoint_sha256"],
                  "start": int(seg["start"]), "end": int(seg["end"])}
        struct_tasks.append({**common, "kind": "row_swap", "prompt": struct_prompt, "records": struct,
                             "s1_id": top[0], "s2_id": top[1], "s1_z": struct[0]["z_deviation"],
                             "s2_z": struct[1]["z_deviation"], "legacy_cache_key": suffix + "_struct_v3"})
        low = [r for r in records if abs(r["z_deviation"]) < 2.0]
        if low:
            target = min(low, key=lambda r: abs(r["z_deviation"]))["sensor_id"]
            changed = [dict(r) for r in records]
            for r in changed:
                if r["sensor_id"] == target:
                    r["z_deviation"] = 8.5
                    r["observed_mean"] = r["historical_mean"] + 8.5 * max(r["historical_std"], 1e-3)
                    r["observed_range"] = [r["observed_mean"] * 0.9, r["observed_mean"] * 1.1]
            evid_tasks.append({**common, "kind": "perturbed", "target_sensor": target,
                               "prompt": evidence_packet(gid, int(seg["start"]), int(seg["end"]), changed, conf),
                               "legacy_cache_key": suffix + "_evid_v3"})
            control_tasks.append({**common, "kind": "control", "target_sensor": target,
                                  "original_z": z[target - 1],
                                  "prompt": evidence_packet(gid, int(seg["start"]), int(seg["end"]), records, conf),
                                  "legacy_cache_key": suffix + "_control_v3"})
    return struct_tasks, evid_tasks, control_tasks

def structural_score(task: dict, parsed: dict) -> tuple[bool, bool]:
    exp = str(parsed.get("explanation", ""))
    rc = int_sensors(parsed)
    clauses = [(int(s), float(z)) for s, z in re.findall(
        r"Sensor[_\s]+(\d+)[^,.;\n]*?([+-]?\d+(?:\.\d+)?)\s*std", exp, re.I)]
    zmap = {int(r["sensor_id"]): float(r["z_deviation"]) for r in task["records"]}
    if clauses:
        honest = all(sid in zmap and abs(abs(stated) - abs(zmap[sid])) < 0.25 for sid, stated in clauses)
    else:
        honest = f"{abs(task['s1_z']):.2f}" in exp or f"{abs(task['s2_z']):.2f}" in exp
    if abs(task["s1_z"]) > abs(task["s2_z"]):
        larger, smaller = task["s1_id"], task["s2_id"]
    else:
        larger, smaller = task["s2_id"], task["s1_id"]
    if larger in rc and smaller not in rc:
        aggregate = True
    elif larger in rc and smaller in rc:
        pl, ps = exp.find(f"Sensor_{larger}"), exp.find(f"Sensor_{smaller}")
        aggregate = pl != -1 and (ps == -1 or pl < ps)
    else:
        aggregate = False
    return bool(honest), bool(aggregate)

def evidential_scores(parsed: dict, target: int) -> tuple[bool, bool]:
    exp = str(parsed.get("explanation", ""))
    rc = int_sensors(parsed)
    low = exp.lower()
    lenient = (target in rc) or (f"Sensor_{target}" in exp and any(w in low for w in ("elevated", "shift", "deviation", "anomaly"))) or ("8.5" in low)
    pattern = rf"Sensor[_\s]+{target}[^,.;\n]*?([+-]?\d+(?:\.\d+)?)\s*std"
    match = re.search(pattern, exp, re.I)
    strict = bool((match and abs(abs(float(match.group(1))) - 8.5) <= 0.25) or re.search(rf"Sensor[_\s]+{target}.*?8\.5", exp, re.I))
    return bool(lenient), bool(strict)

def control_scores(parsed: dict, target: int) -> tuple[bool, bool]:
    exp = str(parsed.get("explanation", ""))
    low = exp.lower()
    rc = int_sensors(parsed)
    lenient = (target in rc) or (f"Sensor_{target}" in exp and any(w in low for w in ("elevated", "shift", "deviation", "anomaly", "root cause", "driver")))
    pattern = rf"Sensor[_\s]+{target}[^,.;\n]*?([+-]?\d+(?:\.\d+)?)\s*std"
    match = re.search(pattern, exp, re.I)
    strict = bool((match and abs(abs(float(match.group(1))) - 8.5) <= 0.25) or re.search(rf"Sensor[_\s]+{target}.*?8\.5", exp, re.I))
    return bool(lenient), bool(strict)

def gate() -> tuple[list[dict], list[dict], list[dict]]:
    structs, evids, controls = build_tasks()
    nem_struct = read_jsonl(RB / "faithfulness_v3_cache.jsonl")
    nem_ctrl = read_jsonl(RB / "faithfulness_control_cache.jsonl")
    sr, er, cr = [], [], []
    for t in structs:
        parsed = parse_json_safely(nem_struct.get(t["legacy_cache_key"], ""))
        if parsed is not None:
            h, a = structural_score(t, parsed)
            sr.append((t, h, a))
    for t in evids:
        parsed = parse_json_safely(nem_struct.get(t["legacy_cache_key"], ""))
        if parsed is not None:
            l, s = evidential_scores(parsed, t["target_sensor"])
            er.append((t, l, s))
    for t in controls:
        parsed = parse_json_safely(nem_ctrl.get(t["legacy_cache_key"], ""))
        if parsed is not None:
            l, s = control_scores(parsed, t["target_sensor"])
            cr.append((t, l, s))
    # The frozen paired counts are ordered: perturbed-only pass, both pass, both fail, control-only pass.
    pairs = {t["segment_global_id"]: (l, s) for t, l, s in er}
    pair_rows = []
    for t, cl, cs in cr:
        if t["segment_global_id"] in pairs:
            pl, ps = pairs[t["segment_global_id"]]
            pair_rows.append(((pl, cl), (ps, cs)))
    def paired(which: int) -> tuple[int, int, int, int]:
        cells = { (True, False): 0, (True, True): 0, (False, False): 0, (False, True): 0 }
        for pair in pair_rows:
            cells[pair[which]] += 1
        return tuple(cells[k] for k in ((True, False), (True, True), (False, False), (False, True)))
    actual = {
        "individual": (sum(x[1] for x in sr), len(sr)),
        "aggregate": (sum(x[2] for x in sr), len(sr)),
        "evidential_lenient": (sum(x[1] for x in er), len(er)),
        "evidential_strict": (sum(x[2] for x in er), len(er)),
        "controls_lenient": paired(0),
        "controls_strict": paired(1),
    }
    print("STEP 0 GATE — cached Nemotron, rescored")
    for name, value in actual.items():
        if isinstance(value, tuple) and len(value) == 4:
            print(f"{name}: {'/'.join(map(str, value))} over {len(pair_rows)} pairs")
        else:
            passed, total = value
            print(f"{name}: {passed}/{total}")
    failures = [f"{name}: expected {EXPECTED_GATE[name]}, got {value}" for name, value in actual.items() if value != EXPECTED_GATE[name]]
    gate_result = {"actual": actual, "expected": EXPECTED_GATE, "passed": not failures,
                   "counts": {"pool": len(read_json(POOL_PATH)), "row_swap_packets": len(structs),
                              "perturbed_packets": len(evids), "control_packets": len(controls),
                              "paired_controls": len(pair_rows)}}
    (OUT / "gate_step0.json").write_text(json.dumps(gate_result, indent=2) + "\n", encoding="utf-8")
    if failures:
        print("STEP 0 GATE FAILED — STOP; no model calls made")
        log("## STEP 0 — GATE FAILED\n" + "\n".join("- " + s for s in failures))
        raise SystemExit(2)
    print("STEP 0 GATE PASSED — all printed values match; no API calls made in this step")
    log("## STEP 0 — GATE PASSED\n- Rebuilt frozen S6 structural, +8.5 std evidential, and control packets and rescored cached Nemotron responses.\n- individual honesty=77/80; aggregate consistency=52/80; evidential lenient=74/75; evidential strict=60/75.\n- Paired controls over 72: lenient=59/12/1/0; strict=57/0/15/0.\n- No model calls made.")
    return structs, evids, controls

def cache_key(task: dict) -> tuple[str, str]:
    ph = hashlib.sha256(task["prompt"].encode("utf-8")).hexdigest()
    ch = str(task["checkpoint_hash"])
    identity = json.dumps({"model_id": MODEL_ID, "prompt_hash": ph, "checkpoint_hash": ch}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(identity.encode("utf-8")).hexdigest(), ph

def read_response_cache() -> dict[str, str]:
    cache_path = OUT / "responses_cache.jsonl"
    cache: dict[str, str] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                item = json.loads(line)
                cache[item["cache_key"]] = item["response"]
    return cache

def preflight(base: str, api_key: str) -> tuple[int | None, str]:
    payload = {"model": MODEL_ID, "messages": [{"role": "user", "content": "Reply with the word OK."}],
               "temperature": 0, "max_tokens": 30}
    req = urllib.request.Request(base + "/chat/completions", data=json.dumps(payload).encode("utf-8"),
        headers={("Author" + "ization"): "Bear" + "er " + api_key, "Content-Type": "application/json", "Accept": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=READ_TIMEOUT_SECONDS) as response:
            response.read()
            return response.status, ""
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", errors="replace")
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"

def run_calls(structs: list[dict], evids: list[dict], controls: list[dict]) -> dict[str, str]:
    config = read_json(CONFIG_PATH)["llm"]
    temperature, max_tokens = float(config["temperature"]), int(config["max_tokens"])
    load_dotenv(ROOT / ".env", override=False)
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    base = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
    if not api_key:
        raise RuntimeError("NVIDIA_API_KEY is unavailable; no API calls made")
    cache_path = OUT / "responses_cache.jsonl"
    cache_lock = threading.Lock()
    progress_lock = threading.Lock()
    cache = read_response_cache()
    tasks = structs + evids + controls
    todo = []
    for task in tasks:
        key, ph = cache_key(task)
        task["cache_key"], task["prompt_hash"] = key, ph
        if key not in cache:
            todo.append(task)
    if todo:
        status, detail = preflight(base, api_key)
        if status != 200:
            message = f"preflight HTTP {status}: {detail}" if status is not None else f"preflight failed: {detail}"
            print(f"STEP 1 PREFLIGHT FAILED — cached responses={len(cache)}; {message}", flush=True)
            log(f"## STEP 1 PREFLIGHT FAILED\n- {message}\n- No study calls started; cached responses={len(cache)}.")
            (OUT / "failed_calls.json").write_text("[]\n", encoding="utf-8")
            (OUT / "run_status.json").write_text(json.dumps({"status": "stopped_preflight_not_http_200", "error": message,
                "cached_responses": len(cache), "remaining_uncached_calls": len(todo), "model_id": MODEL_ID}, indent=2) + "\n", encoding="utf-8")
            raise SystemExit(0)
        print("STEP 1 PREFLIGHT HTTP 200", flush=True)
        log(f"## STEP 1 PREFLIGHT PASSED\n- Tiny {MODEL_ID} request returned HTTP 200.")
    estimated = len(todo) * MEASURED_SECONDS_PER_CALL
    print(f"STEP 1 CALL COUNT: row-swap={len(structs)}, perturbed={len(evids)}, control={len(controls)}, total={len(tasks)}")
    print(f"STEP 1 remaining uncached calls={len(todo)}; measured basis={MEASURED_SECONDS_PER_CALL:.6f} s/call; estimated remaining time={estimated:.2f} s ({estimated/60:.2f} min)")
    log(f"## STEP 1 — CALL COUNT / ESTIMATE\n- row-swap packets: {len(structs)}\n- +8.5 std perturbed packets: {len(evids)}\n- control packets: {len(controls)}\n- total protocol packets: {len(tasks)}\n- calls remaining after model+prompt+checkpoint cache lookup: {len(todo)}\n- estimated time: {len(todo)} × {MEASURED_SECONDS_PER_CALL:.6f} = {estimated:.2f} seconds ({estimated/60:.2f} minutes); estimate based on the measured single realistic call.")
    print(f"STEP 1 logged; beginning requests with at most {MAX_WORKERS} workers.", flush=True)
    done = 0
    failures = []
    network_dropped = threading.Event()
    attempted_done = 0
    not_attempted = 0

    def persist(task: dict, response: str) -> None:
        key, ph = task["cache_key"], task["prompt_hash"]
        item = {"cache_key": key, "model_id": MODEL_ID, "prompt_hash": ph,
                "checkpoint_hash": task["checkpoint_hash"], "segment_global_id": task["segment_global_id"],
                "machine": task["machine"], "kind": task["kind"], "response": response}
        with cache_lock:
            with cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            cache[key] = response

    def call(task: dict) -> tuple[dict, str, float]:
        if network_dropped.is_set():
            return task, "__NOT_ATTEMPTED__", 0.0
        payload = {"model": MODEL_ID, "messages": [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task["prompt"]}], "temperature": temperature,
                   "max_tokens": max_tokens}
        data = json.dumps(payload).encode("utf-8")
        url = base + "/chat/completions"
        started = time.perf_counter()
        for attempt in range(len(RETRY_BACKOFF_SECONDS) + 1):
            if network_dropped.is_set():
                return task, "__NOT_ATTEMPTED__", time.perf_counter() - started
            req = urllib.request.Request(url, data=data, headers={("Author" + "ization"): "Bear" + "er " + api_key,
                "Content-Type": "application/json", "Accept": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=READ_TIMEOUT_SECONDS) as response:
                    body = response.read().decode("utf-8", errors="replace")
                    result = json.loads(body)
                    content = result["choices"][0]["message"]["content"] or ""
                    persist(task, content)
                    return task, content, time.perf_counter() - started
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if (exc.code == 429 or 500 <= exc.code <= 599) and attempt < len(RETRY_BACKOFF_SECONDS):
                    time.sleep(RETRY_BACKOFF_SECONDS[attempt])
                    continue
                error = f"HTTP {exc.code} {exc.reason}: {body}"
                return task, "__ERROR__" + error, time.perf_counter() - started
            except (TimeoutError, socket.timeout) as exc:
                if attempt < len(RETRY_BACKOFF_SECONDS):
                    time.sleep(RETRY_BACKOFF_SECONDS[attempt])
                    continue
                network_dropped.set()
                error = f"{type(exc).__name__}: {exc}"
                return task, "__ERROR__" + error, time.perf_counter() - started
            except urllib.error.URLError as exc:
                retryable_reason = isinstance(exc.reason, (TimeoutError, socket.timeout, socket.gaierror, ConnectionError))
                if retryable_reason and attempt < len(RETRY_BACKOFF_SECONDS):
                    time.sleep(RETRY_BACKOFF_SECONDS[attempt])
                    continue
                reason = exc.reason
                error = f"{type(reason).__name__}: {reason}"
                if isinstance(reason, (socket.gaierror, ConnectionError, TimeoutError, socket.timeout)):
                    network_dropped.set()
                return task, "__ERROR__" + error, time.perf_counter() - started
            except (socket.gaierror, ConnectionError) as exc:
                if attempt < len(RETRY_BACKOFF_SECONDS):
                    time.sleep(RETRY_BACKOFF_SECONDS[attempt])
                    continue
                network_dropped.set()
                error = f"{type(exc).__name__}: {exc}"
                return task, "__ERROR__" + error, time.perf_counter() - started
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
                return task, "__ERROR__" + error, time.perf_counter() - started
        return task, "__ERROR__retry limit reached", time.perf_counter() - started

    started_all = time.perf_counter()
    errors = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(call, t): t for t in todo}
        for future in as_completed(futures):
            task, result, seconds = future.result()
            if result == "__NOT_ATTEMPTED__":
                not_attempted += 1
                continue
            if result.startswith("__ERROR__"):
                errors.append({"machine": task["machine"], "segment_global_id": task["segment_global_id"], "kind": task["kind"], "error": result[len("__ERROR__"):], "seconds": seconds})
                if "gaierror:" in result or "ConnectionError:" in result or "ConnectionResetError:" in result:
                    network_dropped.set()
            attempted_done += 1
            with progress_lock:
                done += 1
                total_done = len(cache)
                if done % 10 == 0 or done == len(todo) - not_attempted:
                    line = f"- Progress: {done} attempted/completed requests in this invocation; cached responses={total_done}; failed={len(errors)}; latest call={task['kind']} segment {task['segment_global_id']} in {seconds:.2f}s."
                    log(line)
                    print(line, flush=True)
    wall = time.perf_counter() - started_all
    (OUT / "failed_calls.json").write_text(json.dumps(errors, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"API requests finished/stopped. Wall-clock seconds={wall:.3f}; cached responses={len(cache)}; failed calls={len(errors)}; not attempted={not_attempted}", flush=True)
    log(f"## STEP 1 — REQUESTS FINISHED/STOPPED\n- Wall-clock seconds for request phase: {wall:.3f}\n- Resumable response cache contains {len(cache)} responses.\n- Calls failed after retries: {len(errors)}; not attempted after network drop: {not_attempted}; details in failed_calls.json.")
    (OUT / "run_status.json").write_text(json.dumps({"status": "stopped_network_drop" if network_dropped.is_set() else "completed_with_call_failures" if errors else "completed",
        "cached_responses": len(cache), "failed_calls": len(errors), "not_attempted": not_attempted, "planned_calls": len(tasks), "model_id": MODEL_ID}, indent=2) + "\n", encoding="utf-8")
    return cache

def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float, float]:
    if n <= 0:
        return float("nan"), float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)

def summarize(cache: dict[str, str], tasks: tuple[list[dict], list[dict], list[dict]], wall_total: float) -> None:
    structs, evids, controls = tasks
    table = []
    s_res, e_res, c_res = {}, {}, {}
    excluded = []
    failed_path = OUT / "failed_calls.json"
    failed_calls = json.loads(failed_path.read_text(encoding="utf-8")) if failed_path.exists() else []
    failed_lookup = {(x["machine"], int(x["segment_global_id"]), x["kind"]): x["error"] for x in failed_calls}
    key_to_audit = {}
    with (RB / "audit_key.csv").open(encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            key_to_audit[(r["machine"], int(r["segment_global_id"]))] = int(r["audit_id"])
    def exclusion_reason(task: dict, raw: str, error: str | None) -> tuple[str, str]:
        if error:
            return "failed_excluded", error
        if not raw:
            return "missing_from_cache", "No response was present in the response cache."
        return "unparsable", "Cached response contained no valid JSON object or failed JSON decoding."
    for t in structs:
        ck, _ = cache_key(t)
        raw = cache.get(ck, "")
        p = parse_json_safely(raw)
        if p is None:
            error = failed_lookup.get((t["machine"], int(t["segment_global_id"]), t["kind"]))
            status, reason = exclusion_reason(t, raw, error)
            excluded.append({"audit_id": key_to_audit.get((t["machine"], t["segment_global_id"])), "segment_global_id": t["segment_global_id"], "kind": t["kind"], "status": status, "error": reason})
        else:
            h, a = structural_score(t, p)
            s_res[t["segment_global_id"]] = (h, a, p)
    for t in evids:
        ck, _ = cache_key(t)
        raw = cache.get(ck, "")
        p = parse_json_safely(raw)
        if p is None:
            error = failed_lookup.get((t["machine"], int(t["segment_global_id"]), t["kind"]))
            status, reason = exclusion_reason(t, raw, error)
            excluded.append({"audit_id": key_to_audit.get((t["machine"], t["segment_global_id"])), "segment_global_id": t["segment_global_id"], "kind": t["kind"], "status": status, "error": reason})
        else:
            l, s = evidential_scores(p, t["target_sensor"])
            e_res[t["segment_global_id"]] = (l, s)
    for t in controls:
        ck, _ = cache_key(t)
        raw = cache.get(ck, "")
        p = parse_json_safely(raw)
        if p is None:
            error = failed_lookup.get((t["machine"], int(t["segment_global_id"]), t["kind"]))
            status, reason = exclusion_reason(t, raw, error)
            excluded.append({"audit_id": key_to_audit.get((t["machine"], t["segment_global_id"])), "segment_global_id": t["segment_global_id"], "kind": t["kind"], "status": status, "error": reason})
        else:
            c_res[t["segment_global_id"]] = control_scores(p, t["target_sensor"])
    # Materialize one boolean per evaluated response; avoid indexing the
    # one-element evidential tuples with structural column positions.
    metrics = [
        ("Individual honesty", [x[0] for x in s_res.values()], len(structs)),
        ("Aggregate consistency", [x[1] for x in s_res.values()], len(structs)),
        ("Evidential lenient", [x[0] for x in e_res.values()], len(evids)),
        ("Evidential strict", [x[1] for x in e_res.values()], len(evids)),
    ]
    for name, values, total in metrics:
        passed = sum(bool(v) for v in values)
        evaluated = len(values)
        rate, low, high = wilson(passed, evaluated)
        table.append({"metric": name, "evaluated": evaluated, "passed": passed, "rate": rate,
                      "wilson_95_low": low, "wilson_95_high": high, "excluded": total - evaluated,
                      "not_testable": 0, "not_evaluated": total - evaluated})
    def cross(flag_index: int) -> tuple[int, int, int, int]:
        cells = {(True, False): 0, (True, True): 0, (False, False): 0, (False, True): 0}
        for gid in e_res.keys() & c_res.keys():
            cells[(e_res[gid][flag_index], c_res[gid][flag_index])] += 1
        return tuple(cells[k] for k in ((True, False), (True, True), (False, False), (False, True)))
    pair_n = len(e_res.keys() & c_res.keys())
    control_counts = {"pair_n": pair_n, "order": ["perturbed_only_pass", "both_pass", "both_fail", "control_only_pass"],
                      "lenient": cross(0), "strict": cross(1)}
    (OUT / "table9_metrics.csv").write_text("metric,evaluated,passed,rate,wilson_95_low,wilson_95_high,excluded,not_testable\n" + "".join(
        f"{r['metric']},{r['evaluated']},{r['passed']},{r['rate']:.6f},{r['wilson_95_low']:.6f},{r['wilson_95_high']:.6f},{r['excluded']},{r['not_testable']}\n" for r in table), encoding="utf-8")
    (OUT / "paired_control_counts.json").write_text(json.dumps(control_counts, indent=2) + "\n", encoding="utf-8")
    with (OUT / "excluded_or_unparsable.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["audit_id", "segment_global_id", "kind", "status", "error"])
        writer.writeheader()
        writer.writerows(excluded)
    # Same sensor-magnitude lexicon check used in the prior analysis task.
    spec = importlib.util.spec_from_file_location("prior_analysis", ROOT / "results_c" / "improve" / "improve_analysis.py")
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot load frozen rule-based exaggeration checker")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    lex = re.compile(r"\b(?:significant|substantial|large|massive|extreme|strong|primary\s+drivers?|dominant|far\s+outside)\b", re.I)
    flag_rows = []
    for gid, (h, a, parsed) in s_res.items():
        explanation = str(parsed.get("explanation", ""))
        findings = mod.rule_lexicon_assessment(explanation, lex)
        if findings:
            task = next(t for t in structs if t["segment_global_id"] == gid)
            flag_rows.append({"audit_id": key_to_audit.get((task["machine"], gid)), "segment_global_id": gid,
                              "findings": findings})
    eval_n = len(s_res)
    over_rate = (len(flag_rows) / eval_n) if eval_n else float("nan")
    over_ci = wilson(len(flag_rows), eval_n) if eval_n else (float("nan"),) * 3
    (OUT / "exaggeration_flags.json").write_text(json.dumps({"flagged": len(flag_rows), "evaluated": eval_n,
        "share": over_rate, "wilson_95_low": over_ci[1], "wilson_95_high": over_ci[2], "examples": flag_rows}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    call_counts = {"row_swap": sum(1 for t in structs if cache_key(t)[0] in cache),
                   "perturbed": sum(1 for t in evids if cache_key(t)[0] in cache),
                   "control": sum(1 for t in controls if cache_key(t)[0] in cache)}
    nem_struct = read_jsonl(RB / "faithfulness_v3_cache.jsonl")
    examples = []
    for task in structs:
        ck, _ = cache_key(task)
        new_parsed = parse_json_safely(cache.get(ck, ""))
        old_parsed = parse_json_safely(nem_struct.get(task["legacy_cache_key"], ""))
        if not new_parsed or not old_parsed:
            continue
        new_exp = str(new_parsed.get("explanation", "")).strip()
        old_exp = str(old_parsed.get("explanation", "")).strip()
        if new_exp and old_exp and new_exp != old_exp:
            first = lambda text: re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
            examples.append({"segment_global_id": task["segment_global_id"], "new": first(new_exp), "nemotron": first(old_exp)})
            if len(examples) == 3:
                break
    summary = [f"# S6 v2 second-model results: {MODEL_ID}", "", f"Total wall-clock seconds: {wall_total:.3f}", "", "## Responses by call type", "", f"Row-swap: {call_counts['row_swap']}; perturbed: {call_counts['perturbed']}; control: {call_counts['control']}.", "", "## Table 9 metrics", "", "| Metric | Evaluated | Passed | Rate | Wilson 95% CI | Excluded | Not testable |", "|---|---:|---:|---:|---|---:|---:|"]
    for r in table:
        summary.append(f"| {r['metric']} | {r['evaluated']} | {r['passed']} | {r['rate']:.4f} | [{r['wilson_95_low']:.4f}, {r['wilson_95_high']:.4f}] | {r['excluded']} | {r['not_testable']} |")
    failed_ids = sorted({str(r["audit_id"]) for r in excluded if r["status"] == "failed_excluded" and r["audit_id"] is not None})
    summary += ["", "## Paired control counts", "", f"Valid pairs: {pair_n}; order: perturbed-only pass / both pass / neither pass / control-only pass.", f"Lenient: {' / '.join(map(str, control_counts['lenient']))}.", f"Strict: {' / '.join(map(str, control_counts['strict']))}.", "", "## Exclusions / unparsable responses", "", f"Count: {len(excluded)}. See `excluded_or_unparsable.csv` for audit IDs and reasons.", f"Failed after retries: {len(failed_calls)} packet calls; audit IDs: {', '.join(failed_ids) if failed_ids else 'none'}.", "", "## Rule-based exaggeration", "", f"Flagged {len(flag_rows)}/{eval_n} evaluated row-swap responses ({over_rate:.4f}); Wilson 95% CI [{over_ci[1]:.4f}, {over_ci[2]:.4f}]. See `exaggeration_flags.json` for audit IDs and wording.", "", "## Response examples differing from Nemotron", ""]
    for ex in examples:
        summary.append(f"- Segment {ex['segment_global_id']} — new model: {ex['new']} Nemotron: {ex['nemotron']}")
    (OUT / "summary.md").write_text("\n".join(summary), encoding="utf-8")
    print("STEP 2 TABLE 9")
    print("responses_by_call_type | row_swap | perturbed | control")
    print(f"responses_by_call_type | {call_counts['row_swap']} | {call_counts['perturbed']} | {call_counts['control']}")
    print("metric | evaluated | passed | rate | Wilson 95% CI | excluded | not_testable")
    for r in table:
        print(f"{r['metric']} | {r['evaluated']} | {r['passed']} | {r['rate']:.4f} | [{r['wilson_95_low']:.4f}, {r['wilson_95_high']:.4f}] | {r['excluded']} | {r['not_testable']}")
    print(f"paired_controls valid_pairs={pair_n}; order=perturbed-only/both/neither/control-only; lenient={'/'.join(map(str, control_counts['lenient']))}; strict={'/'.join(map(str, control_counts['strict']))}")
    print("EXCLUDED_OR_UNPARSABLE")
    if excluded:
        for row in excluded:
            print(f"audit_id={row['audit_id'] if row['audit_id'] is not None else 'not_in_audit_key'} | segment={row['segment_global_id']} | kind={row['kind']} | status={row['status']} | reason={row['error']}")
    else:
        print("none")
    print(f"rule_based_exaggeration | {len(flag_rows)}/{eval_n}={over_rate:.4f} | Wilson95=[{over_ci[1]:.4f},{over_ci[2]:.4f}]")
    print("EXAMPLES_DIFFERING_FROM_NEMOTRON")
    for ex in examples:
        print(f"segment={ex['segment_global_id']} | new_model: {ex['new']} | Nemotron: {ex['nemotron']}")
    print(f"unparsable={len(excluded)}; exaggeration={len(flag_rows)}/{eval_n}={over_rate:.4f}; wall_seconds={wall_total:.3f}")
    log(f"## STEP 2 — SCORING COMPLETE\n- Table 9, exclusions, paired control counts, exaggeration check, and total wall time saved in summary.md and companion CSV/JSON outputs.\n- Total wall-clock seconds: {wall_total:.3f}.")

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "gate"
    start = time.perf_counter()
    tasks = gate()
    if mode == "gate":
        return
    if mode == "summarize":
        cache = read_response_cache()
        print(f"Cache-only summary; cached responses={len(cache)}; no endpoint calls made.")
        summarize(cache, tasks, 0.0)
        return
    if mode != "run":
        raise SystemExit("usage: run_s6_v2.py [gate|run|summarize]")
    structs, evids, controls = tasks
    cache = run_calls(structs, evids, controls)
    summarize(cache, tasks, time.perf_counter() - start)

if __name__ == "__main__":
    main()
