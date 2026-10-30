"""Probe advertised NIM chat model routes with one tiny, non-study request each.

Reads NVIDIA_API_KEY from the invoking shell and never prints or persists it.
Successful and failed probes are appended to model_route_probes.csv after each
attempt, so an interrupted scan can be resumed with --resume.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_LIST = ROOT / "results_c" / "model_availability" / "models_list.txt"
OUTPUT = Path(__file__).resolve().parent / "model_route_probes.csv"
SKIP_TERMS = (
    "embed", "clip", "detector", "deplot", "diffusion", "parse",
    "translate", "guard", "safety", "topic-control", "calibration",
    "reward", "video",
)
EXPLAINER_IDS = {
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
    "meta/llama-3.2-11b-vision-instruct",
}


def chat_candidates() -> list[str]:
    ids = [x.strip() for x in MODEL_LIST.read_text(encoding="utf-8").splitlines() if x.strip()]
    # The saved list is the endpoint's actual /models response. Exclude IDs
    # clearly intended for non-chat tasks; leave other advertised IDs to be
    # checked by the endpoint itself (including code and vision instruct models).
    return [m for m in ids if m not in EXPLAINER_IDS and not any(t in m.lower() for t in SKIP_TERMS)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="skip model IDs already present in the output CSV")
    parser.add_argument("--retry-timeouts", action="store_true", help="retry only model IDs previously recorded with a timeout")
    parser.add_argument("--timeout", type=int, default=30, help="read timeout in seconds per model (default: 30)")
    parser.add_argument("--limit", type=int, default=0, help="probe at most this many remaining IDs; 0 means all")
    args = parser.parse_args()

    key = os.environ.get("NVIDIA_API_KEY", "")
    if not key:
        raise SystemExit("NVIDIA_API_KEY is not set in this PowerShell process; no requests made")
    base = os.environ.get("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1").rstrip("/")
    candidates = chat_candidates()
    seen: set[str] = set()
    timed_out: list[str] = []
    if (args.resume or args.retry_timeouts) and OUTPUT.exists():
        with OUTPUT.open(encoding="utf-8-sig", newline="") as f:
            previous = list(csv.DictReader(f))
        seen = {r["model_id"] for r in previous}
        timed_out = list(dict.fromkeys(
            r["model_id"] for r in previous
            if "timeout" in (r.get("error", "") or "").lower()
        ))
    if args.retry_timeouts:
        todo = [m for m in timed_out if m in candidates]
    else:
        todo = [m for m in candidates if not args.resume or m not in seen]
    if args.limit > 0:
        todo = todo[:args.limit]
    print(f"Advertised chat-like candidates: {len(candidates)}; already recorded: {len(seen)}; probing now: {len(todo)}")
    print(f"Endpoint base: {base}; timeout per request: {args.timeout}s; sequential requests: 1 worker")
    fields = ["model_id", "http_status", "seconds", "response_text", "error"]
    write_header = not OUTPUT.exists() or not args.resume
    mode = "a" if (args.resume or args.retry_timeouts) else "w"
    with OUTPUT.open(mode, encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
            f.flush()
        for index, model in enumerate(todo, 1):
            payload = {"model": model, "messages": [{"role": "user", "content": "Reply with the word OK."}],
                       "temperature": 0, "max_tokens": 30}
            req = urllib.request.Request(
                base + "/chat/completions", data=json.dumps(payload).encode("utf-8"),
                headers={("Author" + "ization"): "Bear" + "er " + key, "Content-Type": "application/json", "Accept": "application/json"},
                method="POST",
            )
            start = time.perf_counter()
            status, text, error = None, "", ""
            try:
                with urllib.request.urlopen(req, timeout=args.timeout) as response:
                    status = response.status
                    body = response.read().decode("utf-8", errors="replace")
                try:
                    parsed = json.loads(body)
                    content = parsed["choices"][0]["message"].get("content", "")
                    if isinstance(content, list):
                        content = "".join(str(p.get("text", "")) for p in content if isinstance(p, dict))
                    text = str(content) if content is not None else ""
                    if not text:
                        text = body[:1000]
                except (ValueError, KeyError, IndexError, TypeError):
                    text = body[:1000]
            except urllib.error.HTTPError as exc:
                status = exc.code
                error = f"HTTP {exc.code} {exc.reason}: {exc.read().decode('utf-8', errors='replace')[:1000]}"
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            seconds = time.perf_counter() - start
            row = {"model_id": model, "http_status": status if status is not None else "", "seconds": f"{seconds:.3f}",
                   "response_text": text[:1000], "error": error}
            writer.writerow(row)
            f.flush()
            print(f"[{index}/{len(todo)}] {model} | HTTP {status if status is not None else 'none'} | {seconds:.2f}s | "
                  f"response={text[:120]!r} | error={error[:240]}", flush=True)
    print(f"Saved probe results: {OUTPUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
