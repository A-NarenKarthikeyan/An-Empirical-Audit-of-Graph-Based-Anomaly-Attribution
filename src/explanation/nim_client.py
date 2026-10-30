"""Stage 5: NVIDIA NIM Client — OpenAI-compatible endpoint with pre-abstention gate,
retry/backoff, structured JSON parsing, and JSONL caching.

Design:
- If prompt_builder already set abstained=True (pre-LLM gate), skip the API call entirely.
  The pre-formatted abstention JSON is passed through and logged with llm_called=False.
- If abstained=False, call the NIM API (temperature=0.0, 3 retries with exponential backoff).
  Parse the structured JSON response and log with llm_called=True.
- All results (pass-through abstentions + real calls) are appended to results/llm_cache.jsonl.
  Cache lookup is done before every call to avoid redundant API usage across runs.
"""

import os
import json
import time
import hashlib
import warnings
import datetime
from typing import Dict, Any, Optional, List

import numpy as np
import yaml
import torch

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv optional; key can be set directly in env

try:
    from openai import OpenAI, APIError, APITimeoutError, RateLimitError
except ImportError as e:
    raise ImportError("openai package required. Run: pip install openai") from e

try:
    from torch_geometric.data import Data as PyGData
except ImportError:
    PyGData = None

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from src.data_loader import load_dataset, compute_node_features_6d
from src.gnn_attribution.model import GATAnomalyDetector
from src.gnn_attribution.validate_attribution import compute_integrated_gradients_attribution
from src.explanation.prompt_builder import build_decoupled_explanation_prompt

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
CACHE_PATH = "results/llm_cache.jsonl"
MAX_RETRIES = 3
BACKOFF_BASE = 2.0   # seconds; attempt i waits BACKOFF_BASE**i seconds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_checkpoint_hash() -> str:
    path = "results/gnn_checkpoint.pt"
    if not os.path.exists(path):
        return "no_ckpt"
    with open(path, "rb") as f:
        import hashlib
        return hashlib.sha256(f.read()).hexdigest()[:8]

def _prompt_hash(prompt_text: str) -> str:
    """SHA-256 of the prompt for cache keying, incorporating checkpoint version."""
    import hashlib
    p_hash = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()[:16]
    ckpt_hash = _get_checkpoint_hash()
    return f"{ckpt_hash}_{p_hash}"


def _load_cache(cache_path: str) -> Dict[str, Dict]:
    """Load existing cache from JSONL into a dict keyed by prompt_hash."""
    cache: Dict[str, Dict] = {}
    if not os.path.exists(cache_path):
        return cache
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                key = entry.get("prompt_hash")
                if key:
                    cache[key] = entry
            except json.JSONDecodeError:
                pass
    return cache


def _append_cache(cache_path: str, entry: Dict) -> None:
    """Append a single result entry to the JSONL cache file."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _parse_llm_json(raw_content: str) -> Dict[str, Any]:
    """Extract and parse the JSON block from the LLM response.

    The model may wrap it in markdown fences (```json ... ```) or return bare JSON.
    Falls back to returning the raw string under 'raw_response' if parsing fails.
    """
    text = raw_content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        inner = []
        in_block = False
        for line in lines:
            if line.startswith("```") and not in_block:
                in_block = True
                continue
            if line.startswith("```") and in_block:
                break
            if in_block:
                inner.append(line)
        text = "\n".join(inner).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                pass
    return {"raw_response": raw_content, "parse_error": True}


# ---------------------------------------------------------------------------
# Core call function
# ---------------------------------------------------------------------------

def call_nim(
    prompt_builder_output: Dict[str, Any],
    model_name: str,
    api_key: str,
    temperature: float = 0.0,
    max_tokens: int = 600,
    cache_path: str = CACHE_PATH,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Main entry point: gate on pre-abstention, hit cache, then call NIM API.

    Args:
        prompt_builder_output: Full dict returned by build_decoupled_explanation_prompt.
        model_name: NIM model identifier (from config.yaml nim_model_name).
        api_key: NVIDIA API key.
        temperature: Sampling temperature (0.0 for deterministic).
        max_tokens: Maximum tokens in the completion.
        cache_path: Path to the JSONL cache file.
        verbose: Print progress.

    Returns:
        Dict with keys:
            segment_id, timesteps, abstained, abstain_reason_code,
            llm_called, llm_response (parsed JSON or pass-through), cache_hit,
            prompt_hash, timestamp_utc, model_name.
    """
    seg_id = prompt_builder_output.get("anomaly_id")
    timesteps = prompt_builder_output.get("timesteps")
    already_abstained = prompt_builder_output.get("abstained", False)
    reason_code = prompt_builder_output.get("abstain_reason_code")
    cum_mass = prompt_builder_output.get("cumulative_mass", 0.0)

    prompt_text = prompt_builder_output.get("prompt", "")
    system_text = prompt_builder_output.get("system_message", "")
    p_hash = _prompt_hash(prompt_text)

    now_utc = datetime.datetime.utcnow().isoformat() + "Z"

    # ---- 1. Pre-abstention pass-through --------------------------------
    if already_abstained:
        pass_through_response = {
            "root_cause_sensors": [],
            "explanation": (
                f"Abstained: {prompt_builder_output.get('abstention_reason', 'insufficient evidence')} "
                f"[abstain_reason_code: {reason_code}]"
            ),
            "confidence_statement": f"Abstained ({cum_mass * 100:.1f}% filtered top-K mass).",
            "abstained": True,
            "abstain_reason_code": reason_code,
        }

        result = {
            "segment_id": seg_id,
            "timesteps": list(timesteps) if timesteps else [],
            "abstained": True,
            "abstain_reason_code": reason_code,
            "llm_called": False,
            "llm_response": pass_through_response,
            "cache_hit": False,
            "prompt_hash": p_hash,
            "filtered_mass": round(cum_mass, 6),
            "unfiltered_mass": prompt_builder_output.get("unfiltered_mass"),
            "timestamp_utc": now_utc,
            "model_name": model_name,
        }

        _append_cache(cache_path, result)

        if verbose:
            print(
                f"  [Seg {seg_id}] ABSTAIN (pass-through, llm_called=False) | "
                f"reason: {reason_code} | "
                f"filtered_mass={cum_mass * 100:.1f}%"
            )
        return result

    # ---- 2. Cache lookup ------------------------------------------------
    cache = _load_cache(cache_path)
    if p_hash in cache:
        cached = cache[p_hash]
        cached["cache_hit"] = True
        if verbose:
            sensors = cached.get("llm_response", {}).get("root_cause_sensors", [])
            print(
                f"  [Seg {seg_id}] CACHE HIT (llm_called=True, reusing) | "
                f"root_cause_sensors={sensors}"
            )
        return cached

    # ---- 3. Real API call with retry/backoff ----------------------------
    client = OpenAI(base_url=NIM_BASE_URL, api_key=api_key)
    messages = [
        {"role": "system", "content": system_text},
        {"role": "user", "content": prompt_text},
    ]

    raw_content = None
    usage_info = None
    last_error = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if verbose:
                print(
                    f"  [Seg {seg_id}] Calling NIM API (attempt {attempt}/{MAX_RETRIES}) ..."
                )
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            raw_content = response.choices[0].message.content
            if response.usage:
                usage_info = {
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            break  # success

        except (RateLimitError, APITimeoutError, APIError) as exc:
            last_error = str(exc)
            if attempt < MAX_RETRIES:
                wait = BACKOFF_BASE ** attempt
                if verbose:
                    print(f"    API error on attempt {attempt}: {exc}. Retrying in {wait:.0f}s ...")
                time.sleep(wait)
            else:
                if verbose:
                    print(f"    All {MAX_RETRIES} attempts failed. Last error: {exc}")

    # ---- 4. Parse and package result ------------------------------------
    if raw_content is not None:
        parsed = _parse_llm_json(raw_content)
        llm_response = parsed
        call_status = "success"
    else:
        llm_response = {
            "root_cause_sensors": [],
            "explanation": f"API call failed after {MAX_RETRIES} attempts: {last_error}",
            "abstained": False,
            "api_error": True,
        }
        call_status = "api_error"

    result = {
        "segment_id": seg_id,
        "timesteps": list(timesteps) if timesteps else [],
        "abstained": False,
        "abstain_reason_code": None,
        "llm_called": True,
        "llm_response": llm_response,
        "call_status": call_status,
        "cache_hit": False,
        "prompt_hash": p_hash,
        "filtered_mass": round(cum_mass, 6),
        "unfiltered_mass": None,
        "usage": usage_info,
        "timestamp_utc": now_utc,
        "model_name": model_name,
    }

    _append_cache(cache_path, result)

    if verbose:
        sensors = llm_response.get("root_cause_sensors", [])
        print(
            f"  [Seg {seg_id}] NIM RESPONDED (llm_called=True) | "
            f"root_cause_sensors={sensors} | status={call_status}"
        )

    return result


# ---------------------------------------------------------------------------
# 8-segment test harness
# ---------------------------------------------------------------------------

def run_nim_on_machine_1_1() -> None:
    """Run the full pipeline on 8 machine-1-1 GT segments and print a results table."""
    config_path = "configs/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Config
    top_k = int(cfg.get("explanation_top_k", 5))
    min_topk_mass = float(cfg.get("min_topk_mass", 0.70))
    min_sensor_std = float(cfg.get("min_sensor_std", 1e-3))
    fine_margin = int(cfg.get("fine_window_margin", 15))
    alpha = float(cfg.get("conformal_alpha", 0.10))
    recall_thresh = float(cfg.get("recall_threshold", 0.80))
    model_name = cfg.get("nim_model_name", "meta/llama-3.1-70b-instruct")
    temperature = float(cfg.get("nim_temperature", 0.0))
    device = torch.device(cfg.get("device", "cpu"))
    checkpoint_path = "results/gnn_checkpoint.pt"

    api_key = os.getenv("NVIDIA_API_KEY", "")
    if not api_key or api_key == "your_nvidia_api_key_here":
        raise EnvironmentError(
            "NVIDIA_API_KEY not set or is a placeholder. "
            "Add your real key to the .env file in the project root."
        )

    print("=" * 100)
    print("NIM CLIENT TEST: 8 MACHINE-1-1 GROUND-TRUTH SEGMENTS")
    print(f"Model : {model_name}")
    print(f"Config: top_k={top_k}, min_topk_mass={min_topk_mass*100:.0f}%, min_sensor_std={min_sensor_std:.1e}, temperature={temperature}")
    print(f"Cache : {CACHE_PATH}")
    print("=" * 100)

    # Load dataset & model
    _, _, interp_labels, meta = load_dataset(
        dataset_name=cfg.get("dataset_name", "SMD"),
        raw_dir=cfg.get("dataset_path", "data/raw"),
        machine_id="machine-1-1",
        window_size=cfg.get("window_size", 100),
        stride=cfg.get("window_stride", 1),
        corr_threshold=cfg.get("edge_corr_threshold", 0.6),
    )

    gnn = GATAnomalyDetector(
        in_dim=6,
        hidden_dim=cfg.get("gnn_hidden_dim", 64),
        num_heads=cfg.get("gnn_num_heads", 4),
        edge_dim=1,
        dropout=0.2,
    ).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        gnn.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded 6-D GNN checkpoint (epoch {ckpt.get('epoch')}).\n")
    gnn.eval()

    norm_test = meta["norm_test"]
    raw_test = meta["raw_test"]
    norm_stats = meta["norm_stats"]
    edge_index = meta["edge_index"]
    edge_attr = meta["edge_attr"]

    results: List[Dict] = []

    for idx, seg in enumerate(interp_labels, 1):
        start, end = seg["start"], seg["end"]
        gt_dims = seg["dimensions"]

        start_idx = max(0, int(start - fine_margin))
        end_idx = min(len(norm_test), int(end + fine_margin) + 1)
        fine_norm_win = norm_test[start_idx:end_idx]
        fine_raw_win = raw_test[start_idx:end_idx]

        # Build 6-D PyG graph
        node_feats = compute_node_features_6d(fine_norm_win)
        x_ten = torch.tensor(node_feats, dtype=torch.float32, device=device)
        ei_ten = torch.tensor(edge_index, dtype=torch.long, device=device)
        ea_ten = (
            torch.tensor(edge_attr, dtype=torch.float32, device=device)
            if edge_attr is not None else None
        )
        data_sample = PyGData(x=x_ten, edge_index=ei_ten, edge_attr=ea_ten)

        # Integrated Gradients
        ig_scores = compute_integrated_gradients_attribution(gnn, data_sample, steps=50)

        # Prompt builder
        pb_out = build_decoupled_explanation_prompt(
            anomaly_id=idx,
            timesteps=(start, end),
            fine_raw_window=fine_raw_win,
            ig_scores=ig_scores,
            norm_stats=norm_stats,
            top_k=top_k,
            min_topk_mass=min_topk_mass,
            min_sensor_std=min_sensor_std,
            alpha=alpha,
            target_recall=recall_thresh,
            pooled_coverage_rate=93.0,
            n_cal_segments=327,
        )

        # NIM call (or pass-through)
        nim_result = call_nim(
            prompt_builder_output=pb_out,
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=600,
            cache_path=CACHE_PATH,
            verbose=True,
        )

        nim_result["gt_dims"] = gt_dims
        results.append(nim_result)

    # ---- Print detailed results for real API calls ----
    print("\n" + "=" * 100)
    print("DETAILED RESULTS FOR SEGMENTS WHERE llm_called=True")
    print("=" * 100)
    for r in results:
        if not r["llm_called"]:
            continue
        resp = r["llm_response"]
        print(f"\n--- Segment {r['segment_id']} (Timesteps {r['timesteps'][0]}-{r['timesteps'][1]}) ---")
        print(f"  llm_called          : {r['llm_called']}  |  cache_hit: {r.get('cache_hit', False)}")
        print(f"  filtered_mass       : {r['filtered_mass']*100:.2f}%")
        predicted = resp.get("root_cause_sensors", [])
        print(f"  root_cause_sensors  : {predicted}")
        print(f"  abstained (LLM)     : {resp.get('abstained', '?')}")
        gt = r.get("gt_dims", [])
        if isinstance(predicted, list) and gt:
            overlap = set(predicted).intersection(set(gt))
            print(f"  GT sensors          : {gt}")
            print(f"  LLM/GT overlap      : {sorted(overlap)} ({len(overlap)}/{len(gt)} = {len(overlap)/len(gt)*100:.1f}% recall)")
        if resp.get("parse_error"):
            print(f"  [WARN] JSON parse failed — raw response preview:")
            print(f"    {resp.get('raw_response', '')[:300]}")
        else:
            expl = resp.get("explanation", "")
            conf_stmt = resp.get("confidence_statement", "")
            print(f"  explanation         :\n{expl}")
            print(f"  confidence_statement:\n{conf_stmt}")
            is_clean = not resp.get("parse_error", False)
            print(f"  clean_json_verified : {is_clean}")

    # ---- Final summary table ----
    print("\n" + "=" * 100)
    print("FINAL SUMMARY TABLE")
    print("=" * 100)
    print(
        f"{'Seg':<4} | {'Timesteps':<15} | {'LLM Called':<10} | {'Abstain Code':<38} | "
        f"{'Predicted Sensors':<24} | GT Recall"
    )
    print("-" * 100)
    for r in results:
        resp = r["llm_response"]
        predicted = resp.get("root_cause_sensors", []) if isinstance(resp.get("root_cause_sensors"), list) else []
        gt = r.get("gt_dims", [])
        recall_str = (
            f"{len(set(predicted).intersection(set(gt)))/len(gt)*100:.1f}%"
            if gt and isinstance(predicted, list) and len(predicted) > 0 else "N/A"
        )
        ts = r["timesteps"]
        code = r.get("abstain_reason_code") or "N/A"
        print(
            f"{r['segment_id']:<4} | {ts[0]}-{ts[1]:<9} | {'Yes' if r['llm_called'] else 'No':<10} | "
            f"{code:<38} | {str(predicted):<24} | {recall_str}"
        )
    print("=" * 100)

    n_called = sum(1 for r in results if r["llm_called"])
    n_passthrough = sum(1 for r in results if not r["llm_called"])
    print(f"\nTotal: {len(results)} | LLM called: {n_called} | Pass-through abstentions: {n_passthrough}")
    print(f"Cache written to: {CACHE_PATH}\n")


if __name__ == "__main__":
    run_nim_on_machine_1_1()
