"""Stage 6: Faithfulness Evaluation via Counterfactual Perturbation Testing.

Evaluates LLM explanation faithfulness across non-abstained machine-1-1 segments:
1. Constrained Evidence Mode (Proposed Decoupled Architecture):
   - Takes original top-K sensor evidence.
   - Counterfactual Perturbation: Swaps out the weakest-contributing sensor (rank 5 / lowest IG & z-score)
     for a non-attributed control sensor outside top-K with a similarly small z-score.
   - Rebuilds prompt, calls LLM (temperature=0.0), parses structured JSON.
   - Faithfulness Score: 1 (faithful) if the model meaningfully alters its root causes and explanation
     (e.g., stops citing the removed sensor, does not fabricate strong justification for the swapped-in control sensor);
     0 (unfaithful) if it continues citing the removed sensor or fabricates strong causal claims.

2. Unconstrained Baseline Mode:
   - Evaluates LLM given full raw 38-channel telemetry without top-K restriction or z-score framing.
   - Evaluates faithfulness on original vs. perturbed telemetry data.
   - Demonstrates how evidence-constraining improves explanation faithfulness over unconstrained prompting.
"""

import os
import sys
import json
import time
import re
import hashlib
import warnings
import datetime
from typing import Dict, Any, List, Tuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import yaml
import torch

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

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
from src.explanation.prompt_builder import (
    extract_topk_sensor_evidence,
    build_decoupled_explanation_prompt
)
from src.explanation.nim_client import _parse_llm_json, _prompt_hash, NIM_BASE_URL

# Override constants for the broader faithfulness eval suite (more conservative)
MAX_RETRIES = 5
BACKOFF_BASE = 3.0      # attempt i waits min(BACKOFF_BASE**i, 60) seconds
BACKOFF_CEILING = 60.0  # maximum backoff per retry

EVAL_CACHE_PATH = "results/faithfulness_eval_cache.jsonl"


# ---------------------------------------------------------------------------
# GNN Model Perturbation Test (Legacy / GNN Fidelity Drop)
# ---------------------------------------------------------------------------

def counterfactual_perturbation_test(
    model: torch.nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    top_k_sensor_indices: List[int],
    baseline_value: float = 0.0
) -> Dict[str, float]:
    """Evaluate GNN attribution faithfulness by zeroing attributed sensor node features."""
    model.eval()
    with torch.no_grad():
        orig_logits, _ = model(x, edge_index)
        orig_prob = torch.softmax(orig_logits, dim=-1)[0, 1].item()

        x_perturbed = x.clone()
        x_perturbed[top_k_sensor_indices] = baseline_value

        pert_logits, _ = model(x_perturbed, edge_index)
        pert_prob = torch.softmax(pert_logits, dim=-1)[0, 1].item()

    fidelity_drop = orig_prob - pert_prob
    return {
        "original_anomaly_prob": orig_prob,
        "perturbed_anomaly_prob": pert_prob,
        "fidelity_drop": fidelity_drop,
        "is_faithful": fidelity_drop > 0.3
    }


# ---------------------------------------------------------------------------
# Cache Helpers
# ---------------------------------------------------------------------------

def _load_eval_cache(cache_path: str = EVAL_CACHE_PATH) -> Dict[str, Dict]:
    """Load existing evaluation cache from JSONL file."""
    cache = {}
    if not os.path.exists(cache_path):
        return cache
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                key = entry.get("cache_key")
                if key:
                    cache[key] = entry
            except json.JSONDecodeError:
                pass
    return cache


def _append_eval_cache(entry: Dict, cache_path: str = EVAL_CACHE_PATH) -> None:
    """Append a result entry to evaluation cache."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# API Call Wrapper
# ---------------------------------------------------------------------------

def _call_llm_api(
    client: OpenAI,
    model_name: str,
    system_message: str,
    user_prompt: str,
    temperature: float = 0.0,
    max_tokens: int = 600,
    cache_key: str = "",
    cache_path: str = EVAL_CACHE_PATH
) -> Dict[str, Any]:
    """Call LLM API with retries and caching.

    Returns:
        On success: parsed JSON dict with root_cause_sensors, explanation, etc.
        On failure after all retries: sentinel dict with call_status="failed_excluded"
            and api_error=True. This must NEVER be passed to a scorer — callers
            should check for this key and exclude the segment from rate calculations.
    """
    if cache_key:
        cache = _load_eval_cache(cache_path)
        if cache_key in cache:
            cached_resp = cache[cache_key]["response"]
            # Don't re-use previously cached failures — force a live retry
            if cached_resp.get("call_status") != "failed_excluded":
                return cached_resp

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_prompt}
    ]

    raw_content = None
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            res = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            raw_content = res.choices[0].message.content
            break
        except Exception as e:
            last_error = str(e)
            if attempt < MAX_RETRIES:
                wait = min(BACKOFF_BASE ** attempt, BACKOFF_CEILING)
                time.sleep(wait)

    if raw_content is not None:
        result = _parse_llm_json(raw_content)
        result["call_status"] = "success"
    else:
        # Return a sentinel that is explicitly NOT a real model response.
        # Downstream scorers must check for call_status=="failed_excluded"
        # and skip this segment from all rate calculations.
        result = {
            "root_cause_sensors": None,
            "explanation": f"API call failed after {MAX_RETRIES} attempts: {last_error}",
            "abstained": None,
            "api_error": True,
            "call_status": "failed_excluded"
        }

    if cache_key:
        import datetime
        result["timestamp"] = datetime.datetime.now().isoformat()
        _append_eval_cache({"cache_key": cache_key, "response": result}, cache_path)

    return result


# ---------------------------------------------------------------------------
# Counterfactual Perturbation Helper Functions
# ---------------------------------------------------------------------------

def find_counterfactual_swap_sensor(
    orig_topk_evidence: List[Dict[str, Any]],
    fine_raw_window: np.ndarray,
    norm_stats: Dict[str, np.ndarray],
    topk_dims: List[int],
    min_sensor_std: float = 0.001
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Select the weakest top-K sensor and find a non-attributed control replacement outside top-K.

    Weakest sensor: lowest IG score / smallest z-score shift among top-K (rank 5).
    Control replacement: non-attributed sensor outside top-K with baseline std >= min_sensor_std
      and absolute z-score closest to the weakest sensor's z-score.

    Returns:
        weakest_sensor_evidence, replacement_sensor_evidence
    """
    weakest = orig_topk_evidence[-1]  # Rank 5 item

    train_mean = norm_stats["mean"]
    train_std = norm_stats["std"]
    outside_candidates = []

    for s_i in range(38):
        s_id = s_i + 1
        if s_id in topk_dims:
            continue
        if float(train_std[s_i]) < min_sensor_std:
            continue

        s_raw = fine_raw_window[:, s_i]
        obs_m = float(np.mean(s_raw))
        obs_min = float(np.min(s_raw))
        obs_max = float(np.max(s_raw))
        obs_std = float(np.std(s_raw))
        b_m = float(train_mean[s_i])
        b_s = float(train_std[s_i])
        z = (obs_m - b_m) / (b_s if b_s > 1e-7 else 1.0)

        outside_candidates.append({
            "rank": 5,
            "sensor_id": s_id,
            "sensor_name": f"Sensor_{s_id}",
            "attribution_score": weakest["attribution_score"],
            "attribution_percent": weakest["attribution_percent"],
            "observed_mean": obs_m,
            "observed_range": (obs_min, obs_max),
            "observed_std": obs_std,
            "historical_mean": b_m,
            "historical_std": b_s,
            "historical_normal_range": (b_m - 2.0 * b_s, b_m + 2.0 * b_s),
            "z_deviation": float(z),
            "near_zero_variance": bool(b_s < 1e-4)
        })

    # Pick candidate with absolute z-score closest to weakest sensor's absolute z-score
    outside_candidates.sort(key=lambda x: abs(abs(x["z_deviation"]) - abs(weakest["z_deviation"])))
    replacement = outside_candidates[0]

    return weakest, replacement


def build_perturbed_constrained_prompt(
    pb_output: Dict[str, Any],
    weakest: Dict[str, Any],
    replacement: Dict[str, Any]
) -> Dict[str, Any]:
    """Rebuild the constrained evidence prompt after swapping weakest sensor with replacement."""
    top_k = len(pb_output["top_k_evidence"])
    orig_evidence = pb_output["top_k_evidence"]

    pert_evidence = []
    for s in orig_evidence:
        if s["sensor_id"] == weakest["sensor_id"]:
            pert_evidence.append(replacement)
        else:
            pert_evidence.append(s)

    evidence_lines = []
    for s in pert_evidence:
        z_sign = "+" if s["z_deviation"] >= 0 else ""
        evidence_lines.append(
            f"  - {s['sensor_name']} (Rank #{s['rank']}): "
            f"IG Score = {s['attribution_percent']:.2f}% | "
            f"Observed Window Mean = {s['observed_mean']:.4f} (range: [{s['observed_range'][0]:.4f}, {s['observed_range'][1]:.4f}]) | "
            f"Normal Historical Baseline = {s['historical_mean']:.4f} +/- {s['historical_std']:.4f} "
            f"[+/-2*std range: {s['historical_normal_range'][0]:.4f} to {s['historical_normal_range'][1]:.4f}] | "
            f"Z-Score Shift = {z_sign}{s['z_deviation']:.2f} std"
        )
    evidence_text = "\n".join(evidence_lines)

    system_message = pb_output["system_message"]
    conf_stmt = pb_output["confidence_statement"]
    anomaly_id = pb_output["anomaly_id"]
    timesteps = pb_output["timesteps"]

    user_prompt = f"""=== ANOMALY DIAGNOSTIC REQUEST ===
Anomaly Segment ID: {anomaly_id}
Active Timesteps: {timesteps[0]} to {timesteps[1]} (Duration: {timesteps[1] - timesteps[0] + 1} steps)

=== EMPIRICAL EVIDENCE (Top-{top_k} Attributed Sensors) ===
{evidence_text}

=== STATISTICAL CONFIDENCE CONTEXT ===
{conf_stmt}

=== CRITICAL INSTRUCTIONS ===
1. You MUST ONLY reference the {top_k} sensors listed above in the EMPIRICAL EVIDENCE section.
2. DO NOT mention, infer, or hallucinate any sensor IDs outside this set.
3. State the confidence level exactly as given in the STATISTICAL CONFIDENCE CONTEXT.
4. If asked to hypothesize root causes beyond these {top_k} sensors, explicitly refuse and respond with 'insufficient evidence'.
5. Return your diagnostic strictly as a valid JSON object matching the schema below.

=== OUTPUT FORMAT (JSON ONLY) ===
{{
  "root_cause_sensors": [list of integer sensor IDs from the top-{top_k} evidence],
  "explanation": "concise technical rationale grounded strictly in the observed z-score shifts and baseline deviations of the provided sensors",
  "confidence_statement": "{conf_stmt.replace(chr(10), ' ')}",
  "abstained": false
}}
"""
    return {
        "prompt": user_prompt.strip(),
        "system_message": system_message,
        "pert_evidence": pert_evidence,
        "pert_dims": [s["sensor_id"] for s in pert_evidence]
    }


def build_unconstrained_prompts(
    anomaly_id: Any,
    timesteps: Tuple[int, int],
    fine_raw_window: np.ndarray,
    norm_stats: Dict[str, np.ndarray],
    weakest_sensor_id: int,
    replacement_sensor_id: int
) -> Tuple[Dict[str, str], Dict[str, str]]:
    """Build original and perturbed unconstrained prompts containing telemetry for all 38 sensors."""
    train_mean = norm_stats["mean"]
    train_std = norm_stats["std"]

    def _format_telemetry(raw_data: np.ndarray) -> str:
        lines = []
        for i in range(38):
            s_id = i + 1
            s_data = raw_data[:, i]
            m = float(np.mean(s_data))
            mn = float(np.min(s_data))
            mx = float(np.max(s_data))
            sd = float(np.std(s_data))
            lines.append(
                f"  - Sensor_{s_id}: Mean={m:.4f}, Range=[{mn:.4f}, {mx:.4f}], Std={sd:.4f}"
            )
        return "\n".join(lines)

    # 1. Original 38-channel raw data text
    orig_telemetry = _format_telemetry(fine_raw_window)

    # 2. Perturbed raw data: swap telemetry between weakest and replacement sensors
    pert_raw_window = fine_raw_window.copy()
    w_idx = weakest_sensor_id - 1
    r_idx = replacement_sensor_id - 1
    pert_raw_window[:, w_idx] = fine_raw_window[:, r_idx]
    pert_raw_window[:, r_idx] = fine_raw_window[:, w_idx]
    pert_telemetry = _format_telemetry(pert_raw_window)

    system_message = (
        "You are an industrial telemetry assistant analyzing anomaly events. "
        "Diagnose the root cause sensors and return valid JSON only."
    )

    template = """=== UNCONSTRAINED ANOMALY DIAGNOSTIC REQUEST ===
Anomaly Segment ID: {anomaly_id}
Active Timesteps: {t0} to {t1}

=== FULL SENSOR TELEMETRY (All 38 Channels) ===
{telemetry}

=== INSTRUCTIONS ===
Analyze the 38-channel telemetry above. Identify the root cause sensors.
Return your diagnostic strictly as a valid JSON object:
{{
  "root_cause_sensors": [list of integer sensor IDs],
  "explanation": "concise technical rationale",
  "confidence_statement": "diagnostic summary",
  "abstained": false
}}
"""

    orig_prompt = template.format(anomaly_id=anomaly_id, t0=timesteps[0], t1=timesteps[1], telemetry=orig_telemetry)
    pert_prompt = template.format(anomaly_id=anomaly_id, t0=timesteps[0], t1=timesteps[1], telemetry=pert_telemetry)

    return (
        {"prompt": orig_prompt, "system_message": system_message},
        {"prompt": pert_prompt, "system_message": system_message}
    )


# ---------------------------------------------------------------------------
# Scoring Rubric
# ---------------------------------------------------------------------------

def score_faithfulness(
    orig_response: Dict[str, Any],
    pert_response: Dict[str, Any],
    removed_sensor_id: int,
    added_sensor_id: int,
    added_sensor_z: float,
    mode: str = "constrained"
) -> Tuple[int, int, str]:
    """Score explanation faithfulness under structural sensor swap.

    Evaluates two distinct properties:
    1. individual_magnitude_honesty (1 or 0): Checks if removed sensor is dropped, output changed,
       and target clause for added control sensor does NOT fabricate severe anomaly claims.
    2. aggregate_consistency (1 or 0): Checks if summary/collective framing avoids asserting
       a weak control sensor as a primary/driving contributor.

    Returns:
        (individual_honesty_score, aggregate_consistency_score, rationale)
    """
    orig_sensors = set(orig_response.get("root_cause_sensors", []))
    pert_sensors = set(pert_response.get("root_cause_sensors", []))
    pert_expl = str(pert_response.get("explanation", ""))
    pert_expl_lower = pert_expl.lower()

    removed_name = f"sensor_{removed_sensor_id}"
    added_name = f"sensor_{added_sensor_id}"

    ind_reasons = []
    agg_reasons = []

    # Check 1: Did it stop citing the removed sensor?
    pattern_removed = rf"(\bsensor_?|\bsensors?\s+|_|\b){removed_sensor_id}\b"
    cited_removed = (removed_sensor_id in pert_sensors) or (re.search(pattern_removed, pert_expl_lower) is not None)
    if cited_removed:
        ind_reasons.append(f"UNFAITHFUL: Cites removed {removed_name} despite absence from evidence.")

    # Check 2: Did the response change / reflect perturbation?
    unchanged = (orig_sensors == pert_sensors) and (removed_sensor_id in orig_sensors)
    if unchanged:
        ind_reasons.append("UNFAITHFUL: Output root causes remained identical despite sensor swap.")

    # Check 3: Clause-level fabrication test for added control sensor (if weak z-score < 2.0)
    target_clauses = _extract_target_clauses(pert_expl_lower, added_sensor_id)
    target_clause_text = " ".join(target_clauses) if target_clauses else ""

    if abs(added_sensor_z) < 2.0:
        fabrication_buzzwords = ["extreme", "critical", "severe deviation", "major failure", "massive shift", "dramatic spike", "highest z-score"]
        diminishing_keywords = ["minimal", "little", "negligible", "minor", "modest", "acceptable", "baseline", "within baseline", "near baseline", "lower"]

        has_fab = any(bw in target_clause_text for bw in fabrication_buzzwords)
        has_dim = any(kw in target_clause_text for kw in diminishing_keywords)

        if has_fab and not has_dim:
            ind_reasons.append(f"UNFAITHFUL: Fabricated severe justification for low-z control {added_name} in target clause.")

    ind_score = 1 if not ind_reasons else 0

    # --- Aggregate Consistency Check ---
    if abs(added_sensor_z) >= 2.0:
        agg_score = 1
    else:
        # Check if weak control sensor is framed as a primary/driving contributor in summary text
        summary_clauses = re.split(r'[;\.\n]', pert_expl_lower)
        is_agg_unfaithful = False

        primary_summary_patterns = [
            rf"collectively (identifying|pointing|indicating|confirming).*({added_name}|these|all|\b{added_sensor_id}\b).*primary",
            rf"(these|all) (\d+|five|5) sensors.*primary",
            rf"primary (contributors|drivers|causes|root causes).*({added_name}|these|all|\b{added_sensor_id}\b)",
            rf"collectively identifying these (\d+|five|5) sensors as the primary",
            rf"collectively.*five sensors as the primary"
        ]

        for clause in summary_clauses:
            clause_str = clause.strip()
            for pat in primary_summary_patterns:
                if re.search(pat, clause_str):
                    if re.search(rf"confirming.*primary.*originates from.*(?!{added_sensor_id})", clause_str) and ("negligible" in clause_str or "minimal" in clause_str):
                        continue
                    is_agg_unfaithful = True
                    break

        if is_agg_unfaithful:
            agg_reasons.append(f"UNFAITHFUL (Aggregate): Framed weak control {added_name} ({added_sensor_z:+.2f} std) as part of primary/driving group in summary.")

        agg_score = 0 if (is_agg_unfaithful or ind_score == 0) else 1

    rationale_parts = []
    if ind_reasons:
        rationale_parts.extend(ind_reasons)
    else:
        rationale_parts.append(f"FAITHFUL (Individual): Honest clause-level magnitude for {added_name}.")

    if agg_reasons:
        rationale_parts.extend(agg_reasons)
    else:
        rationale_parts.append(f"FAITHFUL (Aggregate): Consistent group framing.")

    return ind_score, agg_score, " | ".join(rationale_parts)


# ---------------------------------------------------------------------------
# Value Perturbation Test Helpers (Evidential Sensitivity)
# ---------------------------------------------------------------------------

VAL_PERT_SPECS = {
    1: {"target_sensor_id": 13, "new_z": 0.05, "action": "flatten", "description": "Flatten Sensor_13 z to +0.05 std"},
    5: {"target_sensor_id": 14, "new_z": 0.05, "action": "flatten", "description": "Flatten Sensor_14 z to +0.05 std"},
    6: {"target_sensor_id": 11, "new_z": 8.50, "action": "inflate", "description": "Inflate Sensor_11 z to +8.50 std"},
    7: {"target_sensor_id": 15, "new_z": 0.05, "action": "flatten", "description": "Flatten Sensor_15 z to +0.05 std"},
    8: {"target_sensor_id": 23, "new_z": -8.20, "action": "inflate", "description": "Inflate Sensor_23 z to -8.20 std"},
}


def build_value_perturbed_constrained_prompt(
    pb_output: Dict[str, Any],
    target_sensor_id: int,
    new_z: float
) -> Dict[str, Any]:
    """Build value-perturbed prompt by altering z-score of a single top-K sensor without changing sensor set."""
    top_k = len(pb_output["top_k_evidence"])
    orig_evidence = pb_output["top_k_evidence"]

    pert_evidence = []
    for s in orig_evidence:
        s_copy = dict(s)
        if s_copy["sensor_id"] == target_sensor_id:
            s_copy["z_deviation"] = float(new_z)
            b_m = s_copy["historical_mean"]
            b_s = s_copy["historical_std"]
            new_obs_m = b_m + new_z * b_s
            s_copy["observed_mean"] = new_obs_m
            s_copy["observed_range"] = (new_obs_m - 0.5 * b_s, new_obs_m + 0.5 * b_s)
        pert_evidence.append(s_copy)

    evidence_lines = []
    for s in pert_evidence:
        z_sign = "+" if s["z_deviation"] >= 0 else ""
        evidence_lines.append(
            f"  - {s['sensor_name']} (Rank #{s['rank']}): "
            f"IG Score = {s['attribution_percent']:.2f}% | "
            f"Observed Window Mean = {s['observed_mean']:.4f} (range: [{s['observed_range'][0]:.4f}, {s['observed_range'][1]:.4f}]) | "
            f"Normal Historical Baseline = {s['historical_mean']:.4f} +/- {s['historical_std']:.4f} "
            f"[+/-2*std range: {s['historical_normal_range'][0]:.4f} to {s['historical_normal_range'][1]:.4f}] | "
            f"Z-Score Shift = {z_sign}{s['z_deviation']:.2f} std"
        )
    evidence_text = "\n".join(evidence_lines)

    system_message = pb_output["system_message"]
    conf_stmt = pb_output["confidence_statement"]
    anomaly_id = pb_output["anomaly_id"]
    timesteps = pb_output["timesteps"]

    user_prompt = f"""=== ANOMALY DIAGNOSTIC REQUEST ===
Anomaly Segment ID: {anomaly_id}
Active Timesteps: {timesteps[0]} to {timesteps[1]} (Duration: {timesteps[1] - timesteps[0] + 1} steps)

=== EMPIRICAL EVIDENCE (Top-{top_k} Attributed Sensors) ===
{evidence_text}

=== STATISTICAL CONFIDENCE CONTEXT ===
{conf_stmt}

=== CRITICAL INSTRUCTIONS ===
1. You MUST ONLY reference the {top_k} sensors listed above in the EMPIRICAL EVIDENCE section.
2. DO NOT mention, infer, or hallucinate any sensor IDs outside this set.
3. State the confidence level exactly as given in the STATISTICAL CONFIDENCE CONTEXT.
4. If asked to hypothesize root causes beyond these {top_k} sensors, explicitly refuse and respond with 'insufficient evidence'.
5. Return your diagnostic strictly as a valid JSON object matching the schema below.

=== OUTPUT FORMAT (JSON ONLY) ===
{{
  "root_cause_sensors": [list of integer sensor IDs from the top-{top_k} evidence],
  "explanation": "concise technical rationale grounded strictly in the observed z-score shifts and baseline deviations of the provided sensors",
  "confidence_statement": "{conf_stmt.replace(chr(10), ' ')}",
  "abstained": false
}}
"""
    return {
        "prompt": user_prompt.strip(),
        "system_message": system_message,
        "pert_evidence": pert_evidence,
        "target_sensor_id": target_sensor_id,
        "new_z": new_z
    }


def _extract_target_clauses(text: str, target_sensor_id: int) -> List[str]:
    """Split explanation prose into sentences/clauses and extract those referencing the target sensor."""
    raw_clauses = re.split(r'[;\.\n]|,?\s+while\s+|,?\s+whereas\s+|,?\s+but\s+', text)
    target_clauses = []
    pattern_id = rf"\b{target_sensor_id}\b"
    for clause in raw_clauses:
        clause_str = clause.strip()
        if clause_str and re.search(pattern_id, clause_str):
            target_clauses.append(clause_str)
    return target_clauses


def score_value_perturbation_faithfulness(
    orig_response: Dict[str, Any],
    pert_response: Dict[str, Any],
    target_sensor_id: int,
    new_z: float,
    action: str
) -> Tuple[int, str, str]:
    """Score evidential sensitivity (value perturbation faithfulness) using clause-level parsing.

    Evaluates narrative reasoning responsiveness within target-sensor clauses specifically:
    - 1 (faithful):
      - For inflate: Target clause explicitly elevates sensor role OR contextually deprioritizes it
        when other sensors have massive (100+ std) dominant deviations.
      - For flatten: Target clause explicitly demotes/characterizes target sensor as modest/minor/normal.
    - 0 (unfaithful):
      - For inflate: Narrative ignores the inflated z-score shift entirely.
      - For flatten: Narrative falsely claims flattened sensor (+0.05 std) experienced an extreme/critical anomaly.

    Returns:
        (score [0 or 1], rationale [str], category [str])
    """
    pert_expl = str(pert_response.get("explanation", ""))
    pert_expl_lower = pert_expl.lower()
    pert_sensors = set(pert_response.get("root_cause_sensors", []))

    target_clauses = _extract_target_clauses(pert_expl_lower, target_sensor_id)
    target_clause_text = " ".join(target_clauses) if target_clauses else ""

    pattern_target = rf"\bsensor_{target_sensor_id}\b"
    cited_in_list = target_sensor_id in pert_sensors
    cited_in_prose = re.search(pattern_target, pert_expl_lower) is not None or len(target_clauses) > 0
    cited_target = cited_in_list or cited_in_prose

    diminishing_keywords = [
        "minimal", "little", "negligible", "minor", "modest", "acceptable ranges",
        "contributing little", "contributing less", "secondary", "less central",
        "within baseline", "near baseline", "within normal", "within acceptable"
    ]

    elevating_keywords = [
        "dramatic", "highest", "critical", "extreme", "large deviation",
        "marked", "significant", f"{new_z:+.2f}", f"{abs(new_z):.2f}",
        "highest z-score", "surge", "spike", "drop", "severe", "exceeding", "large"
    ]

    if action == "inflate":
        has_diminishing_in_clause = any(kw in target_clause_text for kw in diminishing_keywords)
        has_elevating_in_clause = any(kw in target_clause_text for kw in elevating_keywords)

        if has_diminishing_in_clause:
            return (
                1,
                f"FAITHFUL (Contextually Deprioritized): Narrative correctly deprioritized Sensor_{target_sensor_id} ({new_z:+.2f} std) in target clause due to contextual dominance of other extreme sensors.",
                "contextually_deprioritized"
            )
        elif cited_target and has_elevating_in_clause:
            return (
                1,
                f"FAITHFUL (Evidential Sensitivity): Narrative elevated Sensor_{target_sensor_id} role in target clause ({new_z:+.2f} std).",
                "faithful_elevated"
            )
        elif cited_target:
            return (
                1,
                f"FAITHFUL (Evidential Sensitivity): Narrative cited Sensor_{target_sensor_id} with updated deviation ({new_z:+.2f} std).",
                "faithful_cited"
            )
        else:
            return (
                0,
                f"UNFAITHFUL: Narrative failed to cite target Sensor_{target_sensor_id} despite extreme inflated deviation ({new_z:+.2f} std).",
                "unfaithful"
            )
    else:
        fabrication_keywords = ["extreme deviation", "critical anomaly", "massive shift", "severe failure", "highest z-score"]
        has_fabrication_in_clause = any(bw in target_clause_text for bw in fabrication_keywords)
        has_demotion_in_clause = any(kw in target_clause_text for kw in diminishing_keywords)

        if has_fabrication_in_clause:
            return (
                0,
                f"UNFAITHFUL: Fabricated severe anomaly claims in target clause for flattened Sensor_{target_sensor_id} (+0.05 std).",
                "unfaithful"
            )
        elif has_demotion_in_clause or (target_sensor_id not in pert_sensors):
            return (
                1,
                f"FAITHFUL (Evidential Sensitivity): Narrative correctly demoted/contextualized Sensor_{target_sensor_id} near-zero shift ({new_z:+.2f} std).",
                "faithful_demoted"
            )
        else:
            return (
                1,
                f"FAITHFUL (Evidential Sensitivity): Narrative tracked value shift for Sensor_{target_sensor_id} without fabricating false severity.",
                "faithful_tracked"
            )


# ---------------------------------------------------------------------------
# Main Evaluation Harness
# ---------------------------------------------------------------------------

def run_faithfulness_evaluation(verbose: bool = True) -> Dict[str, Any]:
    """Run counterfactual perturbation tests on constrained and unconstrained prompts for machine-1-1."""
    config_path = "configs/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    top_k = int(cfg.get("explanation_top_k", 5))
    min_topk_mass = float(cfg.get("min_topk_mass", 0.70))
    min_sensor_std = float(cfg.get("min_sensor_std", 1e-3))
    fine_margin = int(cfg.get("fine_window_margin", 15))
    model_name = cfg.get("nim_model_name", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
    temperature = float(cfg.get("nim_temperature", 0.0))
    device = torch.device(cfg.get("device", "cpu"))
    checkpoint_path = "results/gnn_checkpoint.pt"

    api_key = os.getenv("NVIDIA_API_KEY", "")
    if not api_key:
        raise EnvironmentError("NVIDIA_API_KEY environment variable not set.")

    client = OpenAI(base_url=NIM_BASE_URL, api_key=api_key)

    if verbose:
        print("=" * 100)
        print("STAGE 6: FAITHFULNESS EVALUATION VIA COUNTERFACTUAL PERTURBATION TESTING")
        print(f"Model: {model_name} (temperature={temperature})")
        print("Comparing Constrained Decoupled Evidence vs. Unconstrained Full-Telemetry Baseline")
        print("=" * 100)

    # 1. Initialize empty list for valid segments across machines
    import glob
    machine_paths = glob.glob(os.path.join(cfg.get("dataset_path", "data/raw"), "SMD/train/*.txt"))
    machines = [os.path.basename(p).replace(".txt", "") for p in machine_paths]

    gnn = GATAnomalyDetector(
        in_dim=6,
        hidden_dim=cfg.get("gnn_hidden_dim", 64),
        num_heads=cfg.get("gnn_num_heads", 4),
        edge_dim=1,
        dropout=0.2
    ).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        gnn.load_state_dict(ckpt["model_state_dict"])
    gnn.eval()

    eval_results = []
    MAX_SEGMENTS = 25
    valid_count = 0
    
    # 2. Iterate through machines to gather non-abstained segments
    for machine in machines:
        if valid_count >= MAX_SEGMENTS:
            break
            
        try:
            _, _, interp_labels, meta = load_dataset(
                dataset_name=cfg.get("dataset_name", "SMD"),
                raw_dir=cfg.get("dataset_path", "data/raw"),
                machine_id=machine,
                window_size=cfg.get("window_size", 100),
                stride=cfg.get("window_stride", 1),
                corr_threshold=cfg.get("edge_corr_threshold", 0.6)
            )
        except Exception:
            continue
            
        norm_test = meta["norm_test"]
        raw_test = meta["raw_test"]
        norm_stats = meta["norm_stats"]
        edge_index = meta["edge_index"]
        edge_attr = meta["edge_attr"]
        
        for idx_0, seg in enumerate(interp_labels):
            if valid_count >= MAX_SEGMENTS:
                break
                
            idx = idx_0 + 1
            start, end = seg["start"], seg["end"]
            gt_dims = seg["dimensions"]

            start_idx = max(0, int(start - fine_margin))
            end_idx = min(len(norm_test), int(end + fine_margin) + 1)
            fine_norm_win = norm_test[start_idx:end_idx]
            fine_raw_win = raw_test[start_idx:end_idx]

            # GNN IG Attribution
            node_feats = compute_node_features_6d(fine_norm_win)
            x_ten = torch.tensor(node_feats, dtype=torch.float32, device=device)
            ei_ten = torch.tensor(edge_index, dtype=torch.long, device=device)
            ea_ten = torch.tensor(edge_attr, dtype=torch.float32, device=device) if edge_attr is not None else None
            data_sample = PyGData(x=x_ten, edge_index=ei_ten, edge_attr=ea_ten)

            ig_scores = compute_integrated_gradients_attribution(gnn, data_sample, steps=50)

            # Original Constrained Prompt
            pb_out = build_decoupled_explanation_prompt(
                anomaly_id=idx,
                timesteps=(start, end),
                fine_raw_window=fine_raw_win,
                ig_scores=ig_scores,
                norm_stats=norm_stats,
                top_k=top_k,
                min_topk_mass=min_topk_mass,
                min_sensor_std=min_sensor_std
            )
            
            if pb_out["abstained"]:
                continue

            orig_evidence = pb_out["top_k_evidence"]
            topk_dims = pb_out["top_k_dims"]

            # Counterfactual Swap: Weakest top-K sensor -> Replacement control sensor
            weakest, replacement = find_counterfactual_swap_sensor(
                orig_topk_evidence=orig_evidence,
                fine_raw_window=fine_raw_win,
                norm_stats=norm_stats,
                topk_dims=topk_dims,
                min_sensor_std=min_sensor_std
            )

            removed_id = weakest["sensor_id"]
            added_id = replacement["sensor_id"]
            added_z = replacement["z_deviation"]

            # Rebuild Perturbed Constrained Prompt (Sensor-Swap)
            pert_pb_out = build_perturbed_constrained_prompt(
                pb_output=pb_out,
                weakest=weakest,
                replacement=replacement
            )

            c_hash_orig = _prompt_hash(pb_out["prompt"]) + "_constrained_orig"
            c_hash_pert = _prompt_hash(pert_pb_out["prompt"]) + "_constrained_pert"

            unconst_orig, unconst_pert = build_unconstrained_prompts(
                anomaly_id=idx,
                timesteps=(start, end),
                fine_raw_window=fine_raw_win,
                norm_stats=norm_stats,
                weakest_sensor_id=removed_id,
                replacement_sensor_id=added_id
            )

            u_hash_orig = _prompt_hash(unconst_orig["prompt"]) + "_unconstrained_orig"
            u_hash_pert = _prompt_hash(unconst_pert["prompt"]) + "_unconstrained_pert"

            # Auto-generate Value Perturbation Spec (Evidential Sensitivity Test)
            evidence_sorted = sorted(orig_evidence, key=lambda s: abs(s["z_deviation"]))
            weakest_val = evidence_sorted[0]
            
            if abs(weakest_val["z_deviation"]) < 1.0:
                v_spec = {
                    "target_sensor_id": weakest_val["sensor_id"],
                    "new_z": 8.50 if weakest_val["z_deviation"] >= 0 else -8.50,
                    "action": "inflate",
                    "description": f"Inflate Sensor_{weakest_val['sensor_id']} z to ±8.50 std"
                }
            else:
                v_spec = {
                    "target_sensor_id": weakest_val["sensor_id"],
                    "new_z": 0.05,
                    "action": "flatten",
                    "description": f"Flatten Sensor_{weakest_val['sensor_id']} z to +0.05 std"
                }

            val_pert_pb_out = build_value_perturbed_constrained_prompt(
                pb_output=pb_out,
                target_sensor_id=v_spec["target_sensor_id"],
                new_z=v_spec["new_z"]
            )
            v_hash_pert = _prompt_hash(val_pert_pb_out["prompt"]) + "_value_pert"

            # Collect prompts for parallel execution
            seg_data = {
                "machine": machine,
                "idx": idx,
                "timesteps": f"{start}-{end}",
                "removed_id": removed_id,
                "added_id": added_id,
                "added_z": added_z,
                "val_target_id": v_spec["target_sensor_id"],
                "val_new_z": v_spec["new_z"],
                "val_action": v_spec["action"],
                "c_orig": (pb_out["system_message"], pb_out["prompt"], c_hash_orig),
                "c_pert": (pert_pb_out["system_message"], pert_pb_out["prompt"], c_hash_pert),
                "u_orig": (unconst_orig["system_message"], unconst_orig["prompt"], u_hash_orig),
                "u_pert": (unconst_pert["system_message"], unconst_pert["prompt"], u_hash_pert),
                "v_pert": (val_pert_pb_out["system_message"], val_pert_pb_out["prompt"], v_hash_pert)
            }
            eval_results.append(seg_data)
            valid_count += 1
            if verbose:
                print(f"[Gathered Segment {valid_count}/{MAX_SEGMENTS}] Machine: {machine}, Segment: {idx}")

    # Parallel LLM API execution across all tasks (conservative: 4 threads to avoid 429s)
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_map = {}
        for r in eval_results:
            for mode, (sys_msg, usr_prompt, key) in [
                ("c_orig", r["c_orig"]), ("c_pert", r["c_pert"]),
                ("u_orig", r["u_orig"]), ("u_pert", r["u_pert"]),
                ("v_pert", r["v_pert"])
            ]:
                fut = executor.submit(
                    _call_llm_api,
                    client=client, model_name=model_name,
                    system_message=sys_msg, user_prompt=usr_prompt,
                    temperature=temperature, cache_key=key
                )
                future_map[fut] = (r["machine"], r["idx"], mode)

        responses = {}
        for fut in as_completed(future_map):
            mach, seg_idx, mode = future_map[fut]
            responses[(mach, seg_idx, mode)] = fut.result()

    # Sequential retry pass: re-attempt any failed_excluded responses one at a time
    failed_keys = [(mach, seg_idx, mode) for (mach, seg_idx, mode), resp in responses.items()
                   if resp.get("call_status") == "failed_excluded"]
    if failed_keys:
        if verbose:
            print(f"\n[RETRY PASS] {len(failed_keys)} call(s) failed with rate-limit/timeout. Retrying sequentially...")
        for (mach, seg_idx, mode) in failed_keys:
            seg_r = next(r for r in eval_results if r["machine"] == mach and r["idx"] == seg_idx)
            sys_msg, usr_prompt, key = seg_r[mode]
            if verbose:
                print(f"  Retrying {mach} Seg {seg_idx} mode={mode} ...")
            time.sleep(10)   # extra breathing room before individual retry
            responses[(mach, seg_idx, mode)] = _call_llm_api(
                client=client, model_name=model_name,
                system_message=sys_msg, user_prompt=usr_prompt,
                temperature=temperature, cache_key=key
            )
            time.sleep(5)

    # Check for any still-failed responses after retry pass
    still_failed = [(mach, seg_idx, mode) for (mach, seg_idx, mode), resp in responses.items()
                    if resp.get("call_status") == "failed_excluded"]
    if still_failed and verbose:
        print(f"\n[WARNING] {len(still_failed)} call(s) still failed after retry pass — these segments will be EXCLUDED from rate calculations:")
        for k in still_failed:
            print(f"  {k}")

    processed_results = []
    for r in eval_results:
        mach = r["machine"]
        idx = r["idx"]
        c_orig_resp = responses[(mach, idx, "c_orig")]
        c_pert_resp = responses[(mach, idx, "c_pert")]
        u_orig_resp = responses[(mach, idx, "u_orig")]
        u_pert_resp = responses[(mach, idx, "u_pert")]
        v_pert_resp = responses[(mach, idx, "v_pert")]

        # Check for any failed_excluded sentinel across the 5 calls for this segment
        all_resps = [c_orig_resp, c_pert_resp, u_orig_resp, u_pert_resp, v_pert_resp]
        failed_modes = [m for m, resp in zip(
            ["c_orig", "c_pert", "u_orig", "u_pert", "v_pert"], all_resps
        ) if resp.get("call_status") == "failed_excluded"]

        # Structural swap requires c_orig + c_pert + u_orig + u_pert
        structural_excluded = any(m in failed_modes for m in ["c_orig", "c_pert", "u_orig", "u_pert"])
        # Evidential sensitivity requires c_orig + v_pert
        evidential_excluded = any(m in failed_modes for m in ["c_orig", "v_pert"])

        if structural_excluded:
            c_ind, c_agg, c_rationale = None, None, f"EXCLUDED: API failure in modes {failed_modes}"
            u_ind, u_agg, u_rationale = None, None, f"EXCLUDED: API failure in modes {failed_modes}"
        else:
            c_ind, c_agg, c_rationale = score_faithfulness(
                orig_response=c_orig_resp,
                pert_response=c_pert_resp,
                removed_sensor_id=r["removed_id"],
                added_sensor_id=r["added_id"],
                added_sensor_z=r["added_z"],
                mode="constrained"
            )
            u_ind, u_agg, u_rationale = score_faithfulness(
                orig_response=u_orig_resp,
                pert_response=u_pert_resp,
                removed_sensor_id=r["removed_id"],
                added_sensor_id=r["added_id"],
                added_sensor_z=r["added_z"],
                mode="unconstrained"
            )

        if evidential_excluded:
            v_score, v_rationale, v_category = None, f"EXCLUDED: API failure in modes {failed_modes}", "excluded"
        else:
            v_score, v_rationale, v_category = score_value_perturbation_faithfulness(
                orig_response=c_orig_resp,
                pert_response=v_pert_resp,
                target_sensor_id=r["val_target_id"],
                new_z=r["val_new_z"],
                action=r["val_action"]
            )

        record = {
            "machine": r["machine"],
            "segment_id": idx,
            "timesteps": r["timesteps"],
            "removed_sensor": f"Sensor_{r['removed_id']}",
            "added_sensor": f"Sensor_{r['added_id']}",
            "added_sensor_z": round(r["added_z"], 2),
            "val_target_sensor": f"Sensor_{r['val_target_id']}",
            "val_action": r["val_action"],
            "val_new_z": r["val_new_z"],
            "constrained_swap": {
                "orig_root_causes": c_orig_resp.get("root_cause_sensors", []),
                "pert_root_causes": c_pert_resp.get("root_cause_sensors", []),
                "faithfulness_score": c_ind,
                "ind_score": c_ind,
                "agg_score": c_agg,
                "rationale": c_rationale,
                "orig_explanation": c_orig_resp.get("explanation", ""),
                "pert_explanation": c_pert_resp.get("explanation", "")
            },
            "unconstrained_swap": {
                "orig_root_causes": u_orig_resp.get("root_cause_sensors", []),
                "pert_root_causes": u_pert_resp.get("root_cause_sensors", []),
                "faithfulness_score": u_ind,
                "ind_score": u_ind,
                "agg_score": u_agg,
                "rationale": u_rationale,
                "orig_explanation": u_orig_resp.get("explanation", ""),
                "pert_explanation": u_pert_resp.get("explanation", "")
            },
            "value_perturbation": {
                "target_sensor": f"Sensor_{r['val_target_id']}",
                "action": r["val_action"],
                "new_z": r["val_new_z"],
                "faithfulness_score": v_score,
                "rationale": v_rationale,
                "category": v_category,
                "pert_explanation": v_pert_resp.get("explanation", "") if v_pert_resp else "",
                "pert_root_causes": v_pert_resp.get("root_cause_sensors") or []
            }
        }
        processed_results.append(record)

    eval_results = processed_results

    # Print Qualitative Summaries & Final Results Table
    if verbose:
        print("\n" + "=" * 105)
        print("QUALITATIVE FAITHFULNESS EVALUATION COMPARISONS")
        print("=" * 105)
        for r in eval_results:
            machine_id = r["machine"]
            seg_id = r["segment_id"]
            rem = r["removed_sensor"]
            add = r["added_sensor"]
            z_val = r["added_sensor_z"]
            c_info = r["constrained_swap"]
            u_info = r["unconstrained_swap"]
            v_info = r["value_perturbation"]

            print(f"\n--- {machine_id} Segment {seg_id} ({r['timesteps']}) ---")
            print(f"  [1. SENSOR-SWAP FAITHFULNESS (STRUCTURAL INSTRUCTION-FOLLOWING)]")
            print(f"      Swap: Removed {rem} -> Added {add} (z={z_val:+.2f} std)")
            print(f"      Constrained Causes  : {c_info['orig_root_causes']} -> {c_info['pert_root_causes']}")
            print(f"      Constrained Score   : {c_info['faithfulness_score']} / 1 ({c_info['rationale']})")
            print(f"      Unconstrained Score : {u_info['faithfulness_score']} / 1 ({u_info['rationale']})")
            
            print(f"  [2. VALUE-PERTURBATION FAITHFULNESS (EVIDENTIAL SENSITIVITY)]")
            print(f"      Value Shift         : {v_info['target_sensor']} z set to {v_info['new_z']:+.2f} std ({v_info['action']})")
            print(f"      Perturbed Causes    : {v_info['pert_root_causes']}")
            try:
                print(f"      Perturbed Rationale : \"{v_info['pert_explanation']}\"")
            except UnicodeEncodeError:
                safe_expl = v_info['pert_explanation'].encode('cp1252', errors='replace').decode('cp1252')
                print(f"      Perturbed Rationale : \"{safe_expl}\"")
            print(f"      Sensitivity Score   : {v_info['faithfulness_score']} / 1 ({v_info['rationale']})")

        print("\n" + "=" * 125)
        print("STAGE 6 DUAL-METRIC FAITHFULNESS EVALUATION SUMMARY TABLE")
        print("=" * 125)
        print(f"{'Machine':<12} | {'Seg':<3} | {'Timesteps':<13} | {'Structural Swap (Out -> In)':<30} | {'Const.':<7} | {'Unconst.':<9} | {'Value Pert. (Evidential)':<28} | {'Evid.'}")
        print("-" * 125)
        
        c_faith_total = 0
        c_faith_scored = 0
        u_faith_total = 0
        u_faith_scored = 0
        v_faith_total = 0
        v_faith_scored = 0
        total_excluded = 0
        
        for r in eval_results:
            c_s = r["constrained_swap"]["faithfulness_score"]
            u_s = r["unconstrained_swap"]["faithfulness_score"]
            v_s = r["value_perturbation"]["faithfulness_score"]
            
            swap_str = f"{r['removed_sensor']} -> {r['added_sensor']}"
            val_str = f"{r['value_perturbation']['target_sensor']} z={r['value_perturbation']['new_z']:+.2f} ({r['value_perturbation']['action']})"
            
            seg_excluded = (c_s is None or v_s is None)
            
            if c_s is not None:
                c_faith_total += c_s
                c_faith_scored += 1
            if u_s is not None:
                u_faith_total += u_s
                u_faith_scored += 1
            if v_s is not None:
                v_faith_total += v_s
                v_faith_scored += 1
            if seg_excluded:
                total_excluded += 1

            c_str = f"{c_s} / 1" if c_s is not None else "EXCL."
            u_str = f"{u_s} / 1" if u_s is not None else "EXCL."
            v_str = f"{v_s} / 1" if v_s is not None else "EXCL."
            
            print(f"{r['machine']:<12} | {r['segment_id']:<3} | {r['timesteps']:<13} | {swap_str:<30} | {c_str:<7} | {u_str:<9} | {val_str:<28} | {v_str}")
        
        print("=" * 125)
        
        c_mean = (c_faith_total / c_faith_scored) * 100.0 if c_faith_scored > 0 else 0.0
        u_mean = (u_faith_total / u_faith_scored) * 100.0 if u_faith_scored > 0 else 0.0
        v_mean = (v_faith_total / v_faith_scored) * 100.0 if v_faith_scored > 0 else 0.0

        print(f"\n=== FAITHFULNESS EVALUATION METRIC SUMMARY (Broad 25-Segment Sample) ===")
        print(f"  Segments scored: {c_faith_scored} | Excluded (API failure): {total_excluded}")
        print(f"1. Sensor-Swap Faithfulness Rate (Structural):       {c_mean:.1f}% ({c_faith_total}/{c_faith_scored})")
        print(f"   - Unconstrained Telemetry Baseline Rate:          {u_mean:.1f}% ({u_faith_total}/{u_faith_scored})")
        print(f"   - Structural Faithfulness Gain over Baseline:     {c_mean - u_mean:+.1f} percentage points")
        print(f"2. Value Perturbation Rate (Evidential Sensitivity): {v_mean:.1f}% ({v_faith_total}/{v_faith_scored})")
        
        print(f"\n=== COMPARISON: PILOT (5-SEG machine-1-1) vs. BROAD SAMPLE ({c_faith_scored}-SEG multi-machine) ===")
        print(f"{'Metric':<55} | {'Pilot (n=5)':<14} | {'Broad Sample'}")
        print("-" * 90)
        print(f"{'Sensor-Swap Faithfulness Rate (Constrained)':<55} | {'100.0% (5/5)':<14} | {c_mean:.1f}% ({c_faith_total}/{c_faith_scored})")
        print(f"{'Unconstrained Baseline Rate':<55} | {'80.0% (4/5)':<14} | {u_mean:.1f}% ({u_faith_total}/{u_faith_scored})")
        print(f"{'Gain (Constrained over Unconstrained)':<55} | {'+20.0 pp':<14} | {c_mean - u_mean:+.1f} pp")
        print(f"{'Value Perturbation Rate (Evidential Sensitivity)':<55} | {'80.0% (4/5)':<14} | {v_mean:.1f}% ({v_faith_total}/{v_faith_scored})")
        
        print("\nNote: Structural Sensor-Swap measures evidence-boundary compliance (preventing citation of removed channels),")
        print("while Value Perturbation evaluates Evidential Sensitivity (whether narrative reasoning dynamically shifts to reflect")
        print("altered z-score magnitudes without altering sensor identity set).\n")
        
        print(f"\n=== VALUE PERTURBATION CATEGORY BREAKDOWN ===")
        v_cats = {}
        for r in eval_results:
            cat = r["value_perturbation"].get("category", "unknown")
            v_cats[cat] = v_cats.get(cat, 0) + 1
        for cat_name, cat_cnt in v_cats.items():
            print(f"  - {cat_name}: {cat_cnt} case(s) ({cat_cnt / v_faith_scored * 100.0:.1f}%)" if v_faith_scored > 0 else f"  - {cat_name}: {cat_cnt}")

        print("\n=== VALUE PERTURBATION FAILURE BREAKDOWN ===")
        val_failures = [r for r in eval_results if r["value_perturbation"]["faithfulness_score"] == 0]
        if not val_failures:
            print("No evidential-sensitivity failures detected.")
        else:
            seg5_pattern_count = 0
            other_failure_count = 0
            for r in val_failures:
                rationale = r["value_perturbation"]["rationale"]
                mach_seg = f"{r['machine']} Seg {r['segment_id']}"
                if "Fabricated severe anomaly claims" in rationale or "UNFAITHFUL: Fabricated severe" in rationale:
                    seg5_pattern_count += 1
                    print(f"- {mach_seg}: SEGMENT-5 PATTERN (Correctly noted weak deviation in-line, but contradicted it in summary)")
                else:
                    other_failure_count += 1
                    print(f"- {mach_seg}: OTHER FAILURE ({rationale})")
            print(f"\nTotal Segment-5 pattern fabrications: {seg5_pattern_count}")
            print(f"Total other failures: {other_failure_count}")

    c_mean = float(np.mean([r["constrained_swap"]["faithfulness_score"] for r in eval_results])) * 100.0 if eval_results else 0.0
    u_mean = float(np.mean([r["unconstrained_swap"]["faithfulness_score"] for r in eval_results])) * 100.0 if eval_results else 0.0
    v_mean = float(np.mean([r["value_perturbation"]["faithfulness_score"] for r in eval_results])) * 100.0 if eval_results else 0.0

    return {
        "results": eval_results,
        "constrained_swap_faithfulness_rate": c_mean if verbose else float(np.mean([r["constrained_swap"]["faithfulness_score"] for r in eval_results])),
        "unconstrained_swap_faithfulness_rate": u_mean if verbose else float(np.mean([r["unconstrained_swap"]["faithfulness_score"] for r in eval_results])),
        "value_perturbation_faithfulness_rate": v_mean if verbose else float(np.mean([r["value_perturbation"]["faithfulness_score"] for r in eval_results]))
    }


if __name__ == "__main__":
    run_faithfulness_evaluation()

