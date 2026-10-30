"""Stage 4: Decoupled Evidence + Confidence Prompt Builder for Faithful LLM Root-Cause Explanation.

Implements a decoupled evidence + confidence architecture:
1. Evidence for LLM:
   - Top-K sensors by Integrated Gradients attribution score (configurable K, default 5).
   - Sensor ID (1-indexed and 0-indexed name).
   - Raw value statistics during the fine anomaly window (mean, min, max, std).
   - Normal baseline statistics from training (mean ± std).
2. Confidence Statement:
   - Cumulative IG mass captured by top-K (instance-level signal concentration).
   - Conformal calibration pooled statistical coverage guarantee (population-level guarantee).
3. Abstention Mechanism:
   - Triggered if top-K cumulative IG mass < min_topk_mass (default 0.70).
   - Prevents sending diffuse, uninformative evidence to the LLM.
4. Strict LLM Instructions:
   - Enforce reasoning strictly on the provided top-K evidence.
   - Require structured JSON response.
"""

import os
import json
import warnings
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import yaml
import torch
import torch.nn as nn

try:
    from torch_geometric.data import Data as PyGData
except ImportError:
    PyGData = None

from src.data_loader import load_dataset, compute_node_features_6d
from src.gnn_attribution.model import GATAnomalyDetector
from src.gnn_attribution.validate_attribution import compute_integrated_gradients_attribution

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def extract_topk_sensor_evidence(
    ig_scores: Union[torch.Tensor, np.ndarray],
    fine_raw_window: np.ndarray,
    norm_stats: Dict[str, np.ndarray],
    top_k: int = 5,
    one_indexed: bool = True,
    min_sensor_std: Optional[float] = None
) -> Tuple[List[Dict[str, Any]], float, List[int]]:
    """Extract evidence for top-K sensors ranked by Integrated Gradients attribution score.

    Applies a variance floor mask: if min_sensor_std is provided, sensors whose
    baseline standard deviation in norm_stats['std'] is below min_sensor_std are
    excluded from candidate selection to avoid ranking near-constant/dead sensors.

    Args:
        ig_scores: 1D Tensor or array of attribution scores (length num_sensors, summing to 1.0).
        fine_raw_window: Raw test data array [window_length, num_sensors] for this segment.
        norm_stats: Dict with 'mean' and 'std' arrays from reference training set.
        top_k: Number of highest-attributed sensors to select (default 5).
        one_indexed: If True, report sensor IDs as 1 to 38.
        min_sensor_std: Minimum baseline std required for a sensor to be considered as evidence.

    Returns:
        sensor_evidence: List of dicts with sensor details (ID, score, raw stats, baseline stats, z-score).
        cumulative_mass: Total attribution mass captured by the top-K sensors in [0, 1].
        topk_dims: List of top-K sensor dimension indices.
    """
    if isinstance(ig_scores, torch.Tensor):
        scores = ig_scores.detach().cpu().numpy()
    else:
        scores = np.asarray(ig_scores, dtype=float)

    scores = np.abs(scores)
    total_score = np.sum(scores)
    if total_score > 0:
        scores = scores / total_score

    train_mean = norm_stats["mean"]
    train_std = norm_stats["std"]

    # Candidate indices: filter out sensors with baseline std < min_sensor_std
    if min_sensor_std is not None and min_sensor_std > 0.0:
        candidate_indices = [i for i in range(len(scores)) if float(train_std[i]) >= min_sensor_std]
    else:
        candidate_indices = list(range(len(scores)))

    if len(candidate_indices) == 0:
        candidate_indices = list(range(len(scores)))

    # Sort candidates by descending attribution score
    sorted_candidates = sorted(candidate_indices, key=lambda idx: scores[idx], reverse=True)
    k = min(top_k, len(sorted_candidates))
    topk_indices = sorted_candidates[:k]

    cumulative_mass = float(np.sum(scores[topk_indices]))

    sensor_evidence = []
    topk_dims = []

    for rank, idx in enumerate(topk_indices, 1):
        dim_id = int(idx + 1) if one_indexed else int(idx)
        topk_dims.append(dim_id)

        sensor_raw = fine_raw_window[:, idx]
        obs_mean = float(np.mean(sensor_raw))
        obs_min = float(np.min(sensor_raw))
        obs_max = float(np.max(sensor_raw))
        obs_std = float(np.std(sensor_raw))

        b_mean = float(train_mean[idx])
        b_std = float(train_std[idx])
        is_near_zero_var = bool(b_std < 1e-4)
        b_std_safe = b_std if b_std > 1e-7 else 1.0

        # Mean z-score deviation from historical baseline
        z_deviation = (obs_mean - b_mean) / b_std_safe

        sensor_evidence.append({
            "rank": rank,
            "sensor_id": dim_id,
            "sensor_name": f"Sensor_{dim_id}",
            "attribution_score": float(scores[idx]),
            "attribution_percent": float(scores[idx] * 100.0),
            "observed_mean": obs_mean,
            "observed_range": (obs_min, obs_max),
            "observed_std": obs_std,
            "historical_mean": b_mean,
            "historical_std": b_std,
            "historical_normal_range": (b_mean - 2.0 * b_std, b_mean + 2.0 * b_std),
            "z_deviation": float(z_deviation),
            "near_zero_variance": is_near_zero_var
        })

    return sensor_evidence, cumulative_mass, topk_dims


def build_decoupled_explanation_prompt(
    anomaly_id: Union[int, str],
    timesteps: Tuple[int, int],
    fine_raw_window: np.ndarray,
    ig_scores: Union[torch.Tensor, np.ndarray],
    norm_stats: Dict[str, np.ndarray],
    top_k: int = 5,
    min_topk_mass: float = 0.70,
    min_sensor_std: Optional[float] = None,
    alpha: float = 0.10,
    target_recall: float = 0.80,
    pooled_coverage_rate: float = 93.0,
    n_cal_segments: int = 327
) -> Dict[str, Any]:
    """Construct a prompt using decoupled evidence, confidence statement, and abstention gate.

    Args:
        anomaly_id: Segment identifier (e.g. 1 or 'machine-1-1_seg1').
        timesteps: Tuple of (start_timestep, end_timestep).
        fine_raw_window: Raw test data array [window_length, num_sensors].
        ig_scores: Attribution score vector from Integrated Gradients (length 38).
        norm_stats: Dictionary containing 'mean' and 'std' baseline arrays.
        top_k: Number of sensors to include as evidence (default 5).
        min_topk_mass: Minimum cumulative IG mass required to avoid abstention (default 0.70).
        min_sensor_std: Minimum baseline std for a sensor to be eligible for top-K evidence.
            Sensors with baseline std below this threshold are excluded (variance floor mask).
            If None, no filtering is applied.
        alpha: Nominal conformal significance level (default 0.10).
        target_recall: Target recall threshold for attribution (default 0.80).
        pooled_coverage_rate: Empirically validated calibration coverage rate percentage (e.g. 93.0).
        n_cal_segments: Number of segments used in pooled calibration (327).

    Returns:
        Dict containing:
            - 'prompt': Formatted prompt text for the LLM.
            - 'system_message': System prompt enforcing JSON format and constraints.
            - 'abstained': Boolean flag indicating whether the instance was abstained.
            - 'abstention_reason': Explanation of abstention if triggered.
            - 'cumulative_mass': Cumulative IG mass of top-K sensors.
            - 'top_k_evidence': List of evidence dicts for top-K sensors.
            - 'confidence_statement': Formatted statistical confidence text.
    """
    sensor_evidence, cumulative_mass, topk_dims = extract_topk_sensor_evidence(
        ig_scores=ig_scores,
        fine_raw_window=fine_raw_window,
        norm_stats=norm_stats,
        top_k=top_k,
        one_indexed=True,
        min_sensor_std=min_sensor_std
    )

    # 1. Abstention Trigger Check
    abstained = cumulative_mass < min_topk_mass

    # Determine the reason code for abstentions:
    #   "low_variance_filter_removed_candidate" — the variance floor mask caused the abstention
    #     (unfiltered top-K mass would have cleared the threshold; evidence existed but was
    #      deemed unreliable due to near-constant baseline).
    #   "insufficient_mass" — attribution is genuinely too diffuse even without any filtering.
    abstain_reason_code: Optional[str] = None
    unfiltered_mass: Optional[float] = None

    if abstained:
        if min_sensor_std is not None and min_sensor_std > 0.0:
            # Re-run without the variance floor to see what mass looks like unfiltered
            _, unfiltered_mass, _ = extract_topk_sensor_evidence(
                ig_scores=ig_scores,
                fine_raw_window=fine_raw_window,
                norm_stats=norm_stats,
                top_k=top_k,
                one_indexed=True,
                min_sensor_std=None  # no filter
            )
            if unfiltered_mass >= min_topk_mass:
                abstain_reason_code = "low_variance_filter_removed_candidate"
            else:
                abstain_reason_code = "insufficient_mass"
        else:
            abstain_reason_code = "insufficient_mass"

        abstention_reason = (
            f"ABSTAIN: Top-{top_k} sensors capture only {cumulative_mass * 100:.2f}% cumulative attribution mass "
            f"(after variance-floor filter), which is below the required concentration threshold of "
            f"{min_topk_mass * 100:.1f}%. "
            + (
                f"NOTE: Without the variance floor, unfiltered mass = {unfiltered_mass * 100:.2f}% — "
                f"abstention was caused by filtering unreliable low-variance sensors. "
                f"[abstain_reason: {abstain_reason_code}]"
                if abstain_reason_code == "low_variance_filter_removed_candidate"
                else
                f"The attribution signal is genuinely too diffuse across the sensor graph to reliably "
                f"isolate root causes. [abstain_reason: {abstain_reason_code}]"
            )
        )
    else:
        abstention_reason = None

    # 2. Formulate Confidence Statement (Decoupled from instance evidence)
    nominal_pct = int((1.0 - alpha) * 100)
    recall_pct = int(target_recall * 100)
    confidence_statement = (
        f"Attribution Confidence Context:\n"
        f"  - Instance Concentration: These top-{top_k} sensors account for {cumulative_mass * 100:.2f}% "
        f"of the model's total Integrated Gradients attribution signal.\n"
        f"  - Statistical Guarantee: At {nominal_pct}% nominal coverage, this class of attribution is "
        f"empirically validated to encompass at least {recall_pct}% of ground-truth root causes across "
        f"{n_cal_segments} multi-machine calibration segments (empirical coverage: {pooled_coverage_rate:.1f}%)."
    )

    # 3. Format Evidence Section
    evidence_lines = []
    for s in sensor_evidence:
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

    # 4. System Prompt and User Prompt
    system_message = (
        "You are a rigorous diagnostic assistant for industrial telemetry systems. "
        "Your task is to analyze telemetry anomalies and output structured explanations strictly "
        "grounded in the verified empirical evidence provided. Do not hypothesize unlisted sensor faults."
    )

    if abstained:
        user_prompt = f"""=== ANOMALY DIAGNOSTIC REQUEST ===
Anomaly Segment ID: {anomaly_id} (Timesteps: {timesteps[0]} to {timesteps[1]})

=== STATUS: ABSTENTION TRIGGERED ===
{abstention_reason}

=== INSTRUCTIONS ===
Because the model attribution is diffuse (insufficient concentrated evidence), you MUST abstain from nominating specific root causes.
Respond with the exact structured JSON below:
{{
  "root_cause_sensors": [],
  "explanation": "Insufficient evidence: Attribution signal is diffuse (top-{top_k} mass = {cumulative_mass * 100:.1f}% < {min_topk_mass * 100:.1f}% threshold). Diagnostic abstained to prevent ungrounded attribution.",
  "confidence_statement": "Abstained due to low concentration ({cumulative_mass * 100:.1f}% top-{top_k} mass).",
  "abstained": true
}}
"""
    else:
        user_prompt = f"""=== ANOMALY DIAGNOSTIC REQUEST ===
Anomaly Segment ID: {anomaly_id}
Active Timesteps: {timesteps[0]} to {timesteps[1]} (Duration: {timesteps[1] - timesteps[0] + 1} steps)

=== EMPIRICAL EVIDENCE (Top-{top_k} Attributed Sensors) ===
{evidence_text}

=== STATISTICAL CONFIDENCE CONTEXT ===
{confidence_statement}

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
  "confidence_statement": "{confidence_statement.replace(chr(10), ' ')}",
  "abstained": false
}}
"""

    return {
        "anomaly_id": anomaly_id,
        "timesteps": timesteps,
        "prompt": user_prompt.strip(),
        "system_message": system_message,
        "abstained": abstained,
        "abstain_reason_code": abstain_reason_code,  # "low_variance_filter_removed_candidate" | "insufficient_mass" | None
        "abstention_reason": abstention_reason,
        "cumulative_mass": cumulative_mass,
        "unfiltered_mass": unfiltered_mass,  # set only when abstained and filter was active
        "top_k_dims": topk_dims,
        "top_k_evidence": sensor_evidence,
        "confidence_statement": confidence_statement
    }


def test_prompt_builder_on_machine_1_1():
    """Test decoupled prompt construction on the 8 machine-1-1 segments without calling NIM API."""
    config_path = "configs/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    top_k = int(cfg.get("explanation_top_k", 5))
    min_topk_mass = float(cfg.get("min_topk_mass", 0.70))
    min_sensor_std = float(cfg.get("min_sensor_std", 1e-3))
    fine_margin = int(cfg.get("fine_window_margin", 15))
    alpha = float(cfg.get("conformal_alpha", 0.10))
    recall_thresh = float(cfg.get("recall_threshold", 0.80))
    device = torch.device(cfg.get("device", "cpu"))
    checkpoint_path = "results/gnn_checkpoint.pt"

    print("=" * 95)
    print("PROMPT BUILDER TEST: 8 MACHINE-1-1 GROUND-TRUTH SEGMENTS")
    print(f"Top-K Evidence: {top_k} sensors | Min Top-K Mass (Abstention): {min_topk_mass*100:.1f}%")
    print(f"Variance Floor: min_sensor_std = {min_sensor_std:.1e} (sensors below this are excluded from top-K)")
    print(f"Target Coverage: {(1 - alpha)*100:.0f}% (alpha={alpha}) | Recall Target: {recall_thresh*100:.0f}%")
    print("=" * 95)

    # 1. Load machine-1-1 dataset and model
    _, _, interp_labels, meta = load_dataset(
        dataset_name=cfg.get("dataset_name", "SMD"),
        raw_dir=cfg.get("dataset_path", "data/raw"),
        machine_id="machine-1-1",
        window_size=cfg.get("window_size", 100),
        stride=cfg.get("window_stride", 1),
        corr_threshold=cfg.get("edge_corr_threshold", 0.6)
    )

    model = GATAnomalyDetector(
        in_dim=6,
        hidden_dim=cfg.get("gnn_hidden_dim", 64),
        num_heads=cfg.get("gnn_num_heads", 4),
        edge_dim=1,
        dropout=0.2
    ).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded trained 6-D GNN checkpoint (epoch {ckpt.get('epoch')}).\n")
    model.eval()

    norm_test = meta["norm_test"]
    raw_test = meta["raw_test"]
    norm_stats = meta["norm_stats"]
    edge_index = meta["edge_index"]
    edge_attr = meta["edge_attr"]

    summary_rows = []

    # 2. Iterate through the 8 segments
    for idx, seg in enumerate(interp_labels, 1):
        start = seg["start"]
        end = seg["end"]
        gt_dims = seg["dimensions"]

        start_idx = max(0, int(start - fine_margin))
        end_idx = min(len(norm_test), int(end + fine_margin) + 1)
        fine_norm_win = norm_test[start_idx:end_idx]
        fine_raw_win = raw_test[start_idx:end_idx]

        # Build PyG Data with 6D features
        node_feats = compute_node_features_6d(fine_norm_win)
        x_ten = torch.tensor(node_feats, dtype=torch.float32, device=device)
        ei_ten = torch.tensor(edge_index, dtype=torch.long, device=device)
        ea_ten = torch.tensor(edge_attr, dtype=torch.float32, device=device) if edge_attr is not None else None
        data_sample = PyGData(x=x_ten, edge_index=ei_ten, edge_attr=ea_ten)

        # Compute Integrated Gradients
        ig_scores = compute_integrated_gradients_attribution(model, data_sample, steps=50)

        # Build Prompt
        prompt_data = build_decoupled_explanation_prompt(
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
            n_cal_segments=327
        )

        topk_dims = prompt_data["top_k_dims"]
        cum_mass = prompt_data["cumulative_mass"]
        abstained = prompt_data["abstained"]

        # Check for near-zero variance sensors
        zero_var_sensors = [
            f"{s['sensor_name']} (std={s['historical_std']:.2e})"
            for s in prompt_data["top_k_evidence"]
            if s.get("near_zero_variance", False)
        ]

        # Check GT overlap
        gt_set = set(gt_dims)
        overlap = set(topk_dims).intersection(gt_set)
        rec = (len(overlap) / len(gt_set)) * 100.0 if gt_set else 0.0

        status_str = "ABSTAIN" if abstained else "PROCEED"
        reason_code = prompt_data.get("abstain_reason_code")  # None when PROCEED
        unf_mass = prompt_data.get("unfiltered_mass")         # None when PROCEED or no filter
        summary_rows.append({
            "seg": idx,
            "timesteps": f"{start}-{end}",
            "gt_count": len(gt_dims),
            "topk_dims": topk_dims,
            "cum_mass": cum_mass,
            "unfiltered_mass": unf_mass,
            "status": status_str,
            "abstain_reason_code": reason_code,
            "gt_overlap": len(overlap),
            "recall": rec,
            "zero_var_sensors": zero_var_sensors
        })

        # Print prompt preview for Segment 1 and Segment 2
        if idx in [1, 2]:
            print(f"\n{'#' * 95}")
            print(f"PROMPT PREVIEW: SEGMENT {idx} (Status: {status_str} | Top-{top_k} Mass: {cum_mass*100:.2f}%)")
            print(f"{'#' * 95}")
            print(prompt_data["prompt"])
            print(f"{'#' * 95}\n")

    # 3. Print Overview Summary Table & Variance Diagnostics
    print("\n" + "=" * 107)
    print(f"MACHINE-1-1 PROMPT BUILDER & ABSTENTION SUMMARY (K={top_k}, Threshold={min_topk_mass*100:.0f}%, min_std={min_sensor_std:.1e})")
    print("=" * 107)
    print(
        f"{'Seg':<4} | {'Timesteps':<14} | {'GT':<4} | {'Top-5 Sensors':<22} | "
        f"{'Filtered Mass':<14} | {'Decision':<10} | {'Recall':<8} | {'Abstain Reason':<35}"
    )
    print("-" * 107)

    for r in summary_rows:
        dims_str = str(r["topk_dims"])
        reason_display = r["abstain_reason_code"] if r["abstain_reason_code"] else "N/A (proceeded)"

        print(
            f"{r['seg']:<4} | {r['timesteps']:<14} | {r['gt_count']:<4} | {dims_str:<22} | "
            f"{r['cum_mass']*100:>11.2f}%   | {r['status']:<10} | {r['recall']:>6.1f}%  | {reason_display:<35}"
        )

    print("=" * 107)
    n_proceed = sum(1 for r in summary_rows if r["status"] == "PROCEED")
    n_abstain = sum(1 for r in summary_rows if r["status"] == "ABSTAIN")
    n_filter_abstain = sum(1 for r in summary_rows if r["abstain_reason_code"] == "low_variance_filter_removed_candidate")
    n_genuine_abstain = sum(1 for r in summary_rows if r["abstain_reason_code"] == "insufficient_mass")
    print(f"Total Segments: {len(summary_rows)} | Proceed to LLM: {n_proceed} | Abstained: {n_abstain}")
    print(f"  Abstention Breakdown:")
    print(f"    low_variance_filter_removed_candidate : {n_filter_abstain}  (evidence existed but filtered as unreliable low-variance)")
    print(f"    insufficient_mass                     : {n_genuine_abstain}  (genuinely diffuse attribution, would abstain even without filter)")
    print(f"Average Filtered Top-{top_k} Mass: {np.mean([r['cum_mass'] for r in summary_rows])*100:.2f}%\n")

    # 4. Diagnostic Print: Near-Zero Baseline Variance Check
    print("=" * 95)
    print("DIAGNOSTIC: HISTORICAL BASELINE VARIANCE CHECK ON TOP-K SENSORS (Threshold: std < 1e-4)")
    print("Identifies potential numerical artifacts where near-constant baseline sensors receive attribution.")
    print("=" * 95)
    for r in summary_rows:
        zv = r["zero_var_sensors"]
        if zv:
            print(f"  [!] Segment {r['seg']} ({r['timesteps']}): Top-5 contains near-zero variance sensor(s): {', '.join(zv)}")
        else:
            print(f"  [OK] Segment {r['seg']} ({r['timesteps']}): All top-5 sensors have healthy historical variance (std >= 1e-4).")

    # 5. Reason-Code Breakdown per Abstained Segment
    print("=" * 95)
    print("ABSTENTION REASON CODE BREAKDOWN")
    print("=" * 95)
    for r in summary_rows:
        if r["status"] == "ABSTAIN":
            unf = r["unfiltered_mass"]
            unf_str = f"{unf*100:.2f}% (unfiltered)" if unf is not None else "N/A"
            print(
                f"  Segment {r['seg']} ({r['timesteps']}): "
                f"filtered_mass={r['cum_mass']*100:.2f}%  |  {unf_str}  |  "
                f"abstain_reason_code = \"{r['abstain_reason_code']}\""
            )
    print("=" * 95 + "\n")



if __name__ == "__main__":
    test_prompt_builder_on_machine_1_1()
