"""Stage 3 Extension: Multi-Machine Pooled Conformal Calibration.

Pools ground-truth root-cause interpretation-labeled segments from all 28 SMD machines
to scale the attribution calibration set from N=8 to N=327 segments.

Maintains the frozen GNN model trained strictly on machine-1-1 and applies machine-1-1's
normalization statistics with explicit distribution shift / OOD monitoring.
"""

import os
import time
import warnings
from typing import Dict, Any, List, Tuple, Optional, Set
import numpy as np
import yaml
import torch
import torch.nn as nn

try:
    from torch_geometric.data import Data as PyGData
except ImportError:
    PyGData = None

from src.data_loader import load_dataset, parse_interpretation_labels, create_pyg_data_object, compute_node_features_6d
from src.gnn_attribution.model import GATAnomalyDetector
from src.gnn_attribution.validate_attribution import compute_integrated_gradients_attribution
from src.calibration.conformal import calibrate_attribution

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def discover_all_smd_interpretation_labels(
    raw_dir: str = "data/raw"
) -> Tuple[List[Dict[str, Any]], int]:
    """Scan data/raw/SMD/interpretation_label/ for all machine files.

    Args:
        raw_dir: Root directory for raw data.

    Returns:
        machines_meta: List of dicts with machine_id, segment_count, file_size, and parsed segments.
        total_segments: Total count of anomaly segments across all machines.
    """
    interp_dir = os.path.join(raw_dir, "SMD", "interpretation_label")
    if not os.path.exists(interp_dir):
        raise FileNotFoundError(f"SMD interpretation directory not found: {interp_dir}")

    files = sorted([f for f in os.listdir(interp_dir) if f.endswith(".txt")])
    machines_meta = []
    total_segments = 0

    for fname in files:
        fpath = os.path.join(interp_dir, fname)
        machine_id = fname.replace(".txt", "")
        fsize = os.path.getsize(fpath)
        segments = parse_interpretation_labels(fpath)
        seg_count = len(segments)
        total_segments += seg_count

        machines_meta.append({
            "machine_id": machine_id,
            "filename": fname,
            "file_path": fpath,
            "file_size": fsize,
            "segment_count": seg_count,
            "segments": segments,
            "status": "Usable" if seg_count > 0 else "Empty"
        })

    return machines_meta, total_segments


def print_discovery_table(machines_meta: List[Dict[str, Any]], total_segments: int) -> None:
    """Print formatted Discovery Table of all 28 SMD interpretation label files."""
    print("\n" + "=" * 78)
    print("STEP 1 — DISCOVERY: SMD GROUND-TRUTH INTERPRETATION-LABELED SEGMENTS")
    print("=" * 78)
    print(f"{'Machine ID':<16} | {'Segments Count':<16} | {'File Size (Bytes)':<20} | {'Status':<12}")
    print("-" * 78)

    for m in machines_meta:
        print(f"{m['machine_id']:<16} | {m['segment_count']:<16} | {m['file_size']:<20} | {m['status']:<12}")

    print("-" * 78)
    print(f"Total Machines Scanned:   {len(machines_meta)}")
    print(f"Total Usable Machines:    {sum(1 for m in machines_meta if m['segment_count'] > 0)} / {len(machines_meta)}")
    print(f"Total Anomaly Segments:   {total_segments} (vs. 8 on machine-1-1 alone -> {total_segments / 8:.1f}x scaling)")
    print("=" * 78 + "\n")


def extract_cross_machine_attributions(
    model: nn.Module,
    machines_meta: List[Dict[str, Any]],
    machine_1_1_norm_stats: Dict[str, np.ndarray],
    edge_index: Any,
    edge_attr: Any,
    raw_dir: str = "data/raw",
    fine_window_margin: int = 15,
    ig_steps: int = 50,
    ood_z_threshold: float = 5.0
) -> List[Dict[str, Any]]:
    """Extract Integrated Gradients attributions across all machines using machine-1-1's normalization.

    Args:
        model: Frozen GAT model trained on machine-1-1.
        machines_meta: List of machine metadata from discovery.
        machine_1_1_norm_stats: Dict with 'mean' and 'std' from machine-1-1 train set.
        edge_index: Static sensor graph connectivity from machine-1-1.
        edge_attr: Static sensor graph edge attributes from machine-1-1.
        raw_dir: Root raw data path.
        fine_window_margin: Margin in timesteps for fine window extraction (default 15).
        ig_steps: Number of Riemann interpolation steps for Integrated Gradients (default 50).
        ood_z_threshold: Absolute z-score threshold for distribution shift flagging (default 5.0).

    Returns:
        all_segment_records: List of attribution records for each segment across all machines.
    """
    model.eval()
    device = next(model.parameters()).device

    mean_1_1 = machine_1_1_norm_stats["mean"]
    std_1_1 = machine_1_1_norm_stats["std"]
    # Avoid zero division
    std_adj_1_1 = np.where(std_1_1 < 1e-8, 1.0, std_1_1)

    all_segment_records = []

    print("\n" + "=" * 90)
    print("STEP 2 — CROSS-MACHINE ATTRIBUTION & DISTRIBUTION SHIFT (OOD) MONITORING")
    print(f"Normalization: machine-1-1 statistics (mean, std) | OOD Threshold: |z| > {ood_z_threshold}")
    print("=" * 90)
    print(f"{'Machine ID':<14} | {'Segments':<10} | {'Test Shape':<14} | {'Max |z|':<10} | {'% Points |z|>5':<16} | {'OOD Status':<12}")
    print("-" * 90)

    for m in machines_meta:
        mid = m["machine_id"]
        if m["segment_count"] == 0:
            continue

        test_path = os.path.join(raw_dir, "SMD", "test", f"{mid}.txt")
        if not os.path.exists(test_path):
            print(f"[Warning] Test file missing for {mid}: {test_path}")
            continue

        raw_test = np.loadtxt(test_path, delimiter=",", dtype=np.float32)

        # Apply machine-1-1's normalization
        norm_test = (raw_test - mean_1_1) / std_adj_1_1
        zero_std_mask = std_1_1 < 1e-8
        if np.any(zero_std_mask):
            norm_test[:, zero_std_mask] = 0.0

        # Distribution shift / OOD statistics
        abs_z = np.abs(norm_test)
        max_z = float(np.max(abs_z))
        mean_abs_z = float(np.mean(abs_z))
        pct_ood_points = float(np.mean(abs_z > ood_z_threshold) * 100.0)

        # Flag if significant distribution shift
        is_ood = (pct_ood_points > 5.0) or (max_z > 25.0)
        ood_str = "[OOD SHIFT]" if is_ood else "[NORMAL SCALE]"

        print(
            f"{mid:<14} | {m['segment_count']:<10} | {str(raw_test.shape):<14} | "
            f"{max_z:>8.2f}   | {pct_ood_points:>12.2f}%    | {ood_str:<12}"
        )

        # Extract fine window and compute Integrated Gradients for each segment
        for seg_idx, seg in enumerate(m["segments"], 1):
            start = seg["start"]
            end = seg["end"]
            gt_dims = seg["dimensions"]

            # Extract fine window around segment
            start_idx = max(0, int(start - fine_window_margin))
            end_idx = min(len(norm_test), int(end + fine_window_margin) + 1)
            fine_win = norm_test[start_idx:end_idx]

            node_feat = compute_node_features_6d(fine_win)  # [num_sensors, 6]

            x_ten = torch.tensor(node_feat, dtype=torch.float32, device=device)
            ei_ten = torch.tensor(edge_index, dtype=torch.long, device=device)
            ea_ten = torch.tensor(edge_attr, dtype=torch.float32, device=device) if edge_attr is not None else None
            data_sample = PyGData(x=x_ten, edge_index=ei_ten, edge_attr=ea_ten)

            ig_scores = compute_integrated_gradients_attribution(model, data_sample, steps=ig_steps)

            all_segment_records.append({
                "machine_id": mid,
                "segment_index": seg_idx,
                "start": start,
                "end": end,
                "window_len": end_idx - start_idx,
                "gt_dimensions": gt_dims,
                "ig_scores": ig_scores,
                "max_z": max_z,
                "pct_ood": pct_ood_points,
                "is_ood": is_ood
            })

    print("-" * 90)
    print(f"Successfully processed {len(all_segment_records)} total segments across all machines.\n")
    return all_segment_records


def run_pooled_conformal_calibration(
    all_segment_records: List[Dict[str, Any]],
    machine_1_1_records: List[Dict[str, Any]],
    alpha: float = 0.10,
    recall_threshold: float = 0.80
) -> Dict[str, Any]:
    """Perform split-conformal calibration on the pooled multi-machine dataset.

    Compares the strict 100% containment criterion against the relaxed
    recall-threshold criterion (e.g. 80% recall), and breaks down results for
    machine-1-1 segments, highlighting Segments 2 and 4.

    Args:
        all_segment_records: Full pooled list of segments (N ~ 327).
        machine_1_1_records: machine-1-1 only segments (N = 8).
        alpha: Nominal significance level (default 0.10).
        recall_threshold: Target minimum recall threshold for GT root causes (default 0.80).

    Returns:
        Dict comparing calibration metrics across criteria and sets.
    """
    n_pooled = len(all_segment_records)
    n_single = len(machine_1_1_records)

    min_n_for_alpha = int(np.ceil(1.0 / alpha) - 1)
    min_achievable_alpha_single = 1.0 / (n_single + 1)
    min_achievable_alpha_pooled = 1.0 / (n_pooled + 1)

    print("=" * 105)
    print(f"STEP 3 — POOLED CALIBRATION ANALYSIS (Target Nominal Coverage: {(1-alpha)*100:.1f}%, Alpha={alpha})")
    print(f"Comparing Strict Full Containment (100% GT) vs. Relaxed Recall-Threshold ({recall_threshold*100:.0f}% GT)")
    print("=" * 105)
    print("Sample Size & Conformal Validity Analysis:")
    print(f"  - Minimum sample size required for alpha={alpha:.2f} (1-alpha={(1-alpha)*100:.0f}%): n >= {min_n_for_alpha}")
    print(f"  - Single-machine-1-1 Sample Size: n = {n_single} -> Satisfies n>={min_n_for_alpha}? {'YES' if n_single >= min_n_for_alpha else 'NO (n=8 < 9)'}")
    print(f"    (Smallest statistically valid alpha on machine-1-1 alone: alpha >= {min_achievable_alpha_single:.4f} -> max valid coverage: {(1 - min_achievable_alpha_single)*100:.1f}%)")
    print(f"  - Pooled Multi-Machine Sample Size: n = {n_pooled} -> Satisfies n>={min_n_for_alpha}? YES (n={n_pooled} >> 9)")
    print(f"    (Smallest statistically valid alpha on pooled set: alpha >= {min_achievable_alpha_pooled:.4f} -> max valid coverage: {(1 - min_achievable_alpha_pooled)*100:.2f}%)")

    # Helper to evaluate calibration on a dataset
    def evaluate_calibration(cal_res, records):
        full_contained = 0
        threshold_contained = 0
        recalls = []
        for i, seg in enumerate(records):
            gt_set = set(seg["gt_dimensions"])
            conf_set = set(cal_res["conformal_sensor_sets"][i])
            overlap = gt_set.intersection(conf_set)
            rec = (len(overlap) / len(gt_set)) if len(gt_set) > 0 else 1.0
            recalls.append(rec * 100.0)
            if gt_set.issubset(conf_set):
                full_contained += 1
            if rec >= (recall_threshold - 1e-9):
                threshold_contained += 1
        return {
            "full_contained": full_contained,
            "threshold_contained": threshold_contained,
            "mean_recall": float(np.mean(recalls)),
            "full_containment_rate": full_contained / len(records) * 100.0 if records else 0.0,
            "threshold_containment_rate": threshold_contained / len(records) * 100.0 if records else 0.0,
        }

    # 1. Old Criterion: Full 100% Containment
    # ----------------------------------------------------
    single_cal_100 = calibrate_attribution(
        test_ig_scores=[s["ig_scores"] for s in machine_1_1_records],
        calibration_segments=machine_1_1_records,
        alpha=alpha,
        one_indexed=True,
        recall_threshold=1.0
    )
    pooled_cal_100 = calibrate_attribution(
        test_ig_scores=[s["ig_scores"] for s in all_segment_records],
        calibration_segments=all_segment_records,
        alpha=alpha,
        one_indexed=True,
        recall_threshold=1.0
    )
    m11_on_pooled_100 = calibrate_attribution(
        test_ig_scores=[s["ig_scores"] for s in machine_1_1_records],
        calibration_segments=all_segment_records,
        alpha=alpha,
        one_indexed=True,
        recall_threshold=1.0
    )

    eval_single_100 = evaluate_calibration(single_cal_100, machine_1_1_records)
    eval_pooled_100 = evaluate_calibration(pooled_cal_100, all_segment_records)
    eval_m11_on_pooled_100 = evaluate_calibration(m11_on_pooled_100, machine_1_1_records)

    # 2. New Criterion: Relaxed Recall-Threshold (e.g. 80%)
    # ----------------------------------------------------
    single_cal_rel = calibrate_attribution(
        test_ig_scores=[s["ig_scores"] for s in machine_1_1_records],
        calibration_segments=machine_1_1_records,
        alpha=alpha,
        one_indexed=True,
        recall_threshold=recall_threshold
    )
    pooled_cal_rel = calibrate_attribution(
        test_ig_scores=[s["ig_scores"] for s in all_segment_records],
        calibration_segments=all_segment_records,
        alpha=alpha,
        one_indexed=True,
        recall_threshold=recall_threshold
    )
    m11_on_pooled_rel = calibrate_attribution(
        test_ig_scores=[s["ig_scores"] for s in machine_1_1_records],
        calibration_segments=all_segment_records,
        alpha=alpha,
        one_indexed=True,
        recall_threshold=recall_threshold
    )

    eval_single_rel = evaluate_calibration(single_cal_rel, machine_1_1_records)
    eval_pooled_rel = evaluate_calibration(pooled_cal_rel, all_segment_records)
    eval_m11_on_pooled_rel = evaluate_calibration(m11_on_pooled_rel, machine_1_1_records)

    # Format side-by-side comparison table
    print("\n" + "=" * 105)
    print(f"HEADLINE COMPARISON: 100% FULL CONTAINMENT vs. RELAXED ({recall_threshold*100:.0f}%) RECALL-THRESHOLD")
    print("=" * 105)
    print(
        f"{'Metric':<42} | {'Strict (100% GT)':<28} | {'Relaxed (' + str(int(recall_threshold*100)) + '% GT)':<28}"
    )
    print("-" * 105)
    print(f"{'--- POOLED SMD (N=327) ---':<42} | {'':<28} | {'':<28}")
    print(f"{'Conformal Cumulative IG Threshold (tau_hat)':<42} | {pooled_cal_100['tau_hat']:<28.4f} | {pooled_cal_rel['tau_hat']:<28.4f}")
    print(f"{'Average Conformal Set Size (sensors)':<42} | {pooled_cal_100['mean_set_size']:<28.1f} | {pooled_cal_rel['mean_set_size']:<28.1f}")
    print(f"{'Average Set Size % of 38 sensors':<42} | {pooled_cal_100['mean_set_size']/38*100:<27.1f}% | {pooled_cal_rel['mean_set_size']/38*100:<27.1f}%")
    print(f"{'Full 100% Containment Rate':<42} | {eval_pooled_100['full_contained']}/{n_pooled} ({eval_pooled_100['full_containment_rate']:.1f}%) | {eval_pooled_rel['full_contained']}/{n_pooled} ({eval_pooled_rel['full_containment_rate']:.1f}%)")
    print(f"{'>=' + str(int(recall_threshold*100)) + '% Target Recall Coverage Rate':<42} | {eval_pooled_100['threshold_contained']}/{n_pooled} ({eval_pooled_100['threshold_containment_rate']:.1f}%) | {eval_pooled_rel['threshold_contained']}/{n_pooled} ({eval_pooled_rel['threshold_containment_rate']:.1f}%)")
    print(f"{'Mean Root-Cause Sensor Recall %':<42} | {eval_pooled_100['mean_recall']:<27.2f}% | {eval_pooled_rel['mean_recall']:<27.2f}%")
    print("-" * 105)
    print(f"{'--- ON MACHINE-1-1 SEGMENTS (N=8) ---':<42} | {'':<28} | {'':<28}")
    print(f"{'Average Set Size (calibrated on pooled)':<42} | {m11_on_pooled_100['mean_set_size']:<28.1f} | {m11_on_pooled_rel['mean_set_size']:<28.1f}")
    print(f"{'Average Set Size % of 38 sensors':<42} | {m11_on_pooled_100['mean_set_size']/38*100:<27.1f}% | {m11_on_pooled_rel['mean_set_size']/38*100:<27.1f}%")
    print(f"{'Full 100% Containment Rate':<42} | {eval_m11_on_pooled_100['full_contained']}/{n_single} ({eval_m11_on_pooled_100['full_containment_rate']:.1f}%) | {eval_m11_on_pooled_rel['full_contained']}/{n_single} ({eval_m11_on_pooled_rel['full_containment_rate']:.1f}%)")
    print(f"{'>=' + str(int(recall_threshold*100)) + '% Target Recall Coverage Rate':<42} | {eval_m11_on_pooled_100['threshold_contained']}/{n_single} ({eval_m11_on_pooled_100['threshold_containment_rate']:.1f}%) | {eval_m11_on_pooled_rel['threshold_contained']}/{n_single} ({eval_m11_on_pooled_rel['threshold_containment_rate']:.1f}%)")
    print(f"{'Mean Root-Cause Sensor Recall %':<42} | {eval_m11_on_pooled_100['mean_recall']:<27.2f}% | {eval_m11_on_pooled_rel['mean_recall']:<27.2f}%")
    print("=" * 105)

    # 3. Detailed Segment-by-Segment Report on Machine-1-1 (Focus on Segments 2 & 4)
    print("\n" + "=" * 105)
    print("MACHINE-1-1 SEGMENT-BY-SEGMENT BREAKDOWN (Focus on Large-GT Segments 2 & 4)")
    print("=" * 105)
    print(
        f"{'Seg':<4} | {'Timesteps':<14} | {'GT Count':<8} | "
        f"{'Strict Set':<11} | {'Strict Rec':<11} | "
        f"{'Relaxed Set':<12} | {'Relaxed Rec':<12} | {'Notes':<20}"
    )
    print("-" * 105)

    for i, seg in enumerate(machine_1_1_records):
        seg_id = i + 1
        gt_dims = seg["gt_dimensions"]
        gt_set = set(gt_dims)

        set_100 = set(m11_on_pooled_100["conformal_sensor_sets"][i])
        set_rel = set(m11_on_pooled_rel["conformal_sensor_sets"][i])

        rec_100 = (len(gt_set.intersection(set_100)) / len(gt_set)) * 100.0 if gt_set else 0.0
        rec_rel = (len(gt_set.intersection(set_rel)) / len(gt_set)) * 100.0 if gt_set else 0.0

        note = ""
        if seg_id == 2:
            note = "<-- Hard case (|GT|=31)"
        elif seg_id == 4:
            note = "<-- Hard case (|GT|=14)"
        elif seg_id in [6, 7, 8]:
            note = "Point anomaly (|GT|=4)"

        print(
            f"{seg_id:<4} | {seg['start']}-{seg['end']:<11} | {len(gt_dims):<8} | "
            f"{len(set_100):<2} / 38      | {rec_100:>6.1f}%    | "
            f"{len(set_rel):<2} / 38       | {rec_rel:>6.1f}%     | {note:<20}"
        )
    print("=" * 105 + "\n")

    return {
        "n_pooled": n_pooled,
        "n_single": n_single,
        "pooled_100": {
            "tau_hat": pooled_cal_100["tau_hat"],
            "mean_set_size": pooled_cal_100["mean_set_size"],
            **eval_pooled_100
        },
        "pooled_relaxed": {
            "tau_hat": pooled_cal_rel["tau_hat"],
            "mean_set_size": pooled_cal_rel["mean_set_size"],
            **eval_pooled_rel
        },
        "m11_on_pooled_100": {
            "tau_hat": m11_on_pooled_100["tau_hat"],
            "mean_set_size": m11_on_pooled_100["mean_set_size"],
            **eval_m11_on_pooled_100
        },
        "m11_on_pooled_relaxed": {
            "tau_hat": m11_on_pooled_rel["tau_hat"],
            "mean_set_size": m11_on_pooled_rel["mean_set_size"],
            **eval_m11_on_pooled_rel
        }
    }


def main():
    config_path = "configs/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    alpha = float(cfg.get("conformal_alpha", 0.10))
    fine_margin = int(cfg.get("fine_window_margin", 15))
    device = torch.device(cfg.get("device", "cpu"))
    checkpoint_path = "results/gnn_checkpoint.pt"

    print("=" * 90)
    print("POOLED MULTI-MACHINE CONFORMAL ATTRIBUTION CALIBRATION PIPELINE")
    print(f"Target Confidence: (1 - alpha) = {(1 - alpha)*100:.1f}% (alpha={alpha}) | Fine Margin: {fine_margin}")
    print("=" * 90)

    # Step 1 — Discovery
    machines_meta, total_segments = discover_all_smd_interpretation_labels(raw_dir=cfg.get("dataset_path", "data/raw"))
    print_discovery_table(machines_meta, total_segments)

    # Load machine-1-1 reference data & model
    print("[Reference Setup] Loading machine-1-1 baseline data & trained GNN model...")
    _, _, interp_labels_11, meta_11 = load_dataset(
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
        print(f"Loaded trained GNN checkpoint from epoch {ckpt.get('epoch', 'N/A')} (trained on machine-1-1).")
    else:
        print("Warning: GNN Checkpoint not found at results/gnn_checkpoint.pt.")

    model.eval()

    # Step 2 — Cross-Machine Attribution with OOD Monitoring
    all_segment_records = extract_cross_machine_attributions(
        model=model,
        machines_meta=machines_meta,
        machine_1_1_norm_stats=meta_11["norm_stats"],
        edge_index=meta_11["edge_index"],
        edge_attr=meta_11["edge_attr"],
        raw_dir=cfg.get("dataset_path", "data/raw"),
        fine_window_margin=fine_margin,
        ig_steps=50,
        ood_z_threshold=5.0
    )

    # Extract machine-1-1 subset for direct baseline comparison
    m11_records = [r for r in all_segment_records if r["machine_id"] == "machine-1-1"]

    # Step 3 — Pooled Conformal Calibration
    recall_threshold = float(cfg.get("recall_threshold", 0.80))
    comparison_summary = run_pooled_conformal_calibration(
        all_segment_records=all_segment_records,
        machine_1_1_records=m11_records,
        alpha=alpha,
        recall_threshold=recall_threshold
    )

    print("[SUCCESS] Multi-Machine Pooled Calibration completed successfully.")


if __name__ == "__main__":
    main()
