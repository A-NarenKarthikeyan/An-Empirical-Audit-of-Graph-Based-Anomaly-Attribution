"""Diagnostic Module: src/calibration/diagnose_attribution.py

Read-only analysis to separate two causes of large conformal attribution set size (34-35/38 sensors):
  Diagnostic 1 — Inherent IG diffuseness on machine-1-1 (in-distribution)
  Diagnostic 2 — Distribution shift across machines (out-of-distribution)
"""

import os
import warnings
from typing import Dict, Any, List, Tuple
import numpy as np
import yaml
import torch
import torch.nn as nn

try:
    from torch_geometric.data import Data as PyGData
except ImportError:
    PyGData = None

from src.data_loader import load_dataset, parse_interpretation_labels, compute_node_features_6d
from src.gnn_attribution.model import GATAnomalyDetector
from src.gnn_attribution.validate_attribution import compute_integrated_gradients_attribution
from src.calibration.conformal import calibrate_attribution

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def build_pyg_data_from_window(
    fine_win: np.ndarray,
    edge_index: Any,
    edge_attr: Any,
    device: torch.device
):
    """Construct a PyG Data object from a fine window array [T, C] using 6D features."""
    node_feat = compute_node_features_6d(fine_win)  # [num_sensors, 6]
    x_ten = torch.tensor(node_feat, dtype=torch.float32, device=device)
    ei_ten = torch.tensor(edge_index, dtype=torch.long, device=device)
    ea_ten = torch.tensor(edge_attr, dtype=torch.float32, device=device) if edge_attr is not None else None
    return PyGData(x=x_ten, edge_index=ei_ten, edge_attr=ea_ten)


def compute_ig_concentration_stats(ig_scores: torch.Tensor, mass_targets=(0.50, 0.75, 0.90, 0.95)) -> Dict[str, Any]:
    """Compute concentration statistics for a single IG score vector.

    Args:
        ig_scores: 1D tensor of normalized attribution scores (sum to 1.0).
        mass_targets: Tuple of cumulative mass thresholds to compute sensor counts for.

    Returns:
        Dict containing:
            - 'top_k_cumulative_mass': Dict mapping k -> cumulative mass for k=1,2,3,5,8,10
            - 'sensors_to_reach_mass': Dict mapping target -> number of sensors needed
            - 'gini_coefficient': Gini coefficient as concentration measure (0=uniform, 1=all in one)
    """
    scores = ig_scores.cpu().numpy() if isinstance(ig_scores, torch.Tensor) else np.asarray(ig_scores)
    scores = np.abs(scores)
    total = scores.sum()
    if total > 0:
        scores = scores / total

    sorted_desc = np.sort(scores)[::-1]
    cumsum = np.cumsum(sorted_desc)
    n = len(scores)

    # Top-K cumulative mass
    top_k_mass = {}
    for k in [1, 2, 3, 5, 8, 10, 15, 20]:
        top_k_mass[k] = float(np.sum(sorted_desc[:min(k, n)]))

    # Sensors to reach cumulative mass targets
    sensors_needed = {}
    for target in mass_targets:
        positions = np.searchsorted(cumsum, target)
        sensors_needed[target] = int(positions + 1)  # 1-indexed count

    # Gini coefficient: measures concentration (0 = uniform, 1 = single sensor)
    # Gini = (2 * sum(i * x_i) / (n * sum(x_i))) - (n+1)/n for sorted ascending x_i
    sorted_asc = np.sort(scores)
    ranks = np.arange(1, n + 1)
    gini = float((2 * np.sum(ranks * sorted_asc)) / (n * np.sum(sorted_asc)) - (n + 1) / n) if np.sum(sorted_asc) > 0 else 0.0

    return {
        "top_k_cumulative_mass": top_k_mass,
        "sensors_to_reach_mass": sensors_needed,
        "gini_coefficient": gini,
    }


def diagnostic_1_ig_concentration(
    model: nn.Module,
    norm_test_data: np.ndarray,
    interp_labels: List[Dict[str, Any]],
    edge_index: Any,
    edge_attr: Any,
    fine_margin: int = 15,
    ig_steps: int = 50
) -> List[Dict[str, Any]]:
    """Diagnostic 1: IG concentration on machine-1-1's 8 ground-truth segments (in-distribution)."""
    model.eval()
    device = next(model.parameters()).device
    results = []

    print("\n" + "=" * 100)
    print("DIAGNOSTIC 1 — IG CONCENTRATION ON MACHINE-1-1 (IN-DISTRIBUTION)")
    print("Measures whether IG attribution is inherently diffuse even without cross-machine distribution shift.")
    print("=" * 100)
    print(
        f"{'Seg':<4} | {'Timesteps':<14} | {'GT K':<5} | "
        f"{'Top-5 Mass':<12} | {'Top-8 Mass':<12} | "
        f"{'#Sensors@50%':<14} | {'#Sensors@75%':<14} | {'#Sensors@90%':<14} | {'#Sensors@95%':<14} | {'Gini':<8}"
    )
    print("-" * 100)

    for idx, seg in enumerate(interp_labels, 1):
        start = seg["start"]
        end = seg["end"]
        gt_dims = seg["dimensions"]

        start_idx = max(0, int(start - fine_margin))
        end_idx = min(len(norm_test_data), int(end + fine_margin) + 1)
        fine_win = norm_test_data[start_idx:end_idx]

        data_sample = build_pyg_data_from_window(fine_win, edge_index, edge_attr, device)
        ig_scores = compute_integrated_gradients_attribution(model, data_sample, steps=ig_steps)
        stats = compute_ig_concentration_stats(ig_scores)

        top5 = stats["top_k_cumulative_mass"][5]
        top8 = stats["top_k_cumulative_mass"][8]
        s50 = stats["sensors_to_reach_mass"][0.50]
        s75 = stats["sensors_to_reach_mass"][0.75]
        s90 = stats["sensors_to_reach_mass"][0.90]
        s95 = stats["sensors_to_reach_mass"][0.95]
        gini = stats["gini_coefficient"]

        print(
            f"{idx:<4} | {start}-{end:<11} | {len(gt_dims):<5} | "
            f"{top5*100:>9.2f}%  | {top8*100:>9.2f}%  | "
            f"{s50:<14} | {s75:<14} | {s90:<14} | {s95:<14} | {gini:.4f}"
        )

        results.append({
            "segment_id": idx,
            "start": start,
            "end": end,
            "gt_dims": gt_dims,
            "ig_scores": ig_scores,
            "top5_mass": top5,
            "top8_mass": top8,
            "sensors_at_50pct": s50,
            "sensors_at_75pct": s75,
            "sensors_at_90pct": s90,
            "sensors_at_95pct": s95,
            "gini": gini,
            **stats,
        })

    print("-" * 100)
    avg_top5 = np.mean([r["top5_mass"] for r in results]) * 100
    avg_top8 = np.mean([r["top8_mass"] for r in results]) * 100
    avg_s90 = np.mean([r["sensors_at_90pct"] for r in results])
    avg_gini = np.mean([r["gini"] for r in results])

    print(f"Average Top-5 Cumulative Mass:        {avg_top5:.2f}%")
    print(f"Average Top-8 Cumulative Mass:        {avg_top8:.2f}%")
    print(f"Average #Sensors to reach 90% mass:   {avg_s90:.1f} / 38")
    print(f"Average Gini Coefficient:             {avg_gini:.4f}  (0=uniform, 1=concentrated)")

    # Interpret
    print("\n[Diagnostic 1 Interpretation]")
    if avg_s90 >= 30:
        print(f"  -> IG IS INHERENTLY DIFFUSE: Needs ~{avg_s90:.0f}/38 sensors to reach 90% mass on average.")
        print("     Attribution is spread nearly uniformly across all 38 sensors.")
        print("     This alone explains most of the large conformal set size (35/38).")
        ig_diffuse = True
    elif avg_s90 >= 15:
        print(f"  -> IG IS MODERATELY DIFFUSE: Needs ~{avg_s90:.0f}/38 sensors to reach 90% mass.")
        print("     Attribution is partially concentrated — cross-machine OOD likely compounds the issue.")
        ig_diffuse = None  # Mixed
    else:
        print(f"  -> IG IS CONCENTRATED: Only ~{avg_s90:.0f}/38 sensors needed for 90% mass.")
        print("     Cross-machine OOD pooling likely drives the large conformal sets.")
        ig_diffuse = False

    print("=" * 100 + "\n")
    return results


def diagnostic_2_split_by_distribution_shift(
    model: nn.Module,
    machines_meta: List[Dict[str, Any]],
    norm_stats_11: Dict[str, np.ndarray],
    edge_index: Any,
    edge_attr: Any,
    raw_dir: str = "data/raw",
    fine_margin: int = 15,
    ig_steps: int = 50,
    low_shift_threshold_pct: float = 10.0,
    alpha: float = 0.10
) -> Dict[str, Any]:
    """Diagnostic 2: Split calibration by distribution shift severity.

    Classifies machines as 'low-shift' (<10% OOD timesteps) vs 'high-shift' (>=10% OOD timesteps)
    using machine-1-1 normalization stats. Runs calibrate_attribution separately for each group.
    """
    mean_11 = norm_stats_11["mean"]
    std_11 = norm_stats_11["std"]
    std_adj = np.where(std_11 < 1e-8, 1.0, std_11)
    device = next(model.parameters()).device
    model.eval()

    print("\n" + "=" * 100)
    print(f"DIAGNOSTIC 2 — CALIBRATION SPLIT BY DISTRIBUTION SHIFT SEVERITY (Threshold: <{low_shift_threshold_pct:.0f}% OOD)")
    print("=" * 100)

    low_shift_machines = []
    high_shift_machines = []
    all_ood_stats = []

    print(f"\n{'Machine ID':<16} | {'Segments':<10} | {'Max |z|':<12} | {'% |z|>5':<12} | {'Group':<12}")
    print("-" * 70)

    for m in machines_meta:
        mid = m["machine_id"]
        if m["segment_count"] == 0:
            continue

        test_path = os.path.join(raw_dir, "SMD", "test", f"{mid}.txt")
        if not os.path.exists(test_path):
            continue

        raw_test = np.loadtxt(test_path, delimiter=",", dtype=np.float32)
        norm_test = (raw_test - mean_11) / std_adj
        zero_std_mask = std_11 < 1e-8
        if np.any(zero_std_mask):
            norm_test[:, zero_std_mask] = 0.0

        abs_z = np.abs(norm_test)
        max_z = float(np.max(abs_z))
        pct_ood = float(np.mean(abs_z > 5.0) * 100.0)

        group = "LOW-SHIFT" if pct_ood < low_shift_threshold_pct else "HIGH-SHIFT"
        print(f"{mid:<16} | {m['segment_count']:<10} | {max_z:>10.2f}   | {pct_ood:>9.2f}%   | {group:<12}")

        machine_record = {
            "machine_id": mid,
            "segment_count": m["segment_count"],
            "segments": m["segments"],
            "norm_test": norm_test,
            "raw_test": raw_test,
            "pct_ood": pct_ood,
            "max_z": max_z,
            "group": group,
        }
        all_ood_stats.append(machine_record)

        if pct_ood < low_shift_threshold_pct:
            low_shift_machines.append(machine_record)
        else:
            high_shift_machines.append(machine_record)

    print("-" * 70)
    n_low_segs = sum(m["segment_count"] for m in low_shift_machines)
    n_high_segs = sum(m["segment_count"] for m in high_shift_machines)
    print(f"LOW-SHIFT  group:  {len(low_shift_machines)} machines, {n_low_segs} segments ({np.mean([m['pct_ood'] for m in low_shift_machines]):.2f}% avg OOD)")
    print(f"HIGH-SHIFT group:  {len(high_shift_machines)} machines, {n_high_segs} segments ({np.mean([m['pct_ood'] for m in high_shift_machines]):.2f}% avg OOD)")
    print("=" * 100)

    def extract_segments_from_machines(machine_list: List[Dict]) -> List[Dict[str, Any]]:
        """Extract IG attribution records for all segments from a list of machine data."""
        records = []
        for m in machine_list:
            for seg in m["segments"]:
                start = seg["start"]
                end = seg["end"]
                gt_dims = seg["dimensions"]

                start_idx = max(0, int(start - fine_margin))
                end_idx = min(len(m["norm_test"]), int(end + fine_margin) + 1)
                fine_win = m["norm_test"][start_idx:end_idx]

                data_sample = build_pyg_data_from_window(fine_win, edge_index, edge_attr, device)
                ig_scores = compute_integrated_gradients_attribution(model, data_sample, steps=ig_steps)

                records.append({
                    "machine_id": m["machine_id"],
                    "start": start,
                    "end": end,
                    "gt_dimensions": gt_dims,
                    "ig_scores": ig_scores,
                    "pct_ood": m["pct_ood"],
                    "group": m["group"],
                })
        return records

    print("\n[Computing IG attributions for LOW-SHIFT machines...]")
    low_records = extract_segments_from_machines(low_shift_machines)
    print(f"  Processed {len(low_records)} segments from {len(low_shift_machines)} low-shift machines.")

    print("[Computing IG attributions for HIGH-SHIFT machines...]")
    high_records = extract_segments_from_machines(high_shift_machines)
    print(f"  Processed {len(high_records)} segments from {len(high_shift_machines)} high-shift machines.\n")

    def evaluate_group(records: List[Dict], name: str) -> Dict:
        if not records:
            return {}
        cal_res = calibrate_attribution(
            test_ig_scores=[r["ig_scores"] for r in records],
            calibration_segments=records,
            alpha=alpha,
            one_indexed=True
        )
        contained = 0
        recalls = []
        for i, rec in enumerate(records):
            gt_set = set(rec["gt_dimensions"])
            conf_set = set(cal_res["conformal_sensor_sets"][i])
            overlap = gt_set.intersection(conf_set)
            if gt_set.issubset(conf_set):
                contained += 1
            recalls.append((len(overlap) / len(gt_set)) * 100.0 if gt_set else 0.0)
        return {
            "name": name,
            "n": len(records),
            "tau_hat": cal_res["tau_hat"],
            "mean_set_size": cal_res["mean_set_size"],
            "mean_set_pct": cal_res["mean_set_size"] / 38 * 100.0,
            "full_containment": contained,
            "containment_rate": contained / len(records) * 100.0,
            "mean_recall": float(np.mean(recalls)),
        }

    low_stats = evaluate_group(low_records, "LOW-SHIFT")
    high_stats = evaluate_group(high_records, "HIGH-SHIFT")
    all_stats = evaluate_group(low_records + high_records, "ALL (pooled)")

    print("\n" + "=" * 100)
    print(f"DIAGNOSTIC 2 — CALIBRATION RESULTS BY DISTRIBUTION SHIFT GROUP")
    print("=" * 100)
    print(f"{'Metric':<45} | {'LOW-SHIFT':<22} | {'HIGH-SHIFT':<22} | {'ALL Pooled':<20}")
    print("-" * 100)
    print(f"{'Sample Size (n)':<45} | {low_stats['n']:<22} | {high_stats['n']:<22} | {all_stats['n']:<20}")
    print(f"{'Conformal tau_hat':<45} | {low_stats['tau_hat']:<22.4f} | {high_stats['tau_hat']:<22.4f} | {all_stats['tau_hat']:<20.4f}")
    print(f"{'Average Conformal Set Size (sensors)':<45} | {low_stats['mean_set_size']:<22.1f} | {high_stats['mean_set_size']:<22.1f} | {all_stats['mean_set_size']:<20.1f}")
    print(f"{'Average Set Size % of 38 sensors':<45} | {low_stats['mean_set_pct']:<22.1f} | {high_stats['mean_set_pct']:<22.1f} | {all_stats['mean_set_pct']:<20.1f}")
    print(f"{'Full GT Containment Rate':<45} | {low_stats['full_containment']}/{low_stats['n']} ({low_stats['containment_rate']:.1f}%)  | {high_stats['full_containment']}/{high_stats['n']} ({high_stats['containment_rate']:.1f}%)  | {all_stats['full_containment']}/{all_stats['n']} ({all_stats['containment_rate']:.1f}%)")
    print(f"{'Mean Root-Cause Sensor Recall %':<45} | {low_stats['mean_recall']:<22.2f} | {high_stats['mean_recall']:<22.2f} | {all_stats['mean_recall']:<20.2f}")
    print("=" * 100 + "\n")

    return {
        "low_shift": low_stats,
        "high_shift": high_stats,
        "all": all_stats,
        "low_records": low_records,
        "high_records": high_records,
    }


def print_final_summary(diag1_results: List[Dict], diag2_results: Dict) -> None:
    """Print a clear interpretive summary of both diagnostics."""
    avg_s90 = np.mean([r["sensors_at_90pct"] for r in diag1_results])
    avg_gini = np.mean([r["gini"] for r in diag1_results])
    avg_top5 = np.mean([r["top5_mass"] for r in diag1_results]) * 100
    avg_top8 = np.mean([r["top8_mass"] for r in diag1_results]) * 100

    low = diag2_results.get("low_shift", {})
    high = diag2_results.get("high_shift", {})
    shift_delta = high.get("mean_set_size", 0) - low.get("mean_set_size", 0) if low and high else 0

    print("\n" + "=" * 100)
    print("FINAL SUMMARY: WHAT EXPLAINS THE LARGE CONFORMAL SET SIZE (34-35/38 SENSORS)?")
    print("=" * 100)

    print("\n[DIAGNOSTIC 1 — Inherent IG Diffuseness (in-distribution, machine-1-1 only)]")
    print(f"  - Average top-5 sensors capture only {avg_top5:.1f}% of total IG mass.")
    print(f"  - Average top-8 sensors capture only {avg_top8:.1f}% of total IG mass.")
    print(f"  - Average sensors needed to reach 90% mass: {avg_s90:.1f} / 38.")
    print(f"  - Average Gini coefficient: {avg_gini:.4f} (0=perfectly uniform, 1=fully concentrated)")

    if avg_s90 >= 30:
        d1_verdict = "MAJOR DRIVER"
        d1_explain = (
            f"IG attribution is nearly uniform — even in-distribution it needs {avg_s90:.0f}/38 sensors "
            f"to capture 90% mass. The [mean, std] 2-D feature representation is too compressed to "
            f"produce sparse gradient signals, spreading attribution across all sensors."
        )
    elif avg_s90 >= 15:
        d1_verdict = "MODERATE DRIVER"
        d1_explain = (
            f"IG is moderately diffuse ({avg_s90:.0f}/38 sensors for 90% mass). "
            f"The model can concentrate to some degree, but the feature space is limiting."
        )
    else:
        d1_verdict = "MINOR DRIVER"
        d1_explain = (
            f"IG is reasonably concentrated ({avg_s90:.0f}/38 sensors for 90% mass). "
            f"In-distribution diffuseness is NOT the primary cause."
        )
    print(f"  -> Verdict: {d1_verdict}. {d1_explain}")

    print(f"\n[DIAGNOSTIC 2 — Distribution Shift (cross-machine OOD pooling)]")
    if low and high:
        print(f"  - LOW-SHIFT  machines: average conformal set size = {low['mean_set_size']:.1f} / 38 ({low['mean_set_pct']:.1f}%)")
        print(f"  - HIGH-SHIFT machines: average conformal set size = {high['mean_set_size']:.1f} / 38 ({high['mean_set_pct']:.1f}%)")
        print(f"  - OOD-induced set size increase (HIGH - LOW): {shift_delta:+.1f} sensors")

        if abs(shift_delta) >= 3:
            d2_verdict = "MAJOR DRIVER"
            d2_explain = (
                f"High-shift machines inflate the conformal cumulative IG threshold by "
                f"{shift_delta:+.1f} sensors. Cross-machine OOD data significantly worsens calibration."
            )
        elif abs(shift_delta) >= 1:
            d2_verdict = "MODERATE DRIVER"
            d2_explain = f"Moderate contribution: {shift_delta:+.1f} sensor difference between groups."
        else:
            d2_verdict = "MINOR DRIVER"
            d2_explain = f"Low-shift and high-shift machines produce similar set sizes ({shift_delta:+.1f} sensors difference)."
        print(f"  -> Verdict: {d2_verdict}. {d2_explain}")
    else:
        d2_verdict = "NOT COMPUTED"
        d2_explain = "Insufficient data for one or both groups."
        print(f"  -> Verdict: {d2_verdict}.")

    # Combined conclusion
    print("\n[COMBINED CONCLUSION]")
    both_major = (d1_verdict == "MAJOR DRIVER") and (d2_verdict == "MAJOR DRIVER")
    d1_only = (d1_verdict == "MAJOR DRIVER") and (d2_verdict != "MAJOR DRIVER")
    d2_only = (d1_verdict != "MAJOR DRIVER") and (d2_verdict == "MAJOR DRIVER")

    if both_major:
        print("  Both IG diffuseness and cross-machine distribution shift contribute substantially.")
        print("  Recommendations to improve conformal set tightness:")
        print("    1. Enrich node features beyond [mean, std] (e.g., add FFT dominant frequency,")
        print("       skewness, kurtosis, or residuals from an auxiliary predictor) to sharpen IG gradients.")
        print("    2. Train per-machine normalizers or include a fine-tuning step for cross-machine transfer.")
        print("    3. Filter pooled calibration to low-shift machines only for deployment.")
    elif d1_only:
        print("  PRIMARY CAUSE: Inherent IG diffuseness from [mean, std] feature compression.")
        print("  Distribution shift has minor impact — adding richer per-sensor features")
        print("  (frequency content, temporal trend, residuals) would directly tighten conformal sets.")
    elif d2_only:
        print("  PRIMARY CAUSE: Cross-machine distribution shift in the pooled calibration set.")
        print("  The model CAN produce sparse attribution in-distribution.")
        print("  Filtering pooled calibration to low-shift machines only would tighten sets significantly.")
    else:
        print("  Both causes are minor / comparable — the conformal guarantee of 90% coverage is")
        print("  inherently non-trivial to tighten without changing the attribution method or feature set.")

    print("=" * 100 + "\n")


def main():
    config_path = "configs/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    alpha = float(cfg.get("conformal_alpha", 0.10))
    fine_margin = int(cfg.get("fine_window_margin", 15))
    device = torch.device(cfg.get("device", "cpu"))
    checkpoint_path = "results/gnn_checkpoint.pt"
    raw_dir = cfg.get("dataset_path", "data/raw")

    print("=" * 100)
    print("ATTRIBUTION DIAGNOSTIC MODULE — Read-only analysis, no pipeline modification")
    print(f"Target: Explain large conformal set size (34-35/38 sensors) | Alpha: {alpha} | Fine Margin: {fine_margin}")
    print("=" * 100)

    # Load machine-1-1 data
    print("\n[Setup] Loading machine-1-1 reference data...")
    _, _, interp_labels_11, meta_11 = load_dataset(
        dataset_name=cfg.get("dataset_name", "SMD"),
        raw_dir=raw_dir,
        machine_id="machine-1-1",
        window_size=cfg.get("window_size", 100),
        stride=cfg.get("window_stride", 1),
        corr_threshold=cfg.get("edge_corr_threshold", 0.6)
    )

    model = GATAnomalyDetector(
        in_dim=6, hidden_dim=cfg.get("gnn_hidden_dim", 64),
        num_heads=cfg.get("gnn_num_heads", 4), edge_dim=1, dropout=0.2
    ).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded GNN checkpoint from epoch {ckpt.get('epoch', 'N/A')}.")
    model.eval()

    # ---- Diagnostic 1 ----
    diag1_results = diagnostic_1_ig_concentration(
        model=model,
        norm_test_data=meta_11["norm_test"],
        interp_labels=interp_labels_11,
        edge_index=meta_11["edge_index"],
        edge_attr=meta_11["edge_attr"],
        fine_margin=fine_margin,
        ig_steps=50
    )

    # ---- Diagnostic 2 ----
    from src.calibration.pooled_conformal import discover_all_smd_interpretation_labels
    machines_meta, _ = discover_all_smd_interpretation_labels(raw_dir=raw_dir)

    diag2_results = diagnostic_2_split_by_distribution_shift(
        model=model,
        machines_meta=machines_meta,
        norm_stats_11=meta_11["norm_stats"],
        edge_index=meta_11["edge_index"],
        edge_attr=meta_11["edge_attr"],
        raw_dir=raw_dir,
        fine_margin=fine_margin,
        ig_steps=50,
        low_shift_threshold_pct=10.0,
        alpha=alpha
    )

    # ---- Final Summary ----
    print_final_summary(diag1_results, diag2_results)


if __name__ == "__main__":
    main()
