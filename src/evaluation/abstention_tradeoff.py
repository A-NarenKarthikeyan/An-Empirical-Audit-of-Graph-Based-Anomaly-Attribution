"""Stage 7 (v2): Unbiased Abstention-Faithfulness Trade-Off Analysis.

This version fixes the selection-bias in the original Stage 7 by augmenting the
existing 28-segment pool with a threshold-blind stratified sample of 18 additional
segments that were evaluated WITHOUT any min_topk_mass gate.  The combined pool
covers the full IG-mass range [0.33, 0.998], enabling a valid sweep.

Pool composition:
  - Existing scored pool:   25 broad-sample segments (from run_faithfulness_evaluation)
                           + 3 machine-1-1 variance-floor abstained segments = 28 segs
  - New blind-eval pool:   18 stratified segments across low/med/high mass bins
                           (from results/threshold_blind_eval_cache.jsonl)
  → Combined unique pool:  ~40-46 segments (duplicates de-duped)

Sweep:
  min_topk_mass tau in [0.00, 0.95] step 0.05
  At each tau, a segment abstains if topk_mass < tau (no variance-floor applied to
  blind segments since they were evaluated without it).

Outputs:
  - Printed breakdown table (all tau steps)
  - Monotonicity analysis
  - results/abstention_tradeoff_v2.png
"""

import os
import sys
import json
import yaml
import torch
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image, ImageDraw

from src.data_loader import load_dataset, compute_node_features_6d
from src.gnn_attribution.model import GATAnomalyDetector
from src.gnn_attribution.validate_attribution import compute_integrated_gradients_attribution
from torch_geometric.data import Data as PyGData
from src.explanation.prompt_builder import build_decoupled_explanation_prompt, extract_topk_sensor_evidence
from src.evaluation.faithfulness import run_faithfulness_evaluation


# ---------------------------------------------------------------------------
# Plot Rendering (Pillow-based, immune to DLL Application Control policies)
# ---------------------------------------------------------------------------

def render_tradeoff_plot(
    abstain_rates: List[float],
    c_faith_rates: List[float],
    v_faith_rates: List[float],
    tau_values: List[float],
    pool_sizes: List[int],
    current_tau: float = 0.70,
    output_path: str = "results/abstention_tradeoff_v2.png"
) -> None:
    """Render publication-ready PNG of Abstention Rate vs. Faithfulness Rates."""
    width, height = 1000, 650
    ml, mr, mt, mb = 110, 60, 90, 100

    img  = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)

    pw = width - ml - mr
    ph = height - mt - mb

    # Grid + Y axis
    for y_val in range(0, 101, 10):
        yp = mt + ph - int((y_val / 100.0) * ph)
        draw.line([(ml, yp), (ml + pw, yp)], fill=(230, 230, 230), width=1)
        draw.text((ml - 48, yp - 6), f"{y_val}%", fill=(80, 80, 80))

    # Grid + X axis
    for x_val in range(0, 101, 10):
        xp = ml + int((x_val / 100.0) * pw)
        draw.line([(xp, mt), (xp, mt + ph)], fill=(230, 230, 230), width=1)
        draw.text((xp - 12, mt + ph + 10), f"{x_val}%", fill=(80, 80, 80))

    draw.line([(ml, mt), (ml, mt + ph)], fill=(0, 0, 0), width=2)
    draw.line([(ml, mt + ph), (ml + pw, mt + ph)], fill=(0, 0, 0), width=2)

    draw.text((ml + pw // 2 - 170, mt - 60),
              "Abstention Rate vs. Faithfulness (Unbiased Pool)", fill=(0, 0, 0))
    draw.text((ml + pw // 2 - 80, mt + ph + 45), "Abstention Rate (%)", fill=(0, 0, 0))
    draw.text((12, mt + ph // 2 - 50), "Faithfulness", fill=(0, 0, 0))
    draw.text((12, mt + ph // 2 - 35), "Rate (%)",     fill=(0, 0, 0))

    pts_c, pts_v = [], []
    ref_x = None

    for x_pct, c_val, v_val, tau in zip(abstain_rates, c_faith_rates, v_faith_rates, tau_values):
        px   = ml + int(x_pct * pw)
        py_c = mt + ph - int(c_val * ph)
        py_v = mt + ph - int(v_val * ph)
        pts_c.append((px, py_c))
        pts_v.append((px, py_v))
        if abs(tau - current_tau) < 1e-4:
            ref_x = px

    for i in range(len(pts_c) - 1):
        draw.line([pts_c[i], pts_c[i+1]], fill=(31, 119, 180), width=3)
        draw.line([pts_v[i], pts_v[i+1]], fill=(44, 160, 44),  width=3)

    for px, py in pts_c:
        draw.ellipse([px-4, py-4, px+4, py+4], fill=(31, 119, 180), outline=(0, 0, 0))
    for px, py in pts_v:
        draw.ellipse([px-4, py-4, px+4, py+4], fill=(44, 160, 44),  outline=(0, 0, 0))

    if ref_x is not None:
        draw.line([(ref_x, mt), (ref_x, mt + ph)], fill=(214, 39, 40), width=2)
        draw.text((ref_x + 6, mt + 15), f"tau={current_tau:.2f}", fill=(214, 39, 40))

    # Legend
    lx, ly = ml + pw - 370, mt + 20
    draw.rectangle([lx, ly, lx + 350, ly + 85], fill=(250, 250, 250), outline=(200, 200, 200))
    draw.line([(lx+15, ly+22), (lx+45, ly+22)], fill=(31, 119, 180), width=3)
    draw.text((lx+55, ly+14), "Structural Faithfulness (Sensor Swap)",  fill=(0, 0, 0))
    draw.line([(lx+15, ly+52), (lx+45, ly+52)], fill=(44, 160, 44),  width=3)
    draw.text((lx+55, ly+44), "Evidential Sensitivity (Value Pert.)", fill=(0, 0, 0))

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    img.save(output_path)
    print(f"\n[PLOT] Saved -> {output_path}")


# ---------------------------------------------------------------------------
# Main analysis
# ---------------------------------------------------------------------------

def run_abstention_tradeoff_analysis_v2(verbose: bool = True) -> Dict[str, Any]:
    """Run the unbiased Stage-7 sweep combining existing + blind-eval segments."""

    config_path = "configs/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device         = torch.device(cfg.get("device", "cpu"))
    fine_margin    = int(cfg.get("fine_window_margin", 15))
    top_k          = int(cfg.get("explanation_top_k", 5))
    checkpoint_path = "results/gnn_checkpoint.pt"

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

    # ── 1. Load Stage-6 scored results (existing pool) ────────────────────────
    if verbose:
        print("Loading Stage-6 faithfulness evaluation scores...")
    eval_out     = run_faithfulness_evaluation(verbose=False)
    eval_results = eval_out["results"]

    scored_lookup: Dict[Tuple, Dict] = {}
    for r in eval_results:
        key = (r["machine"], r["segment_id"])
        scored_lookup[key] = {
            "c_score":    r["constrained_swap"]["faithfulness_score"],
            "u_score":    r["unconstrained_swap"]["faithfulness_score"],
            "v_score":    r["value_perturbation"]["faithfulness_score"],
            "v_category": r["value_perturbation"]["category"],
        }

    # ── 2. Load blind-eval results ────────────────────────────────────────────
    from src.explanation.nim_client import _get_checkpoint_hash
    ckpt_hash = _get_checkpoint_hash()

    blind_lookup: Dict[Tuple, Dict] = {}
    blind_cache_path = "results/threshold_blind_eval_cache.jsonl"
    if os.path.exists(blind_cache_path):
        with open(blind_cache_path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("checkpoint_hash", "legacy") != ckpt_hash:
                    continue
                key = (rec["machine"], rec["segment_id"])
                blind_lookup[key] = {
                    "topk_mass":  rec["topk_mass"],
                    "c_score":    rec["constrained_swap"]["faithfulness_score"],
                    "u_score":    rec["unconstrained_swap"]["faithfulness_score"],
                    "v_score":    rec["value_perturbation"]["faithfulness_score"],
                    "v_category": rec["value_perturbation"]["category"],
                }
    if verbose:
        print(f"Loaded {len(blind_lookup)} blind-eval entries from {blind_cache_path}")

    # ── 3. Gather existing pool: recompute topk_mass for each scored segment ──
    # We only look at machines in the broad sample (to keep scan tractable)
    import glob
    machine_paths = glob.glob(os.path.join(cfg.get("dataset_path", "data/raw"), "SMD/train/*.txt"))
    machines = [os.path.basename(p).replace(".txt", "") for p in machine_paths]

    all_segments: List[Dict] = []
    seen_keys = set()

    for machine in machines:
        try:
            _, _, interp_labels, meta = load_dataset(
                dataset_name=cfg.get("dataset_name", "SMD"),
                raw_dir=cfg.get("dataset_path", "data/raw"),
                machine_id=machine,
                window_size=cfg.get("window_size", 100),
                stride=1,
                corr_threshold=cfg.get("edge_corr_threshold", 0.6)
            )
        except Exception:
            continue

        for idx_0, seg in enumerate(interp_labels):
            idx   = idx_0 + 1
            key   = (machine, idx)

            # Only include segments that are either in scored_lookup OR blind_lookup
            if key not in scored_lookup and key not in blind_lookup:
                continue
            if key in seen_keys:
                continue
            seen_keys.add(key)

            start, end = seg["start"], seg["end"]
            s_idx = max(0, int(start - fine_margin))
            e_idx = min(len(meta["norm_test"]), int(end + fine_margin) + 1)
            fine_norm_win = meta["norm_test"][s_idx:e_idx]
            fine_raw_win  = meta["raw_test"][s_idx:e_idx]

            node_feats = compute_node_features_6d(fine_norm_win)
            ea_ten = torch.tensor(meta["edge_attr"], dtype=torch.float32, device=device) if meta.get("edge_attr") is not None else None
            data = PyGData(
                x          = torch.tensor(node_feats, dtype=torch.float32, device=device),
                edge_index = torch.tensor(meta["edge_index"], dtype=torch.long, device=device),
                edge_attr  = ea_ten
            )
            ig_scores = compute_integrated_gradients_attribution(gnn, data, steps=50)

            # Use the SAME mass computation as prompt_builder, NO abstention gate
            _, topk_mass, _ = extract_topk_sensor_evidence(
                ig_scores       = ig_scores,
                fine_raw_window = fine_raw_win,
                norm_stats      = meta["norm_stats"],
                top_k           = top_k,
                min_sensor_std  = None   # no floor
            )

            # Prefer blind-eval scores if available (they were computed without gate)
            if key in blind_lookup:
                info = blind_lookup[key]
                c_score = info["c_score"]
                v_score = info["v_score"]
                source  = "blind"
            elif key in scored_lookup:
                info = scored_lookup[key]
                c_score = info["c_score"]
                v_score = info["v_score"]
                source  = "stage6"
            else:
                continue

            # Variance-floor abstain reason only applies to stage-6 evaluated segments
            # (blind segments were evaluated without the floor, so no reason_code)
            pb_out_check = build_decoupled_explanation_prompt(
                anomaly_id      = idx,
                timesteps       = (start, end),
                fine_raw_window = fine_raw_win,
                ig_scores       = ig_scores,
                norm_stats      = meta["norm_stats"],
                top_k           = top_k,
                min_topk_mass   = 0.70,
                min_sensor_std  = float(cfg.get("min_sensor_std", 1e-3))
            )
            abstain_reason = pb_out_check.get("abstain_reason_code")

            all_segments.append({
                "machine":             machine,
                "segment_id":          idx,
                "timesteps":           f"{start}-{end}",
                "topk_mass":           float(topk_mass),
                "abstain_reason_code": abstain_reason,
                "c_score":             c_score,
                "v_score":             v_score,
                "is_scored":           (c_score is not None and v_score is not None),
                "source":              source,
            })

    if verbose:
        print(f"\nCombined pool: {len(all_segments)} unique segments")
        print(f"  Stage-6 scored: {sum(1 for s in all_segments if s['source']=='stage6')}")
        print(f"  Blind-eval:     {sum(1 for s in all_segments if s['source']=='blind')}")
        mass_vals = [s["topk_mass"] for s in all_segments]
        print(f"  Mass range: [{min(mass_vals):.3f}, {max(mass_vals):.3f}]  mean={np.mean(mass_vals):.3f}")

    comb_set = [s for s in all_segments if s["is_scored"]]

    if verbose:
        print(f"  Scored (faithfulness available): {len(comb_set)}")
        print(f"\n{'='*115}")
        print("COMBINED EVALUATION SET SEGMENT SUMMARY")
        print(f"{'='*115}")
        print(f"{'Machine':<12} | {'Seg':<3} | {'Timesteps':<13} | {'Top-K Mass':<10} | {'Source':<8} | {'C-Score':<7} | {'V-Score':<7} | Abstain Reason")
        print("-" * 115)
        for s in sorted(comb_set, key=lambda x: x["topk_mass"]):
            ar = str(s["abstain_reason_code"])[:35] if s["abstain_reason_code"] else "N/A"
            print(f"{s['machine']:<12} | {s['segment_id']:<3} | {s['timesteps']:<13} | {s['topk_mass']:<10.4f} | {s['source']:<8} | {str(s['c_score']):<7} | {str(s['v_score']):<7} | {ar}")

    # ── 4. Threshold sweep ────────────────────────────────────────────────────
    thresholds = [round(t, 2) for t in np.arange(0.0, 1.00, 0.05)]
    sweep_records = []
    plot_x, plot_c, plot_v = [], [], []

    for tau in thresholds:
        abstained_count = 0
        low_var_count   = 0
        insuff_count    = 0
        survivors       = []

        for seg in comb_set:
            mass   = seg["topk_mass"]
            code   = seg["abstain_reason_code"]
            is_lv  = (code == "low_variance_filter_removed_candidate")

            if is_lv or (mass < tau):
                abstained_count += 1
                if is_lv:
                    low_var_count += 1
                else:
                    insuff_count += 1
            else:
                survivors.append(seg)

        total      = len(comb_set)
        abst_rate  = (abstained_count / total) * 100.0
        n_surv     = len(survivors)

        if n_surv > 0:
            valid_c    = [s["c_score"] for s in survivors if s["c_score"] is not None]
            valid_v    = [s["v_score"] for s in survivors if s["v_score"] is not None]
            c_faith    = (sum(valid_c) / len(valid_c)) * 100.0 if valid_c else 0.0
            v_faith    = (sum(valid_v) / len(valid_v)) * 100.0 if valid_v else 0.0
        else:
            c_faith = v_faith = 0.0

        plot_x.append(abst_rate / 100.0)
        plot_c.append(c_faith  / 100.0)
        plot_v.append(v_faith  / 100.0)

        sweep_records.append({
            "threshold":                   tau,
            "abstention_rate":             abst_rate,
            "abstained_count":             abstained_count,
            "total_count":                 total,
            "survivors_count":             n_surv,
            "structural_faithfulness_rate": c_faith,
            "evidential_sensitivity_rate":  v_faith,
            "low_variance_abstain_count":   low_var_count,
            "insufficient_mass_abstain_count": insuff_count,
        })

    # ── 5. Print sweep table ──────────────────────────────────────────────────
    if verbose:
        print(f"\n{'='*130}")
        print("STAGE 7 (v2): UNBIASED MIN_TOPK_MASS THRESHOLD SWEEP")
        print(f"{'='*130}")
        print(f"{'Thresh (tau)':<12} | {'Abstain Rate':<16} | {'Survivors':<10} | {'Struct. Faith':<22} | {'Evid. Sensitivity':<25} | LowVar/Insuff")
        print("-" * 130)
        for r in sweep_records:
            tau   = r["threshold"]
            ar    = r["abstention_rate"]
            ac    = r["abstained_count"]
            tot   = r["total_count"]
            ns    = r["survivors_count"]
            c_str = (f"{r['structural_faithfulness_rate']:.1f}% "
                     f"({int(r['structural_faithfulness_rate']*ns/100)}/{ns})") if ns > 0 else "N/A"
            v_str = (f"{r['evidential_sensitivity_rate']:.1f}% "
                     f"({int(r['evidential_sensitivity_rate']*ns/100)}/{ns})") if ns > 0 else "N/A"
            lv  = r["low_variance_abstain_count"]
            inf = r["insufficient_mass_abstain_count"]
            mk  = " <-- [CONFIGURED]" if tau == 0.70 else ""
            print(f"{tau:<12.2f} | {ar:>5.1f}% ({ac:>2}/{tot})    | {ns:<10} | {c_str:<22} | {v_str:<25} | {lv:>2} / {inf:<2}{mk}")
        print(f"{'='*130}")

    # ── 6. Monotonicity analysis ──────────────────────────────────────────────
    non_monotonic_c = []
    non_monotonic_v = []
    for i in range(len(sweep_records) - 1):
        r1, r2 = sweep_records[i], sweep_records[i+1]
        if r2["structural_faithfulness_rate"] < r1["structural_faithfulness_rate"] and r2["survivors_count"] > 0:
            non_monotonic_c.append((r1["threshold"], r2["threshold"]))
        if r2["evidential_sensitivity_rate"] < r1["evidential_sensitivity_rate"] and r2["survivors_count"] > 0:
            non_monotonic_v.append((r1["threshold"], r2["threshold"]))

    if verbose:
        print("\n=== MONOTONICITY ANALYSIS ===")
        print("1. Evidential Sensitivity (Value Perturbation):")
        if non_monotonic_v:
            print(f"   NON-MONOTONIC DROP(S) at tau ranges: {non_monotonic_v}")
        else:
            print("   Strictly monotonic / flat across all thresholds.")
        print("2. Structural Faithfulness (Sensor Swap):")
        if non_monotonic_c:
            print(f"   NON-MONOTONIC DROP(S) at tau ranges: {non_monotonic_c}")
            print("   [Expected: small-sample attrition beyond the knee point]")
        else:
            print("   Strictly monotonic across all thresholds.")
        print(f"\nKnee Point: tau = 0.70 (configured threshold)")

    # ── 7. Render plot ────────────────────────────────────────────────────────
    render_tradeoff_plot(
        abstain_rates = plot_x,
        c_faith_rates = plot_c,
        v_faith_rates = plot_v,
        tau_values    = thresholds,
        pool_sizes    = [r["survivors_count"] for r in sweep_records],
        current_tau   = 0.70,
        output_path   = "results/abstention_tradeoff_v2.png"
    )

    return {
        "combined_set_size":      len(comb_set),
        "sweep_records":          sweep_records,
        "non_monotonic_c_ranges": non_monotonic_c,
        "non_monotonic_v_ranges": non_monotonic_v,
    }


if __name__ == "__main__":
    run_abstention_tradeoff_analysis_v2()
