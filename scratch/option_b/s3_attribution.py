"""Stage S3: Sensor Attribution Benchmark across all 327 Labelled Segments."""

import os
import sys
import glob
import json
import time
import hashlib
from typing import Dict, Any, List, Tuple, Set

def get_file_sha256(filepath: str) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data as PyGData

sys.path.insert(0, os.path.abspath("."))
from src.data_loader import (
    load_smd_raw, normalize_data, compute_sensor_correlation_graph,
    compute_node_features_6d, parse_interpretation_labels
)
from src.gnn_attribution.model import GATAnomalyDetector, get_node_attribution
from src.gnn_attribution.validate_attribution import (
    compute_attention_attribution,
    compute_integrated_gradients_attribution,
    compute_gnnexplainer_attribution,
    get_topk_sensor_dimensions
)
from scratch.option_b.s1_detectors import cluster_bootstrap_ci

def compute_attribution_metrics(
    scores: torch.Tensor,
    gt_dims: Set[int],
    top_k: int = 5
) -> Dict[str, float]:
    """Compute top5_recall, top1_hit, and legacy_overlap from attribution score tensor."""
    k_gt = len(gt_dims)
    if k_gt == 0:
        return {
            "top5_recall": float("nan"),
            "top1_hit": float("nan"),
            "legacy_overlap": float("nan")
        }
        
    scores_np = scores.cpu().numpy()
    
    # 1. Top-5 recall
    top5_set = get_topk_sensor_dimensions(scores, k=top_k) # 1-indexed
    top5_recall = float(len(top5_set.intersection(gt_dims)) / k_gt)
    
    # 2. Top-1 hit
    top1_set = get_topk_sensor_dimensions(scores, k=1)
    top1_dim = list(top1_set)[0]
    top1_hit = 1.0 if top1_dim in gt_dims else 0.0
    
    # 3. Legacy overlap: top-k sensors where k = len(gt_dims)
    topk_legacy_set = get_topk_sensor_dimensions(scores, k=k_gt)
    legacy_overlap = float(len(topk_legacy_set.intersection(gt_dims)) / k_gt)
    
    return {
        "top5_recall": top5_recall,
        "top1_hit": top1_hit,
        "legacy_overlap": legacy_overlap
    }

def run_s3_attribution():
    print("=" * 70)
    print("STAGE S3 — Sensor Attribution on All 327 Segments")
    print("=" * 70)
    
    t0 = time.time()
    device = torch.device("cpu")
    torch.set_num_threads(1)
    
    # Gather all machines and interpretation segments
    raw_dir = "data/raw"
    interp_dir = os.path.join(raw_dir, "SMD", "interpretation_label")
    interp_files = sorted(glob.glob(os.path.join(interp_dir, "*.txt")))
    machines = [os.path.basename(f).replace(".txt", "") for f in interp_files]
    
    all_segments = []
    for mid in machines:
        ipath = os.path.join(interp_dir, f"{mid}.txt")
        segs = parse_interpretation_labels(ipath)
        for s_idx, s in enumerate(segs, 1):
            all_segments.append({
                "machine": mid,
                "segment_index": s_idx,
                "start": s["start"],
                "end": s["end"],
                "dimensions": s["dimensions"]
            })
            
    assert len(all_segments) == 327, f"Expected 327 segments, got {len(all_segments)}"
    print(f"Loaded {len(all_segments)} interpretation segments across {len(machines)} machines.")
    
    # Load detectors and cache data per machine
    machine_cache = {}
    for mid in machines:
        ckpt_path = f"results_b/checkpoints/{mid}_seed42.pt"
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Detector checkpoint not found: {ckpt_path}. Ensure S1 completed!")
            
        train_raw, test_raw, _, _ = load_smd_raw("data/raw", mid)
        train_norm, test_norm, norm_stats = normalize_data(train_raw, test_raw)
        edge_index, edge_attr = compute_sensor_correlation_graph(train_norm, threshold=0.6)
        
        # Load GNN
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model = GATAnomalyDetector(in_dim=6, hidden_dim=64, num_heads=4, edge_dim=1, dropout=0.2)
        model.load_state_dict(state_dict)
        model.eval()
        
        machine_cache[mid] = {
            "model": model,
            "norm_test": test_norm,
            "raw_test": test_raw,
            "norm_stats": norm_stats,
            "edge_index": edge_index,
            "edge_attr": edge_attr,
            "ckpt_sha256": get_file_sha256(ckpt_path)
        }
        
    # Benchmark GNNExplainer on 5 segments first
    print("\nBenchmarking GNNExplainer on 5 segments to test projection...")
    t_gnnexp_start = time.time()
    for seg in all_segments[:5]:
        mid = seg["machine"]
        mc = machine_cache[mid]
        s_idx = max(0, int(seg["start"] - 15))
        e_idx = min(len(mc["norm_test"]), int(seg["end"] + 15) + 1)
        fine_win = mc["norm_test"][s_idx:e_idx]
        node_feats = compute_node_features_6d(fine_win)
        data = PyGData(
            x=torch.tensor(node_feats, dtype=torch.float32),
            edge_index=torch.tensor(mc["edge_index"], dtype=torch.long),
            edge_attr=torch.tensor(mc["edge_attr"], dtype=torch.float32) if mc["edge_attr"] is not None else None
        )
        _ = compute_gnnexplainer_attribution(mc["model"], data, epochs=60)
    gnnexp_5_time = time.time() - t_gnnexp_start
    gnnexp_per_seg = gnnexp_5_time / 5.0
    projected_gnnexp_327_h = (gnnexp_per_seg * 327) / 3600.0
    print(f"5 GNNExplainer runs took: {gnnexp_5_time:.2f}s ({gnnexp_per_seg:.2f}s/seg). Projected 327 time: {projected_gnnexp_327_h:.2f}h")
    
    use_gnnexp_subset = False
    gnnexp_subset_indices = set(range(327))
    if projected_gnnexp_327_h > 3.0:
        use_gnnexp_subset = True
        rng = np.random.RandomState(42)
        subset_100 = sorted(rng.choice(327, size=100, replace=False).tolist())
        gnnexp_subset_indices = set(subset_100)
        subset_file = "results_b/gnnexplainer_subset_100.json"
        with open(subset_file, "w", encoding="utf-8") as f:
            json.dump({
                "projected_hours_full": projected_gnnexp_327_h,
                "subset_size": 100,
                "seed": 42,
                "subset_segment_indices": subset_100
            }, f, indent=4)
        print(f"Projected GNNExplainer time > 3h! Using fixed random subset of 100 segments (seed 42), saved to {subset_file}")
    else:
        print("Projected GNNExplainer time <= 3h. Running GNNExplainer on all 327 segments.")
        
    # Evaluate all 327 segments
    segment_results = []
    cached_rankings = []
    
    print("\nRunning Attribution on all 327 segments (IG, Attention, GNNExplainer)...")
    for idx, seg in enumerate(all_segments):
        mid = seg["machine"]
        mc = machine_cache[mid]
        model = mc["model"]
        gt_dims = set(seg["dimensions"])
        
        # Fine window extraction (margin = 15)
        s_idx = max(0, int(seg["start"] - 15))
        e_idx = min(len(mc["norm_test"]), int(seg["end"] + 15) + 1)
        fine_norm_win = mc["norm_test"][s_idx:e_idx]
        fine_raw_win = mc["raw_test"][s_idx:e_idx]
        
        node_feats = compute_node_features_6d(fine_norm_win)
        ei_tensor = torch.tensor(mc["edge_index"], dtype=torch.long)
        ea_tensor = torch.tensor(mc["edge_attr"], dtype=torch.float32) if mc["edge_attr"] is not None else None
        
        data = PyGData(x=torch.tensor(node_feats, dtype=torch.float32), edge_index=ei_tensor, edge_attr=ea_tensor)
        
        # 1. Integrated Gradients
        ig_scores = compute_integrated_gradients_attribution(model, data, steps=50)
        ig_metrics = compute_attribution_metrics(ig_scores, gt_dims, top_k=5)
        segment_results.append({
            "segment_global_id": idx + 1,
            "machine": mid,
            "segment_index": seg["segment_index"],
            "method": "Integrated_Gradients",
            "start": seg["start"],
            "end": seg["end"],
            "k_gt": len(gt_dims),
            "top5_recall": ig_metrics["top5_recall"],
            "top1_hit": ig_metrics["top1_hit"],
            "legacy_overlap": ig_metrics["legacy_overlap"]
        })
        
        # Save ranking for conformal and S5
        ig_np = ig_scores.cpu().numpy()
        sorted_sensors_1indexed = [int(i + 1) for i in np.argsort(ig_np)[::-1]]
        cached_rankings.append({
            "segment_global_id": idx + 1,
            "machine": mid,
            "segment_index": seg["segment_index"],
            "start": seg["start"],
            "end": seg["end"],
            "gt_dimensions": seg["dimensions"],
            "ig_scores": ig_np.tolist(),
            "ig_ranking_1indexed": sorted_sensors_1indexed,
            "fine_raw_window": fine_raw_win.tolist(),
            "norm_stats_mean": mc["norm_stats"]["mean"].tolist(),
            "norm_stats_std": mc["norm_stats"]["std"].tolist()
        })
        
        # 2. Layer-2 Attention
        attn_scores = compute_attention_attribution(model, data)
        attn_metrics = compute_attribution_metrics(attn_scores, gt_dims, top_k=5)
        segment_results.append({
            "segment_global_id": idx + 1,
            "machine": mid,
            "segment_index": seg["segment_index"],
            "method": "Layer2_Attention",
            "start": seg["start"],
            "end": seg["end"],
            "k_gt": len(gt_dims),
            "top5_recall": attn_metrics["top5_recall"],
            "top1_hit": attn_metrics["top1_hit"],
            "legacy_overlap": attn_metrics["legacy_overlap"]
        })
        
        # 3. GNNExplainer
        if (not use_gnnexp_subset) or (idx in gnnexp_subset_indices):
            gnnexp_scores = compute_gnnexplainer_attribution(model, data, epochs=60)
            gnnexp_metrics = compute_attribution_metrics(gnnexp_scores, gt_dims, top_k=5)
            segment_results.append({
                "segment_global_id": idx + 1,
                "machine": mid,
                "segment_index": seg["segment_index"],
                "method": "GNNExplainer",
                "start": seg["start"],
                "end": seg["end"],
                "k_gt": len(gt_dims),
                "top5_recall": gnnexp_metrics["top5_recall"],
                "top1_hit": gnnexp_metrics["top1_hit"],
                "legacy_overlap": gnnexp_metrics["legacy_overlap"]
            })
            
        if (idx + 1) % 50 == 0 or (idx + 1) == 327:
            print(f"  Processed {idx + 1}/327 segments...")
            
    df_seg = pd.DataFrame(segment_results)
    seg_csv = "results_b/attribution_per_segment.csv"
    df_seg.to_csv(seg_csv, index=False)
    print(f"\nSaved attribution per segment to {seg_csv}")
    
    # Save cached rankings for S4 and S5
    cache_json = "results_b/s3_cached_rankings.json"
    with open(cache_json, "w", encoding="utf-8") as f:
        json.dump(cached_rankings, f)
    print(f"Saved cached segment rankings to {cache_json}")
    
    # -------------------------------------------------------------
    # Attribution Summary with Cluster Bootstrap CIs by Machine
    # -------------------------------------------------------------
    summary_rows = []
    methods = ["Integrated_Gradients", "Layer2_Attention", "GNNExplainer"]
    
    # Precompute per-machine means for each method
    machine_means = {}
    for mth in methods:
        sub = df_seg[df_seg["method"] == mth]
        m_grouped = sub.groupby("machine")[["top5_recall", "top1_hit", "legacy_overlap"]].mean()
        machine_means[mth] = m_grouped
        
    for mth in methods:
        m_df = machine_means[mth]
        
        t5_m, t5_sd, t5_l, t5_u = cluster_bootstrap_ci(dict(zip(m_df.index, m_df["top5_recall"])))
        t1_m, t1_sd, t1_l, t1_u = cluster_bootstrap_ci(dict(zip(m_df.index, m_df["top1_hit"])))
        leg_m, leg_sd, leg_l, leg_u = cluster_bootstrap_ci(dict(zip(m_df.index, m_df["legacy_overlap"])))
        
        row = {
            "method": mth,
            "n_segments": len(df_seg[df_seg["method"] == mth]),
            "top5_recall_mean": t5_m,
            "top5_recall_sd": t5_sd,
            "top5_recall_95ci": f"[{t5_l:.4f}, {t5_u:.4f}]",
            "top1_hit_mean": t1_m,
            "top1_hit_sd": t1_sd,
            "top1_hit_95ci": f"[{t1_l:.4f}, {t1_u:.4f}]",
            "legacy_overlap_mean": leg_m,
            "legacy_overlap_sd": leg_sd,
            "legacy_overlap_95ci": f"[{leg_l:.4f}, {leg_u:.4f}]"
        }
        
        # Paired differences vs IG
        if mth != "Integrated_Gradients":
            ig_df = machine_means["Integrated_Gradients"]
            common_m = sorted(list(set(m_df.index).intersection(set(ig_df.index))))
            
            diff_t5 = {m: ig_df.loc[m, "top5_recall"] - m_df.loc[m, "top5_recall"] for m in common_m}
            diff_t1 = {m: ig_df.loc[m, "top1_hit"] - m_df.loc[m, "top1_hit"] for m in common_m}
            diff_leg = {m: ig_df.loc[m, "legacy_overlap"] - m_df.loc[m, "legacy_overlap"] for m in common_m}
            
            d_t5_m, _, d_t5_l, d_t5_u = cluster_bootstrap_ci(diff_t5)
            d_t1_m, _, d_t1_l, d_t1_u = cluster_bootstrap_ci(diff_t1)
            d_leg_m, _, d_leg_l, d_leg_u = cluster_bootstrap_ci(diff_leg)
            
            row["diff_vs_ig_top5_recall"] = f"{d_t5_m:+.4f} [{d_t5_l:+.4f}, {d_t5_u:+.4f}]"
            row["diff_vs_ig_top1_hit"] = f"{d_t1_m:+.4f} [{d_t1_l:+.4f}, {d_t1_u:+.4f}]"
            row["diff_vs_ig_legacy_overlap"] = f"{d_leg_m:+.4f} [{d_leg_l:+.4f}, {d_leg_u:+.4f}]"
        else:
            row["diff_vs_ig_top5_recall"] = "reference"
            row["diff_vs_ig_top1_hit"] = "reference"
            row["diff_vs_ig_legacy_overlap"] = "reference"
            
        summary_rows.append(row)
        
    df_summary = pd.DataFrame(summary_rows)
    sum_csv = "results_b/attribution_summary.csv"
    df_summary.to_csv(sum_csv, index=False)
    print(f"Saved attribution summary to {sum_csv}")
    
    # Print clean summary
    print("\n--- ATTRIBUTION BENCHMARK SUMMARY ---")
    for _, r in df_summary.iterrows():
        print(f"\nMethod: {r['method']} (N={r['n_segments']})")
        print(f"  Top-5 Recall:   {r['top5_recall_mean']*100:.2f}% +/- {r['top5_recall_sd']*100:.2f}%  95% CI: {r['top5_recall_95ci']}  (Diff vs IG: {r['diff_vs_ig_top5_recall']})")
        print(f"  Top-1 Hit:      {r['top1_hit_mean']*100:.2f}% +/- {r['top1_hit_sd']*100:.2f}%  95% CI: {r['top1_hit_95ci']}  (Diff vs IG: {r['diff_vs_ig_top1_hit']})")
        print(f"  Legacy Overlap: {r['legacy_overlap_mean']*100:.2f}% +/- {r['legacy_overlap_sd']*100:.2f}%  95% CI: {r['legacy_overlap_95ci']}  (Diff vs IG: {r['diff_vs_ig_legacy_overlap']})")
        
    elapsed = time.time() - t0
    print(f"\nStage S3 Attribution completed in {elapsed:.2f}s ({elapsed/60.0:.2f} min)")
    
    with open("results_b/LOG.md", "a", encoding="utf-8") as f:
        f.write(f"\n## Stage S3 — Attribution Benchmark Completed\n")
        f.write(f"- Wall-clock: {elapsed:.2f}s\n")
        f.write(f"- Segments evaluated: 327 (IG, Attention) and {'100 subset' if use_gnnexp_subset else '327'} (GNNExplainer)\n")
        f.write(f"- Outputs: `results_b/attribution_per_segment.csv`, `results_b/attribution_summary.csv`\n")

if __name__ == "__main__":
    run_s3_attribution()
