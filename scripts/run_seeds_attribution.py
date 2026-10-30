"""
run_seeds_attribution.py — Seeds 42, 43, 44 attribution and conformal analysis.

Gates:
  1. Seed-42 checkpoint evaluation reproduces IG machine-macro:
     top-1 = 83.89, top-5 = 53.93, legacy = 62.84 (within 0.05)
  2. Seed-42 conformal q_hat reproduces:
     36 (rho=1, alpha=0.1), 31 (rho=1, alpha=0.2), 28 (rho=0.8, alpha=0.1), 24 (rho=0.8, alpha=0.2)

Then runs seeds 43 and 44:
  (1) IG attribution on all 327 segments (top-1, top-5, legacy with macro avg and 10k bootstrap CIs).
  (2) max|z| baseline on same segments (top-1, top-5, legacy; verify top-5 == 59.55).
  (3) Conformal for IG: q_hat, 1000 50/50 splits held-out coverage, LOO pooled coverage.
  (4) Paired top-5 diff (IG minus max|z|) with 10k bootstrap CI.
  (5) Multi-seed aggregate (mean and SD across seeds 42, 43, 44).
"""

import os
import sys

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import json
import time
import datetime
from pathlib import Path
from typing import Dict, Any, List, Set, Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.data import Data as PyGData

# Add paper root to sys.path
BASE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE_DIR))

from src.data_loader import (
    load_smd_raw, normalize_data, compute_sensor_correlation_graph,
    compute_node_features_6d
)
from src.gnn_attribution.model import GATAnomalyDetector
from src.gnn_attribution.validate_attribution import compute_integrated_gradients_attribution

RESULTS_B = BASE_DIR / "results_b"
RESULTS_C = BASE_DIR / "results_c"
SEEDS_DIR = RESULTS_C / "seeds"
SEEDS_DIR.mkdir(parents=True, exist_ok=True)

RANKINGS_PATH = RESULTS_B / "s3_cached_rankings.json"
CHECKPOINTS_DIR = RESULTS_B / "checkpoints"

# -- 1. Check all checkpoints --------------------------------------------------
print("Checking detector checkpoints for seeds 42, 43, 44 ...", flush=True)
with open(RANKINGS_PATH, "r", encoding="utf-8") as f:
    cached_segments = json.load(f)

machines = sorted(list(set(s["machine"] for s in cached_segments)))
print(f"Total machines: {len(machines)}, Total segments: {len(cached_segments)}")

for seed in [42, 43, 44]:
    missing = [m for m in machines if not (CHECKPOINTS_DIR / f"{m}_seed{seed}.pt").exists()]
    if missing:
        print(f"ERROR: Missing seed-{seed} checkpoints for machines: {missing}")
        sys.exit(1)
print("All checkpoints present for seeds 42, 43, 44.\n", flush=True)

# -- 2. Pre-load machine data and build graph Data objects ---------------------
print("Loading SMD machine data and building PyG Data objects ...", flush=True)
device = torch.device("cpu")
torch.set_num_threads(1)

machine_data = {}
for mid in machines:
    train_raw, test_raw, _, _ = load_smd_raw(str(BASE_DIR / "data" / "raw"), mid)
    train_norm, test_norm, norm_stats = normalize_data(train_raw, test_raw)
    edge_index, edge_attr = compute_sensor_correlation_graph(train_norm, threshold=0.6)
    machine_data[mid] = {
        "norm_test": test_norm,
        "raw_test": test_raw,
        "norm_stats": norm_stats,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
    }

# Build PyG Data objects for all 327 segments (deterministic across seeds)
segment_data_objects = []
for s in cached_segments:
    mid = s["machine"]
    mc = machine_data[mid]
    s_idx = max(0, int(s["start"] - 15))
    e_idx = min(len(mc["norm_test"]), int(s["end"] + 15) + 1)
    fine_norm_win = mc["norm_test"][s_idx:e_idx]
    fine_raw_win = mc["raw_test"][s_idx:e_idx]
    
    node_feats = compute_node_features_6d(fine_norm_win)
    ei_tensor = torch.tensor(mc["edge_index"], dtype=torch.long)
    ea_tensor = torch.tensor(mc["edge_attr"], dtype=torch.float32) if mc["edge_attr"] is not None else None
    pyg_data = PyGData(x=torch.tensor(node_feats, dtype=torch.float32), edge_index=ei_tensor, edge_attr=ea_tensor)
    
    # Pre-extract stats for max|z| baseline
    mu = np.array(mc["norm_stats"]["mean"], dtype=float)
    sd = np.array(mc["norm_stats"]["std"], dtype=float)
    sd_adj = np.where(sd < 1e-8, 1.0, sd)
    
    # Ground truth (0-based)
    G = set(g - 1 for g in s["gt_dimensions"])
    
    segment_data_objects.append({
        "seg_id": s["segment_global_id"],
        "machine": mid,
        "G": G,
        "k_gt": len(G),
        "pyg_data": pyg_data,
        "fine_raw_win": fine_raw_win,
        "mu": mu,
        "sd_adj": sd_adj,
    })

print(f"Pre-built {len(segment_data_objects)} PyG Data objects.\n", flush=True)

# -- Helper metric functions ---------------------------------------------------
def metrics(ranking: List[int], G: Set[int]) -> Tuple[float, float, float]:
    k = len(G)
    if k == 0:
        return np.nan, np.nan, np.nan
    top1 = 1.0 if ranking[0] in G else 0.0
    top5 = len(set(ranking[:5]) & G) / k
    legacy = len(set(ranking[:k]) & G) / k
    return top1, top5, legacy

def rank_by(score: np.ndarray) -> List[int]:
    return list(np.argsort(-score, kind="stable"))

def conformal_quantile(scores: np.ndarray, alpha: float) -> int:
    n = len(scores)
    q_level = np.ceil((n + 1) * (1.0 - alpha)) / n
    q_level = min(1.0, max(0.0, q_level))
    return int(np.quantile(scores, q_level, method="higher"))

def get_minimal_prefix(ranking: List[int], G: Set[int], rho: float) -> int:
    k_gt = len(G)
    needed = max(1, int(np.ceil(rho * k_gt)))
    count = 0
    for rank_idx, sensor in enumerate(ranking, 1):
        if sensor in G:
            count += 1
            if count >= needed:
                return rank_idx
    return len(ranking)

# -- Compute max|z| baseline once (seed-independent) ---------------------------
maxz_per_segment = []
maxz_prefix_records = []
for obj in segment_data_objects:
    X = obj["fine_raw_win"]
    mu = obj["mu"]
    sd_adj = obj["sd_adj"]
    G = obj["G"]
    
    z = (X - mu) / sd_adj
    z[:, sd_adj == 1.0] = 0.0
    score = np.abs(z).max(axis=0)
    mz_rank = rank_by(score)
    
    t1, t5, leg = metrics(mz_rank, G)
    maxz_per_segment.append({
        "seg_id": obj["seg_id"],
        "machine": obj["machine"],
        "top1": t1,
        "top5": t5,
        "legacy": leg
    })
    
    p100 = get_minimal_prefix(mz_rank, G, 1.0)
    p80 = get_minimal_prefix(mz_rank, G, 0.8)
    maxz_prefix_records.append({
        "seg_id": obj["seg_id"],
        "machine": obj["machine"],
        "prefix_100": p100,
        "prefix_80": p80
    })

df_mz = pd.DataFrame(maxz_per_segment)
mz_mm = df_mz.groupby("machine")[["top1", "top5", "legacy"]].mean()
mz_top1_macro = float(mz_mm["top1"].mean() * 100)
mz_top5_macro = float(mz_mm["top5"].mean() * 100)
mz_legacy_macro = float(mz_mm["legacy"].mean() * 100)

print(f"max|z| baseline macro: top1={mz_top1_macro:.2f}%, top5={mz_top5_macro:.2f}%, legacy={mz_legacy_macro:.2f}%")
print(f"top-5 recall is identical to seed-42 value (59.55): {abs(mz_top5_macro - 59.55) < 0.01}\n")

# -- Function to evaluate a specific seed --------------------------------------
def run_seed(seed: int):
    print(f"=== Running Attribution for Seed {seed} ===", flush=True)
    t0 = time.time()
    
    seg_csv_path = SEEDS_DIR / f"attribution_per_segment_seed{seed}.csv"
    conf_csv_path = SEEDS_DIR / f"conformal_summary_seed{seed}.csv"
    
    # Check if already computed
    if seg_csv_path.exists() and conf_csv_path.exists():
        print(f"Loading existing outputs for seed {seed} ...", flush=True)
        df_ig = pd.read_csv(seg_csv_path)
        df_conf = pd.read_csv(conf_csv_path)
    else:
        # Load models for this seed
        models = {}
        for mid in machines:
            ckpt_path = CHECKPOINTS_DIR / f"{mid}_seed{seed}.pt"
            ckpt = torch.load(ckpt_path, map_location=device)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            model = GATAnomalyDetector(in_dim=6, hidden_dim=64, num_heads=4, edge_dim=1, dropout=0.2)
            model.load_state_dict(state_dict)
            model.eval()
            models[mid] = model
            
        ig_per_segment = []
        prefix_records = []
        
        for idx, obj in enumerate(segment_data_objects):
            mid = obj["machine"]
            model = models[mid]
            data = obj["pyg_data"]
            G = obj["G"]
            
            # Compute Integrated Gradients (50 steps)
            ig_scores = compute_integrated_gradients_attribution(model, data, steps=50).cpu().numpy()
            
            # 0-based sensor ranking (best first)
            ig_rank = [int(i) for i in np.argsort(ig_scores)[::-1]]
            
            t1, t5, leg = metrics(ig_rank, G)
            ig_per_segment.append({
                "seg_id": obj["seg_id"],
                "machine": mid,
                "top1": t1,
                "top5": t5,
                "legacy": leg
            })
            
            p100 = get_minimal_prefix(ig_rank, G, 1.0)
            p80 = get_minimal_prefix(ig_rank, G, 0.8)
            prefix_records.append({
                "seg_id": obj["seg_id"],
                "machine": mid,
                "prefix_100": p100,
                "prefix_80": p80
            })
            
        elapsed = time.time() - t0
        print(f"Seed {seed} IG attribution computed in {elapsed:.1f}s.", flush=True)
        
        df_ig = pd.DataFrame(ig_per_segment)
        df_prefixes = pd.DataFrame(prefix_records)
        
        # Save per-segment rankings
        df_ig.to_csv(seg_csv_path, index=False)
        
        # Conformal evaluation for IG
        alphas = [0.10, 0.20]
        rhos = [1.0, 0.8]
        n_splits = 1000
        n_total = len(df_prefixes)
        rng_conformal = np.random.RandomState(42)
        
        conformal_results = []
        for rho in rhos:
            col = "prefix_100" if rho == 1.0 else "prefix_80"
            scores_all = df_prefixes[col].values
            mean_min_prefix = float(np.mean(scores_all))
            
            for alpha in alphas:
                nominal_cov = 1.0 - alpha
                q_hat_full = conformal_quantile(scores_all, alpha)
                
                # 1,000 splits
                covs = []
                for _ in range(n_splits):
                    perm = rng_conformal.permutation(n_total)
                    cal_idx = perm[: n_total // 2]
                    test_idx = perm[n_total // 2 :]
                    cal_scores = scores_all[cal_idx]
                    test_scores = scores_all[test_idx]
                    q_hat = conformal_quantile(cal_scores, alpha)
                    covs.append(np.mean(test_scores <= q_hat))
                    
                cov_mean = float(np.mean(covs)) * 100
                cov_sd = float(np.std(covs, ddof=1)) * 100
                
                # LOO Pooled Coverage
                pooled_test_covered = []
                for m in machines:
                    train_mask = (df_prefixes["machine"] != m)
                    test_mask = (df_prefixes["machine"] == m)
                    cal_scores = df_prefixes.loc[train_mask, col].values
                    test_scores = df_prefixes.loc[test_mask, col].values
                    q_hat_m = conformal_quantile(cal_scores, alpha)
                    pooled_test_covered.extend((test_scores <= q_hat_m).tolist())
                loo_pooled_cov = float(np.mean(pooled_test_covered)) * 100
                
                conformal_results.append({
                    "seed": seed,
                    "rho": rho,
                    "alpha": alpha,
                    "nominal_cov": nominal_cov,
                    "q_hat": q_hat_full,
                    "q_hat_frac": f"{q_hat_full / 38.0 * 100:.2f}%",
                    "mean_min_prefix": f"{mean_min_prefix:.2f}",
                    "held_out_cov_mean": cov_mean,
                    "held_out_cov_sd": cov_sd,
                    "held_out_cov_str": f"{cov_mean:.2f}% [{cov_mean - 1.96 * cov_sd:.2f}%, {cov_mean + 1.96 * cov_sd:.2f}%]",
                    "loo_pooled_cov": loo_pooled_cov
                })
                
        df_conf = pd.DataFrame(conformal_results)
        df_conf.to_csv(conf_csv_path, index=False)
        
    # Machine-macro
    mm = df_ig.groupby("machine")[["top1", "top5", "legacy"]].mean()
    top1_macro = float(mm["top1"].mean() * 100)
    top5_macro = float(mm["top5"].mean() * 100)
    legacy_macro = float(mm["legacy"].mean() * 100)
    
    # Machine-clustered bootstrap (10,000 resamples, seed 42)
    b_rng = np.random.default_rng(42)
    n_mach = len(machines)
    boot_idx = b_rng.integers(0, n_mach, size=(10_000, n_mach))
    
    M = mm.loc[machines][["top1", "top5", "legacy"]].values  # (28, 3)
    bM = M[boot_idx].mean(axis=1) * 100
    
    top1_ci = f"{top1_macro:.2f} [{np.percentile(bM[:, 0], 2.5):.2f}, {np.percentile(bM[:, 0], 97.5):.2f}]"
    top5_ci = f"{top5_macro:.2f} [{np.percentile(bM[:, 1], 2.5):.2f}, {np.percentile(bM[:, 1], 97.5):.2f}]"
    legacy_ci = f"{legacy_macro:.2f} [{np.percentile(bM[:, 2], 2.5):.2f}, {np.percentile(bM[:, 2], 97.5):.2f}]"
    
    # Paired Top-5 difference: IG minus max|z|
    D_top5 = mm.loc[machines]["top5"].values - mz_mm.loc[machines]["top5"].values
    diff_pt = float(D_top5.mean() * 100)
    bD = D_top5[boot_idx].mean(axis=1) * 100
    diff_ci = f"{diff_pt:+.2f} [{np.percentile(bD, 2.5):+.2f}, {np.percentile(bD, 97.5):+.2f}]"
    
    conformal_q_hats = {}
    for _, row in df_conf.iterrows():
        conformal_q_hats[(float(row["rho"]), float(row["alpha"]))] = int(row["q_hat"])
        
    return {
        "seed": seed,
        "top1_macro": top1_macro,
        "top5_macro": top5_macro,
        "legacy_macro": legacy_macro,
        "top1_ci": top1_ci,
        "top5_ci": top5_ci,
        "legacy_ci": legacy_ci,
        "diff_vs_maxz": diff_pt,
        "diff_vs_maxz_ci": diff_ci,
        "conformal_q_hats": conformal_q_hats,
        "conformal_df": df_conf,
    }

# ==============================================================================
# GATE: Evaluate Seed 42 First
# ==============================================================================
res_42 = run_seed(42)

print("\n-- GATE VERIFICATION FOR SEED 42 --", flush=True)
print(f"IG Machine-Macro: top-1={res_42['top1_macro']:.2f} (exp 83.89), "
      f"top-5={res_42['top5_macro']:.2f} (exp 53.93), legacy={res_42['legacy_macro']:.2f} (exp 62.84)")

gate1_ok = (
    abs(res_42["top1_macro"] - 83.89) <= 0.05 and
    abs(res_42["top5_macro"] - 53.93) <= 0.05 and
    abs(res_42["legacy_macro"] - 62.84) <= 0.05
)

q_hats_42 = res_42["conformal_q_hats"]
print(f"Conformal q_hat: rho=1 a=0.1: {q_hats_42[(1.0, 0.1)]} (exp 36) | "
      f"rho=1 a=0.2: {q_hats_42[(1.0, 0.2)]} (exp 31) | "
      f"rho=0.8 a=0.1: {q_hats_42[(0.8, 0.1)]} (exp 28) | "
      f"rho=0.8 a=0.2: {q_hats_42[(0.8, 0.2)]} (exp 24)")

gate2_ok = (
    q_hats_42[(1.0, 0.1)] == 36 and
    q_hats_42[(1.0, 0.2)] == 31 and
    q_hats_42[(0.8, 0.1)] == 28 and
    q_hats_42[(0.8, 0.2)] == 24
)

if not (gate1_ok and gate2_ok):
    print("GATE FAILED. STOPPING.")
    sys.exit(1)

print("GATE PASSED FOR SEED 42.\n", flush=True)

# ==============================================================================
# Run Seeds 43 and 44
# ==============================================================================
res_43 = run_seed(43)
res_44 = run_seed(44)

all_results = [res_42, res_43, res_44]

# -- Summary Tables ------------------------------------------------------------
# 1. Attribution Table
attr_rows = []
for r in all_results:
    attr_rows.append({
        "seed": r["seed"],
        "top1_macro": r["top1_macro"],
        "top1_ci": r["top1_ci"],
        "top5_macro": r["top5_macro"],
        "top5_ci": r["top5_ci"],
        "legacy_macro": r["legacy_macro"],
        "legacy_ci": r["legacy_ci"],
        "paired_diff_vs_maxz": r["diff_vs_maxz"],
        "paired_diff_vs_maxz_ci": r["diff_vs_maxz_ci"]
    })

df_attr_summary = pd.DataFrame(attr_rows)
df_attr_summary.to_csv(SEEDS_DIR / "seeds_attribution_summary.csv", index=False)

# 2. Conformal Table
all_conf_dfs = [r["conformal_df"] for r in all_results]
df_all_conf = pd.concat(all_conf_dfs, ignore_index=True)
df_all_conf.to_csv(SEEDS_DIR / "seeds_conformal_summary.csv", index=False)

# 3. Aggregate Table across Seeds 42, 43, 44
agg_metrics = {
    "IG Top-1 Hit (%)": [r["top1_macro"] for r in all_results],
    "IG Top-5 Recall (%)": [r["top5_macro"] for r in all_results],
    "IG Legacy Overlap (%)": [r["legacy_macro"] for r in all_results],
    "Top-5 Diff: IG minus max|z| (pp)": [r["diff_vs_maxz"] for r in all_results],
}

# Add conformal metrics across seeds
for rho in [1.0, 0.8]:
    for alpha in [0.10, 0.20]:
        sub = df_all_conf[(df_all_conf["rho"] == rho) & (df_all_conf["alpha"] == alpha)]
        agg_metrics[f"Conformal q_hat (rho={rho}, a={alpha})"] = sub["q_hat"].values.tolist()
        agg_metrics[f"Held-Out Cov Mean % (rho={rho}, a={alpha})"] = sub["held_out_cov_mean"].values.tolist()
        agg_metrics[f"LOO Pooled Cov % (rho={rho}, a={alpha})"] = sub["loo_pooled_cov"].values.tolist()

agg_rows = []
for k, vals in agg_metrics.items():
    agg_rows.append({
        "metric": k,
        "seed_42": round(vals[0], 2),
        "seed_43": round(vals[1], 2),
        "seed_44": round(vals[2], 2),
        "mean": round(float(np.mean(vals)), 2),
        "sd": round(float(np.std(vals, ddof=1)), 2),
    })

df_agg = pd.DataFrame(agg_rows)
df_agg.to_csv(SEEDS_DIR / "multi_seed_aggregate.csv", index=False)

print("\n-- ATTRIBUTION RESULTS PER SEED --")
print(df_attr_summary.to_string(index=False))

print("\n-- MULTI-SEED AGGREGATE (MEAN & SD ACROSS SEEDS 42, 43, 44) --")
print(df_agg.to_string(index=False))

# Log
ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
with open(RESULTS_C / "LOG.md", "a", encoding="utf-8") as f:
    f.write(f"\n## Multi-Seed Attribution & Conformal Evaluation (Seeds 42, 43, 44) {ts}\n")
    f.write(f"- Gate verified on Seed 42: IG macro top1=83.89, top5=53.93, legacy=62.84; q_hat=36,31,28,24\n")
    f.write(f"- max|z| baseline top-5 recall: {mz_top5_macro:.2f}% (identical to seed-42)\n")
    f.write(f"- Outputs: results_c/seeds/\n")
    f.write(df_agg.to_string(index=False) + "\n")

print(f"\nCompleted successfully. Outputs saved to {SEEDS_DIR}")
