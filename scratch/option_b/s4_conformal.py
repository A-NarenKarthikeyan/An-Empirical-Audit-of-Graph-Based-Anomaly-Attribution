"""Stage S4: Conformal Calibration on Attribution Rankings.

Nonconformity score = smallest prefix length of the IG ranking that reaches
containment rho for rho in {1.0, 0.8}; alpha in {0.10, 0.20};
Quantile rule = ceil((n+1)(1-alpha))/n.
(a) Pooled held-out coverage: 1,000 random 50/50 splits of the 327 segments (seed 42).
(b) Leave-one-machine-out: calibrate on 27 machines, test on held-out machine.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple, Set
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath("."))

def compute_minimal_prefix_lengths(
    segments_data: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Compute minimal prefix lengths for rho in {1.0, 0.8} for each segment."""
    records = []
    for s in segments_data:
        gt = set(s["gt_dimensions"])
        k_gt = len(gt)
        ranking = s["ig_ranking_1indexed"]  # 1-indexed sensor IDs ordered descending
        
        if k_gt == 0:
            continue
            
        # rho = 1.0 (100% containment)
        prefix_100 = max(ranking.index(d) + 1 for d in gt)
        
        # rho = 0.8 (80% containment)
        needed_80 = max(1, int(np.ceil(0.80 * k_gt)))
        count = 0
        prefix_80 = 38
        for rank_idx, sensor_id in enumerate(ranking, 1):
            if sensor_id in gt:
                count += 1
                if count >= needed_80:
                    prefix_80 = rank_idx
                    break
                    
        records.append({
            "segment_global_id": s["segment_global_id"],
            "machine": s["machine"],
            "segment_index": s["segment_index"],
            "k_gt": k_gt,
            "prefix_100": prefix_100,
            "prefix_80": prefix_80
        })
    return records

def conformal_quantile(scores: np.ndarray, alpha: float) -> int:
    """Compute conformal quantile at level ceil((n+1)(1-alpha))/n using 'higher' method."""
    n = len(scores)
    q_level = np.ceil((n + 1) * (1.0 - alpha)) / n
    q_level = min(1.0, max(0.0, q_level))
    q_hat = int(np.quantile(scores, q_level, method="higher"))
    return q_hat

def run_s4_conformal():
    print("=" * 70)
    print("STAGE S4 — Conformal Calibration (Pooled Splits & LOO-Machine)")
    print("=" * 70)
    
    t0 = time.time()
    
    # Load rankings from Stage S3
    rankings_path = "results_b/s3_cached_rankings.json"
    if not os.path.exists(rankings_path):
        raise FileNotFoundError(f"Missing S3 rankings: {rankings_path}. Ensure Stage S3 completed!")
        
    with open(rankings_path, "r", encoding="utf-8") as f:
        segments_data = json.load(f)
        
    assert len(segments_data) == 327, f"Expected 327 segments, got {len(segments_data)}"
    
    prefix_records = compute_minimal_prefix_lengths(segments_data)
    df_prefixes = pd.DataFrame(prefix_records)
    
    alphas = [0.10, 0.20]
    rhos = [1.0, 0.8]
    n_splits = 1000
    rng = np.random.RandomState(42)
    n_total = len(df_prefixes)
    
    summary_rows = []
    
    # -----------------------------------------------------------------
    # (a) Pooled Held-Out Coverage: 1,000 random 50/50 splits
    # -----------------------------------------------------------------
    print(f"\nRunning 1,000 random 50/50 splits (seed 42) across {n_total} segments...")
    
    for rho in rhos:
        score_col = "prefix_100" if rho == 1.0 else "prefix_80"
        scores_all = df_prefixes[score_col].values
        mean_min_prefix = float(np.mean(scores_all))
        
        for alpha in alphas:
            nominal_cov = 1.0 - alpha
            
            # Full pool q-hat
            q_hat_full = conformal_quantile(scores_all, alpha)
            
            # 1,000 splits
            held_out_coverages = []
            q_hats = []
            
            for _ in range(n_splits):
                perm = rng.permutation(n_total)
                cal_idx = perm[: n_total // 2]
                test_idx = perm[n_total // 2 :]
                
                cal_scores = scores_all[cal_idx]
                test_scores = scores_all[test_idx]
                
                q_hat = conformal_quantile(cal_scores, alpha)
                q_hats.append(q_hat)
                
                cov = np.mean(test_scores <= q_hat)
                held_out_coverages.append(cov)
                
            cov_mean = float(np.mean(held_out_coverages))
            cov_sd = float(np.std(held_out_coverages, ddof=1))
            
            summary_rows.append({
                "containment_rho": rho,
                "nominal_coverage": nominal_cov,
                "alpha": alpha,
                "full_pool_q_hat": q_hat_full,
                "q_hat_fraction_of_38": round(q_hat_full / 38.0, 4),
                "mean_minimal_prefix": round(mean_min_prefix, 2),
                "held_out_coverage_mean": round(cov_mean, 4),
                "held_out_coverage_sd": round(cov_sd, 4),
                "held_out_coverage_95ci": f"[{cov_mean - 1.96 * cov_sd:.4f}, {cov_mean + 1.96 * cov_sd:.4f}]"
            })
            
    df_summary = pd.DataFrame(summary_rows)
    summary_path = "results_b/conformal_summary.csv"
    df_summary.to_csv(summary_path, index=False)
    print(f"Saved conformal summary to {summary_path}")
    
    # -----------------------------------------------------------------
    # (b) Leave-One-Machine-Out (LOO) Calibration
    # -----------------------------------------------------------------
    print("\nRunning Leave-One-Machine-Out (LOO) Calibration across 28 machines...")
    machines = sorted(df_prefixes["machine"].unique().tolist())
    loo_rows = []
    
    for rho in rhos:
        score_col = "prefix_100" if rho == 1.0 else "prefix_80"
        
        for alpha in alphas:
            nominal_cov = 1.0 - alpha
            pooled_test_covered = []
            
            for m in machines:
                train_mask = (df_prefixes["machine"] != m)
                test_mask = (df_prefixes["machine"] == m)
                
                cal_scores = df_prefixes.loc[train_mask, score_col].values
                test_scores = df_prefixes.loc[test_mask, score_col].values
                
                q_hat_m = conformal_quantile(cal_scores, alpha)
                n_test_m = len(test_scores)
                
                covered_m = np.sum(test_scores <= q_hat_m)
                cov_m = float(covered_m / n_test_m) if n_test_m > 0 else float("nan")
                
                pooled_test_covered.extend((test_scores <= q_hat_m).tolist())
                
                loo_rows.append({
                    "held_out_machine": m,
                    "containment_rho": rho,
                    "nominal_coverage": nominal_cov,
                    "alpha": alpha,
                    "n_cal_segments": len(cal_scores),
                    "n_test_segments": n_test_m,
                    "q_hat": q_hat_m,
                    "machine_empirical_coverage": cov_m
                })
                
            pooled_cov = float(np.mean(pooled_test_covered))
            print(f"LOO Pooled Coverage (rho={rho}, alpha={alpha}, nominal={nominal_cov:.2f}): {pooled_cov*100:.2f}%")
            
    df_loo = pd.DataFrame(loo_rows)
    loo_path = "results_b/conformal_loo_machine.csv"
    df_loo.to_csv(loo_path, index=False)
    print(f"Saved LOO conformal per-machine metrics to {loo_path}")
    
    # Print formatted summary table
    print("\n--- CONFORMAL CALIBRATION SUMMARY ---")
    for _, r in df_summary.iterrows():
        print(f"Containment rho={r['containment_rho']} | Nominal={r['nominal_coverage']*100:.0f}% (alpha={r['alpha']}):")
        print(f"  Full Pool q-hat:           {r['full_pool_q_hat']} sensors ({r['q_hat_fraction_of_38']*100:.1f}% of 38)")
        print(f"  Mean Minimal Prefix:       {r['mean_minimal_prefix']} sensors")
        print(f"  Held-Out Coverage (1k splits): {r['held_out_coverage_mean']*100:.2f}% +/- {r['held_out_coverage_sd']*100:.2f}% (95% range: {r['held_out_coverage_95ci']})")
        
    elapsed = time.time() - t0
    print(f"\nStage S4 completed in {elapsed:.2f}s")
    
    with open("results_b/LOG.md", "a", encoding="utf-8") as f:
        f.write(f"\n## Stage S4 — Conformal Calibration Completed\n")
        f.write(f"- Wall-clock: {elapsed:.2f}s\n")
        f.write(f"- Outputs: `results_b/conformal_summary.csv`, `results_b/conformal_loo_machine.csv`\n")

if __name__ == "__main__":
    run_s4_conformal()
