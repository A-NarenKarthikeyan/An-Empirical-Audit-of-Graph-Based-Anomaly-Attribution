"""
compute_conformal_baselines.py — Conformal Calibration for max|z| baseline vs IG.

Reproduces gate on IG (s4_conformal.py):
  q_hat = 36 (rho=1, alpha=0.1), 31 (rho=1, alpha=0.2), 28 (rho=0.8, alpha=0.1), 24 (rho=0.8, alpha=0.2)
  held-out coverage = 92.47, 81.71, 91.67, 81.96 (1,000 50/50 splits, seed 42)

Then computes:
  - max|z| conformal sets (q_hat, fraction of 38, held-out coverage, LOO pooled coverage)
  - Paired top-1 hit difference (IG minus max|z|) with machine-clustered bootstrap CI (10,000, seed 42)
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import datetime, hashlib

BASE_DIR = Path(__file__).resolve().parents[1]
RESULTS_B = BASE_DIR / "results_b"
RESULTS_C = BASE_DIR / "results_c"
RESULTS_C.mkdir(exist_ok=True)

RANKINGS_PATH = RESULTS_B / "s3_cached_rankings.json"
ATTR_PATH = RESULTS_B / "attribution_per_segment.csv"

with open(RANKINGS_PATH, "r", encoding="utf-8") as f:
    _raw = json.load(f)
rankings_by_seg = {e["segment_global_id"]: e for e in _raw}

EPS = 1e-8
def load_segment(seg_id):
    entry = rankings_by_seg[seg_id]
    mu = np.array(entry["norm_stats_mean"], dtype=float)
    sd = np.array(entry["norm_stats_std"], dtype=float)
    sd_adj = np.where(sd < EPS, 1.0, sd)
    X = np.array(entry["fine_raw_window"], dtype=float)
    G = {g - 1 for g in entry["gt_dimensions"]}
    return X, mu, sd_adj, G, entry["machine"]

def z_scores(X, mu, sd_adj):
    z = (X - mu) / sd_adj
    z[:, sd_adj == 1.0] = 0.0
    return z

def rank_by(score):
    return list(np.argsort(-score, kind="stable"))

def conformal_quantile(scores: np.ndarray, alpha: float) -> int:
    n = len(scores)
    q_level = np.ceil((n + 1) * (1.0 - alpha)) / n
    q_level = min(1.0, max(0.0, q_level))
    return int(np.quantile(scores, q_level, method="higher"))

seg_ids = list(rankings_by_seg.keys())

records_ig = []
records_maxz = []

for seg_id in seg_ids:
    X, mu, sd_adj, G, machine = load_segment(seg_id)
    entry = rankings_by_seg[seg_id]
    k_gt = len(G)
    
    ig_rank = [i - 1 for i in entry["ig_ranking_1indexed"]]
    z = z_scores(X, mu, sd_adj)
    maxz_score = np.abs(z).max(axis=0)
    maxz_rank = rank_by(maxz_score)
    
    def get_prefixes(ranking):
        prefix_100 = max(ranking.index(g) + 1 for g in G)
        needed_80 = max(1, int(np.ceil(0.80 * k_gt)))
        count = 0
        prefix_80 = 38
        for rank_idx, sensor_idx in enumerate(ranking, 1):
            if sensor_idx in G:
                count += 1
                if count >= needed_80:
                    prefix_80 = rank_idx
                    break
        return prefix_100, prefix_80

    ig_100, ig_80 = get_prefixes(ig_rank)
    mz_100, mz_80 = get_prefixes(maxz_rank)
    
    records_ig.append({"seg_id": seg_id, "machine": machine, "prefix_100": ig_100, "prefix_80": ig_80})
    records_maxz.append({"seg_id": seg_id, "machine": machine, "prefix_100": mz_100, "prefix_80": mz_80})

df_ig = pd.DataFrame(records_ig)
df_mz = pd.DataFrame(records_maxz)

alphas = [0.10, 0.20]
rhos = [1.0, 0.8]
n_splits = 1000
n_total = len(df_ig)

# ══════════════════════════════════════════════════════════════════════════════
# GATE: IG Reproduction Check
# ══════════════════════════════════════════════════════════════════════════════
print("── CONFORMAL GATE: IG Reproduction Check ──", flush=True)
expected_ig = {
    (1.0, 0.10): (36, 92.47),
    (1.0, 0.20): (31, 81.71),
    (0.8, 0.10): (28, 91.67),
    (0.8, 0.20): (24, 81.96),
}

rng_gate = np.random.RandomState(42)
gate_passed = True

for rho in rhos:
    col = "prefix_100" if rho == 1.0 else "prefix_80"
    scores_all = df_ig[col].values
    for alpha in alphas:
        q_hat_full = conformal_quantile(scores_all, alpha)
        covs = []
        for _ in range(n_splits):
            perm = rng_gate.permutation(n_total)
            cal_idx = perm[: n_total // 2]
            test_idx = perm[n_total // 2 :]
            cal_scores = scores_all[cal_idx]
            test_scores = scores_all[test_idx]
            q_hat = conformal_quantile(cal_scores, alpha)
            covs.append(np.mean(test_scores <= q_hat))
        cov_mean = round(float(np.mean(covs)) * 100, 2)
        exp_q, exp_cov = expected_ig[(rho, alpha)]
        diff_cov = abs(cov_mean - exp_cov)
        print(f"  rho={rho}, alpha={alpha}: q_hat={q_hat_full} (exp {exp_q}), cov={cov_mean:.2f}% (exp {exp_cov:.2f}%)")
        if q_hat_full != exp_q or diff_cov > 0.05:
            gate_passed = False

if not gate_passed:
    print("CONFORMAL GATE FAILED. STOPPING.")
    raise SystemExit(1)
print("CONFORMAL GATE PASSED.\n")

# ══════════════════════════════════════════════════════════════════════════════
# MAX|Z| Conformal Analysis
# ══════════════════════════════════════════════════════════════════════════════
machines = sorted(df_mz["machine"].unique().tolist())
rng_mz = np.random.RandomState(42)

mz_summary_rows = []
for rho in rhos:
    col = "prefix_100" if rho == 1.0 else "prefix_80"
    scores_all = df_mz[col].values
    mean_min_prefix = float(np.mean(scores_all))
    
    for alpha in alphas:
        nominal_cov = 1.0 - alpha
        q_hat_full = conformal_quantile(scores_all, alpha)
        
        # 1000 splits
        covs = []
        for _ in range(n_splits):
            perm = rng_mz.permutation(n_total)
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
            train_mask = (df_mz["machine"] != m)
            test_mask = (df_mz["machine"] == m)
            cal_scores = df_mz.loc[train_mask, col].values
            test_scores = df_mz.loc[test_mask, col].values
            q_hat_m = conformal_quantile(cal_scores, alpha)
            pooled_test_covered.extend((test_scores <= q_hat_m).tolist())
        loo_pooled_cov = float(np.mean(pooled_test_covered)) * 100
        
        mz_summary_rows.append({
            "ranking": "max|z|",
            "rho": rho,
            "alpha": alpha,
            "nominal_cov": nominal_cov,
            "q_hat": q_hat_full,
            "q_hat_frac": f"{q_hat_full / 38.0 * 100:.2f}%",
            "mean_min_prefix": f"{mean_min_prefix:.2f}",
            "held_out_cov": f"{cov_mean:.2f}% [{cov_mean - 1.96 * cov_sd:.2f}%, {cov_mean + 1.96 * cov_sd:.2f}%]",
            "loo_pooled_cov": f"{loo_pooled_cov:.2f}%"
        })

df_conformal_out = pd.DataFrame(mz_summary_rows)
df_conformal_out.to_csv(RESULTS_C / "conformal_max_abs_z_summary.csv", index=False)

# ══════════════════════════════════════════════════════════════════════════════
# Paired Top-1 Hit Difference: IG minus max|z|
# ══════════════════════════════════════════════════════════════════════════════
df_b = pd.read_csv(RESULTS_C / "baseline_per_segment.csv")
mm = df_b.groupby(["method", "machine"])[["top1", "top5", "legacy"]].mean()

n_mach = len(machines)
b_rng = np.random.default_rng(42)
boot_idx = b_rng.integers(0, n_mach, size=(10_000, n_mach))

m_ig = mm.loc["IG"].loc[machines]["top1"].values
m_maxz = mm.loc["max_abs_z"].loc[machines]["top1"].values
D_top1 = m_ig - m_maxz

diff_mean = D_top1.mean() * 100
bD = D_top1[boot_idx].mean(axis=1) * 100
diff_lo = float(np.percentile(bD, 2.5))
diff_hi = float(np.percentile(bD, 97.5))

print("── CONFORMAL RESULTS (max|z|) ──")
print(df_conformal_out.to_string(index=False))

print(f"\n── PAIRED TOP-1 HIT DIFFERENCE (IG minus max|z|) ──")
print(f"Point estimate: {diff_mean:+.2f}%")
print(f"95% Machine-Clustered Bootstrap CI (10,000, seed 42): [{diff_lo:+.2f}%, {diff_hi:+.2f}%]")

# Log
ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
with open(RESULTS_C / "LOG.md", "a", encoding="utf-8") as f:
    f.write(f"\n## Conformal max|z| run {ts}\n")
    f.write("- Gate passed: IG q_hat (36, 31, 28, 24) and held-out coverage (92.47, 81.71, 91.67, 81.96)\n")
    f.write(f"- Paired Top-1 Hit Difference (IG - max|z|): {diff_mean:+.2f}% [{diff_lo:+.2f}%, {diff_hi:+.2f}%]\n")
    f.write(df_conformal_out.to_string(index=False) + "\n")
