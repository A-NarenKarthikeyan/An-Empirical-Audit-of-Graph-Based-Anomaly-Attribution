"""
compute_baselines.py — Trivial attribution baselines vs IG on 327 segments.

Loaders (Step 1 / 2 / 3 verified by debug_loaders.py):
  - G        = {g-1 for g in entry["gt_dimensions"]}   (0-based)
  - X        = entry["fine_raw_window"]                  (raw, shape T×38)
  - mu/sd    = entry["norm_stats_mean/std"]              (training stats from normalize_data)
  - std_adj  = where(sd < eps, 1.0, sd)                  (mirrors normalize_data exactly)
  - z        = (X - mu) / std_adj  then constant-sensor columns set to 0.0

Gate order (STOP on any failure):
  1. Per-segment exact-match ≥ 326/327 vs attribution_per_segment.csv
  2. IG machine-macro top1/top5/legacy within 0.05 of 83.89/53.93/62.85
  3. Random machine-macro top5 in [8,20]%, top1 in [12,30]%

Then: baselines (a)-(d), machine-macro, bootstrap CIs, paired diffs.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import hashlib, datetime

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parents[1]
RESULTS_B   = BASE_DIR / "results_b"
RESULTS_C   = BASE_DIR / "results_c"
RESULTS_C.mkdir(exist_ok=True)

RANKINGS_PATH = RESULTS_B / "s3_cached_rankings.json"
ATTR_PATH     = RESULTS_B / "attribution_per_segment.csv"

N        = 38          # sensors
EPS      = 1e-8        # mirrors normalize_data eps

# ── Load JSON cache ───────────────────────────────────────────────────────────
print("Loading s3_cached_rankings.json …", flush=True)
with open(RANKINGS_PATH) as f:
    _raw = json.load(f)
rankings_by_seg = {e["segment_global_id"]: e for e in _raw}

# Config hash (for LOG.md)
_hash = hashlib.md5(json.dumps(
    {"n_segs": len(rankings_by_seg), "n_boot": 10000, "seed": 42, "n_rand": 100},
    sort_keys=True).encode()).hexdigest()[:8]

# ── Loader: load_segment ──────────────────────────────────────────────────────
def load_segment(seg_id):
    """Return (X, mu, sd_adj, G_0based, machine).

    X       : raw fine window, shape (T, 38)
    mu      : per-sensor training mean (raw space), shape (38,)
    sd_adj  : per-sensor training std  (raw space) with constant-sensor floor=1.0,
              exactly as normalize_data uses (eps=1e-8, zero→1.0)
    G_0based: set of 0-based ground-truth sensor indices
    machine : machine name string
    """
    entry = rankings_by_seg[seg_id]
    mu    = np.array(entry["norm_stats_mean"], dtype=float)
    sd    = np.array(entry["norm_stats_std"],  dtype=float)
    sd_adj = np.where(sd < EPS, 1.0, sd)          # mirrors normalize_data
    X     = np.array(entry["fine_raw_window"],  dtype=float)  # shape (T,38)
    G     = {g - 1 for g in entry["gt_dimensions"]}           # 0-based
    return X, mu, sd_adj, G, entry["machine"]

# ── Loader: load_ig_ranking ───────────────────────────────────────────────────
def load_ig_ranking(seg_id):
    """Return seed-42 IG ranking as list of 0-based sensor indices (best first)."""
    entry = rankings_by_seg[seg_id]
    return [i - 1 for i in entry["ig_ranking_1indexed"]]      # 1→0-based

# ── Metric function ───────────────────────────────────────────────────────────
def metrics(ranking, G):
    """ranking: list of 0-based indices (best first). G: set of 0-based indices."""
    k = len(G)
    if k == 0:
        return np.nan, np.nan, np.nan
    top1   = 1.0 if ranking[0] in G else 0.0
    top5   = len(set(ranking[:5]) & G) / k
    legacy = len(set(ranking[:k]) & G) / k
    return top1, top5, legacy

# ── Ranking functions ─────────────────────────────────────────────────────────
def z_scores(X, mu, sd_adj):
    """
    Compute per-sensor z-scores for the window.
    Constant-sensor columns (sd_adj==1.0 because original sd<eps) → z=0.
    """
    z = (X - mu) / sd_adj                 # shape (T, 38)
    zero_mask = sd_adj == 1.0             # constant sensors get sd_adj=1.0
    z[:, zero_mask] = 0.0
    return z                               # shape (T, 38)

def rank_by(score):
    """Best-first 0-based ranking; ties broken by lower sensor index (stable)."""
    return list(np.argsort(-score, kind="stable"))

def scores(X, mu, sd_adj):
    """Return dict of per-sensor scalar scores for the three stat baselines."""
    z = z_scores(X, mu, sd_adj)
    return {
        "max_abs_z":  np.abs(z).max(axis=0),                             # (38,)
        "mean_abs_z": np.abs(z).mean(axis=0),                            # (38,)
        "mean_shift": np.abs(X.mean(axis=0) - mu) / sd_adj,             # (38,)
    }

# ── All 327 segment IDs (in JSON order = original pipeline order) ─────────────
seg_ids = list(rankings_by_seg.keys())   # already 327 integers in order

# ══════════════════════════════════════════════════════════════════════════════
# GATE 1 — Per-segment exact-match reproduction
# ══════════════════════════════════════════════════════════════════════════════
print("\n── GATE 1: per-segment reproduction ──", flush=True)
attr_df = pd.read_csv(ATTR_PATH)
ig_ref  = (attr_df[attr_df["method"].str.strip() == "Integrated_Gradients"]
           .set_index("segment_global_id"))

mismatches = []
for seg_id in seg_ids:
    if seg_id not in ig_ref.index:
        continue
    X, mu, sd_adj, G, machine = load_segment(seg_id)
    rank = load_ig_ranking(seg_id)
    t1, t5, leg = metrics(rank, G)
    row = ig_ref.loc[seg_id]
    ok = (abs(t1  - row["top1_hit"])       < 1e-9 and
          abs(t5  - row["top5_recall"])    < 1e-9 and
          abs(leg - row["legacy_overlap"]) < 1e-9)
    if not ok:
        mismatches.append((seg_id, machine, G,
                           (t1, t5, leg),
                           (row["top1_hit"], row["top5_recall"], row["legacy_overlap"])))

n_match = len(seg_ids) - len(mismatches)
print(f"  Exact matches: {n_match} / {len(seg_ids)}")
if mismatches:
    print(f"  {len(mismatches)} mismatch(es) — first 3:")
    for sid, mach, G, comp, ref in mismatches[:3]:
        print(f"    seg={sid} {mach}  G={sorted(G)}")
        print(f"      computed={tuple(round(v,6) for v in comp)}")
        print(f"      reference={tuple(round(float(v),6) for v in ref)}")

if n_match < 326:
    print(f"GATE 1 FAILED: only {n_match}/327 exact matches (need ≥326). STOPPING.")
    raise SystemExit(1)
print("  GATE 1 passed.")

# ══════════════════════════════════════════════════════════════════════════════
# Compute per-segment results for all 5 methods
# ══════════════════════════════════════════════════════════════════════════════
print("\nComputing per-segment metrics for all methods …", flush=True)

rng_rand = np.random.default_rng(42)

rows = []
for seg_id in seg_ids:
    X, mu, sd_adj, G, machine = load_segment(seg_id)
    ig_rank = load_ig_ranking(seg_id)

    # IG
    t1, t5, leg = metrics(ig_rank, G)
    rows.append((seg_id, machine, "IG", t1, t5, leg))

    # Three stat baselines
    sc = scores(X, mu, sd_adj)
    for name, score in sc.items():
        t1b, t5b, legb = metrics(rank_by(score), G)
        rows.append((seg_id, machine, name, t1b, t5b, legb))

    # Random: average of 100 draws
    rand_vals = np.array([
        metrics(list(rng_rand.permutation(N)), G) for _ in range(100)
    ]).mean(axis=0)
    rows.append((seg_id, machine, "random", *rand_vals))

df = pd.DataFrame(rows, columns=["seg", "machine", "method", "top1", "top5", "legacy"])
df.to_csv(RESULTS_C / "baseline_per_segment.csv", index=False)
print(f"  Saved baseline_per_segment.csv ({len(df)} rows)")

# ══════════════════════════════════════════════════════════════════════════════
# Machine-macro
# ══════════════════════════════════════════════════════════════════════════════
mm = df.groupby(["method", "machine"])[["top1", "top5", "legacy"]].mean()

# ── GATE 2 — IG machine-macro ──────────────────────────────────────────────
print("\n── GATE 2: IG machine-macro ──", flush=True)
ig_macro = mm.loc["IG"].mean() * 100
print(f"  IG machine-macro: top1={ig_macro['top1']:.2f}  top5={ig_macro['top5']:.2f}  "
      f"legacy={ig_macro['legacy']:.2f}")
target = {"top1": 83.89, "top5": 53.93, "legacy": 62.85}
diffs  = {k: abs(ig_macro[k] - v) for k, v in target.items()}
if any(v > 0.05 for v in diffs.values()):
    print(f"  Differences: {diffs}")
    print("GATE 2 FAILED. STOPPING.")
    raise SystemExit(1)
print("  GATE 2 passed.")

# ── GATE 3 — Random sanity ─────────────────────────────────────────────────
print("\n── GATE 3: random machine-macro ──", flush=True)
rnd_macro = mm.loc["random"].mean() * 100
print(f"  random machine-macro: top1={rnd_macro['top1']:.2f}  top5={rnd_macro['top5']:.2f}  "
      f"legacy={rnd_macro['legacy']:.2f}")
if not (8 <= rnd_macro["top5"] <= 20):
    print(f"GATE 3 FAILED: random top5={rnd_macro['top5']:.2f}% not in [8,20]%. STOPPING.")
    raise SystemExit(1)
if not (12 <= rnd_macro["top1"] <= 30):
    print(f"GATE 3 FAILED: random top1={rnd_macro['top1']:.2f}% not in [12,30]%. STOPPING.")
    raise SystemExit(1)
print("  GATE 3 passed.")

# ══════════════════════════════════════════════════════════════════════════════
# Machine-clustered bootstrap (10,000 resamples, seed 42)
# ══════════════════════════════════════════════════════════════════════════════
print("\nRunning bootstrap (10,000 resamples) …", flush=True)

machines  = sorted(df["machine"].unique())
n_mach    = len(machines)
b_rng     = np.random.default_rng(42)
boot_idx  = b_rng.integers(0, n_mach, size=(10_000, n_mach))   # (10000, 28)

methods_order = ["IG", "max_abs_z", "mean_abs_z", "mean_shift", "random"]

res_rows = []
for method in methods_order:
    M = mm.loc[method].loc[machines][["top1", "top5", "legacy"]].values   # (28,3)
    D = M - mm.loc["IG"].loc[machines][["top1", "top5", "legacy"]].values  # paired diff

    bM = M[boot_idx].mean(axis=1) * 100    # (10000,3) bootstrap macro means
    bD = D[boot_idx].mean(axis=1) * 100    # (10000,3) bootstrap paired diffs

    def ci_str(col_idx):
        pt  = M[:, col_idx].mean() * 100
        lo  = float(np.percentile(bM[:, col_idx], 2.5))
        hi  = float(np.percentile(bM[:, col_idx], 97.5))
        return f"{pt:.2f} [{lo:.2f}, {hi:.2f}]"

    diff_pt = D[:, 1].mean() * 100
    diff_lo = float(np.percentile(bD[:, 1], 2.5))
    diff_hi = float(np.percentile(bD[:, 1], 97.5))
    diff_str = f"{diff_pt:+.2f} [{diff_lo:+.2f}, {diff_hi:+.2f}]"

    res_rows.append([method, ci_str(0), ci_str(1), ci_str(2), diff_str])

out_df = pd.DataFrame(
    res_rows,
    columns=["method", "top1 [95%CI]", "top5 [95%CI]", "legacy [95%CI]",
             "top5 diff vs IG [95%CI]"]
)
out_df.to_csv(RESULTS_C / "baseline_metrics.csv", index=False)

print("\n── RESULTS TABLE ──")
print(out_df.to_string(index=False))
print(f"\nNumber of segments: {len(seg_ids)}")
print(f"Number of machines: {n_mach}")

# ══════════════════════════════════════════════════════════════════════════════
# LOG.md
# ══════════════════════════════════════════════════════════════════════════════
ts = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")
log_path = RESULTS_C / "LOG.md"
with open(log_path, "a", encoding="utf-8") as f:
    f.write(f"\n## Baseline run {ts}  config_hash={_hash}\n")
    f.write(f"- Gate 1 (reproduction): {n_match}/327 exact matches\n")
    f.write(f"- Gate 2 (IG macro): top1={ig_macro['top1']:.2f} top5={ig_macro['top5']:.2f} "
            f"legacy={ig_macro['legacy']:.2f}\n")
    f.write(f"- Gate 3 (random): top1={rnd_macro['top1']:.2f} top5={rnd_macro['top5']:.2f}\n")
    f.write(f"- Output: results_c/baseline_metrics.csv, baseline_per_segment.csv\n")
    f.write(out_df.to_string(index=False) + "\n")

print(f"\nLog appended to {log_path}")
