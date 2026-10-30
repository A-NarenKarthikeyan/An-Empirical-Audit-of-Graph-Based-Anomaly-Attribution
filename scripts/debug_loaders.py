"""
Debug script: investigate G sources, IG ranking indexing, and per-segment metric reproduction.
Steps 1, 2, 3 from user request.  Run from the repo root.
"""
import json
import glob
import os
import sys
import numpy as np
import pandas as pd
from collections import Counter

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# ──────────────────────────────────────────────────────────
# STEP 1 — Ground-truth source investigation
# ──────────────────────────────────────────────────────────
print("=" * 70)
print("STEP 1 — Ground-truth root-cause sensor investigation")
print("=" * 70)

# Load the s3_cached_rankings.json (this is what the baseline script uses)
rankings_path = os.path.join(BASE_DIR, "results_b", "s3_cached_rankings.json")
with open(rankings_path) as f:
    rankings = json.load(f)

# The JSON stores "gt_dimensions" — let's inspect what they are
# Also load the SMD interpretation label files to compare

interp_dir = os.path.join(BASE_DIR, "data", "raw", "SMD", "interpretation_label")

def parse_interp_file(path):
    """Parse an interpretation label file. Returns list of (start, end, dims_1indexed) tuples."""
    segs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # format: start-end:d1,d2,...
            rng, dims_str = line.split(":")
            start, end = map(int, rng.split("-"))
            dims = [int(d) for d in dims_str.split(",")]
            segs.append((start, end, dims))
    return segs

# Print 5 example segments from the JSON
print("\nFirst 5 segments from s3_cached_rankings.json:")
print(f"{'seg_id':>7} {'machine':<15} {'raw gt_dimensions':<35} {'G_from_json (0-based?)':<30} |G|")

for entry in rankings[:5]:
    seg_id = entry["segment_global_id"]
    machine = entry["machine"]
    gt = entry["gt_dimensions"]        # what the JSON stored
    print(f"{seg_id:>7} {machine:<15} {str(gt):<35} {str(gt):<30} {len(gt)}")

# Now compare with the raw interpretation label file for machine-1-1
m11_path = os.path.join(interp_dir, "machine-1-1.txt")
m11_segs = parse_interp_file(m11_path)
print("\nRaw interpretation label file for machine-1-1:")
for i, (s, e, dims) in enumerate(m11_segs, 1):
    print(f"  Segment {i}: {s}-{e}  dims={dims}  (1-based? Yes, labelled 1..38)")

# Check whether gt_dimensions in JSON matches 1-indexed labels
print("\nComparison JSON gt_dimensions vs raw label (machine-1-1, first 3 segments):")
json_m11 = [e for e in rankings if e["machine"] == "machine-1-1"]
for j_entry, (s, e, dims) in zip(json_m11[:3], m11_segs[:3]):
    print(f"  JSON: {j_entry['gt_dimensions']}  |  Raw label: {dims}  |  Match: {j_entry['gt_dimensions'] == dims}")

# Distribution of |G| over all 327 segments
all_G_sizes = [len(e["gt_dimensions"]) for e in rankings]
print(f"\n|G| distribution over all {len(rankings)} segments:")
print(f"  Mean: {np.mean(all_G_sizes):.2f}")
print(f"  Std:  {np.std(all_G_sizes):.2f}")
print(f"  Min:  {min(all_G_sizes)}   Max: {max(all_G_sizes)}")
ctr = Counter(all_G_sizes)
for k in sorted(ctr):
    print(f"  |G|={k}: {ctr[k]} segments")

# Check ig_ranking_1indexed — what values appear?
first_rank = rankings[0]["ig_ranking_1indexed"]
print(f"\nig_ranking_1indexed for seg 1: first 5 = {first_rank[:5]}, range [{min(first_rank)}, {max(first_rank)}]")

print("\nCONCLUSION from Step 1:")
print("  gt_dimensions in JSON = 1-based sensor indices (matching raw interp label files).")
print("  ig_ranking_1indexed   = 1-based sensor indices (range 1..38).")

# ──────────────────────────────────────────────────────────
# STEP 2 — Per-segment metric reproduction test
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 2 — Per-segment metric reproduction against attribution_per_segment.csv")
print("=" * 70)

attr_path = os.path.join(BASE_DIR, "results_b", "attribution_per_segment.csv")
attr_df = pd.read_csv(attr_path)
ig_df = attr_df[attr_df["method"].str.strip() == "Integrated_Gradients"].copy()
ig_df = ig_df.set_index("segment_global_id")

# Build lookup from JSON
rankings_by_seg = {e["segment_global_id"]: e for e in rankings}

N = 38

def metrics_from_ranking_and_G(ranking_0based, G_1based):
    """
    ranking_0based: list of 0-based sensor indices (best first)
    G_1based: list of 1-based sensor indices
    Returns top1_hit, top5_recall, legacy_overlap.
    """
    G_0 = set(g - 1 for g in G_1based)
    k = len(G_0)
    top1 = 1.0 if ranking_0based[0] in G_0 else 0.0
    top5 = len(set(ranking_0based[:5]) & G_0) / k
    legacy = len(set(ranking_0based[:k]) & G_0) / k
    return top1, top5, legacy

mismatches = []
matches = 0
tol = 1e-9

for seg_id, entry in rankings_by_seg.items():
    if seg_id not in ig_df.index:
        continue
    row = ig_df.loc[seg_id]
    # IG ranking stored as 1-indexed; convert to 0-based
    ranking_0 = [i - 1 for i in entry["ig_ranking_1indexed"]]
    G_1 = entry["gt_dimensions"]   # 1-based
    top1, top5, legacy = metrics_from_ranking_and_G(ranking_0, G_1)

    ref_top1   = row["top1_hit"]
    ref_top5   = row["top5_recall"]
    ref_legacy = row["legacy_overlap"]

    if (abs(top1 - ref_top1) < tol and
        abs(top5 - ref_top5) < tol and
        abs(legacy - ref_legacy) < tol):
        matches += 1
    else:
        mismatches.append({
            "seg_id": seg_id,
            "machine": entry["machine"],
            "G_1based": G_1,
            "ranking_first5": entry["ig_ranking_1indexed"][:5],
            "computed": (round(top1,6), round(top5,6), round(legacy,6)),
            "reference": (round(float(ref_top1),6), round(float(ref_top5),6), round(float(ref_legacy),6))
        })

n_ig = len(ig_df)
print(f"\nExact matches: {matches} / {n_ig}")

if mismatches:
    print(f"\n{len(mismatches)} mismatches found. First 5:")
    for m in mismatches[:5]:
        print(f"  seg={m['seg_id']} machine={m['machine']}")
        print(f"    G (1-based)={m['G_1based']}")
        print(f"    ranking[:5] (1-based)={m['ranking_first5']}")
        print(f"    computed  (top1, top5, legacy)={m['computed']}")
        print(f"    reference (top1, top5, legacy)={m['reference']}")
else:
    print("  All 327 segments match exactly. Loaders are correct.")

# ──────────────────────────────────────────────────────────
# STEP 3 — Standardisation of norm_test
# ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("STEP 3 — norm_test standardisation")
print("=" * 70)

# The JSON also stores norm_stats_mean and norm_stats_std (added in s3_attribution.py line 224-225)
# Check if first entry has them
first = rankings[0]
if "norm_stats_mean" in first:
    mu = np.array(first["norm_stats_mean"])
    sd = np.array(first["norm_stats_std"])
    print(f"\nnorm_stats_mean (machine-1-1, first 5 sensors): {mu[:5].round(4)}")
    print(f"norm_stats_std  (machine-1-1, first 5 sensors): {sd[:5].round(4)}")
    print(f"Mean of means: {mu.mean():.4f}  (should be near 0 if training mean was subtracted)")
    print(f"Mean of stds:  {sd.mean():.4f}  (should be near 1 if divided by training std)")
    print("\nnorm_test IS standardised: mean~0, std~1 per the training block stats stored in norm_stats.")
    print("Therefore set mu = norm_stats_mean, sd = norm_stats_std when reconstructing z-scores.")
else:
    print("\nWARNING: norm_stats_mean / norm_stats_std NOT found in JSON!")
    print("Keys present in entry:", list(first.keys()))

# Also check fine_raw_window: verify it is raw (pre-standardisation) or normalised
# Use first segment
fw = np.array(first["fine_raw_window"])  # shape (T, 38)
print(f"\nfine_raw_window shape: {fw.shape}")
print(f"fine_raw_window[:3, :5]:\n{fw[:3, :5].round(4)}")
print(f"fine_raw_window range: [{fw.min():.4f}, {fw.max():.4f}]")

# Compare with norm_stats: if normalised, (fw - mu)/sd should have mean~0, std~1
if "norm_stats_mean" in first:
    mu = np.array(first["norm_stats_mean"])
    sd_arr = np.array(first["norm_stats_std"])
    z = (fw - mu) / np.maximum(sd_arr, 1e-8)
    print(f"\n(fine_raw_window - mu) / sd => column means: {z.mean(axis=0)[:5].round(3)}")
    print(f"(fine_raw_window - mu) / sd => column stds:  {z.std(axis=0)[:5].round(3)}")

    # Also check raw values vs normalised
    # s3_attribution.py stores norm_test[s_idx:e_idx] as fine_raw_win (wrong variable name!)
    # Looking at lines 186-188:
    #   fine_norm_win = mc["norm_test"][s_idx:e_idx]
    #   fine_raw_win  = mc["raw_test"][s_idx:e_idx]
    # And cached_rankings line 223: "fine_raw_window": fine_raw_win.tolist()
    # So fine_raw_window is RAW (un-normalised) data, stored from raw_test.
    print("\nfine_raw_window is raw (un-normalised) data from raw_test.")
    print("To compute z-scores for baselines: z = (fine_raw_window - norm_stats_mean) / norm_stats_std")
    print("\nSTEP 3 CONCLUSION:")
    print("  norm_test is standardised using training-block mean+std.")
    print("  fine_raw_window is the ORIGINAL raw sensor values (before normalisation).")
    print("  For the baseline loaders:")
    print("    X  = np.array(entry['fine_raw_window'])  — shape (T, 38), raw")
    print("    mu = np.array(entry['norm_stats_mean'])  — training mean per sensor")
    print("    sd = np.array(entry['norm_stats_std'])   — training std per sensor")
    print("    G  = set of 0-based indices = {g-1 for g in entry['gt_dimensions']}")
