"""One fixed reciprocal-rank fusion analysis for detector seeds 42, 43, 44."""
from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RB = ROOT / "results_b"
RANKINGS_PATH = RB / "s3_cached_rankings.json"
SEEDS_OUT = ROOT / "results_c" / "seeds"
SEEDS = (42, 43, 44)
BOOTSTRAPS = 10_000
BOOTSTRAP_SEED = 42
RRF_OFFSET = 60


def load_analysis_module():
    path = OUT / "improve_analysis.py"
    spec = importlib.util.spec_from_file_location("improve_analysis_for_rrf", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import existing Table 5 analysis: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_tuple(ranking: list[int], truth: set[int]) -> tuple[float, float, float]:
    k = len(truth)
    if not k:
        return float("nan"), float("nan"), float("nan")
    return (float(ranking[0] in truth),
            len(set(ranking[:5]) & truth) / k,
            len(set(ranking[:k]) & truth) / k)


def rank_maxz(entry: dict) -> list[int]:
    x = np.asarray(entry["fine_raw_window"], dtype=float)
    mu = np.asarray(entry["norm_stats_mean"], dtype=float)
    sd = np.asarray(entry["norm_stats_std"], dtype=float)
    sd_adj = np.where(sd < 1e-8, 1.0, sd)
    z = (x - mu) / sd_adj
    z[:, sd_adj == 1.0] = 0.0
    score = np.abs(z).max(axis=0)
    return list(np.argsort(-score, kind="stable"))


def machine_macro(df: pd.DataFrame, col: str) -> float:
    return float(df.groupby("machine")[col].mean().mean() * 100.0)


def bootstrap_diff(df: pd.DataFrame, diff_col: str) -> tuple[float, float, float]:
    machine_means = df.groupby("machine")[diff_col].mean().to_numpy(dtype=float)
    point = float(machine_means.mean() * 100.0)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    sampled = rng.integers(0, len(machine_means), size=(BOOTSTRAPS, len(machine_means)))
    boot = machine_means[sampled].mean(axis=1) * 100.0
    low, high = np.percentile(boot, [2.5, 97.5])
    return point, float(low), float(high)


def check_seed42_gate(analysis, rankings: list[dict]) -> tuple[pd.DataFrame, dict[int, dict]]:
    base, _ = analysis.aggregate_attributions(rankings)
    fields = {
        "IG": ("IG_top5", "IG_top1", "IG_legacy"),
        "Max |z|": ("max_abs_z_top5", "max_abs_z_top1", "max_abs_z_legacy"),
    }
    got = {}
    for label, (top5, top1, legacy) in fields.items():
        got[label] = (machine_macro(base, top5), machine_macro(base, top1), machine_macro(base, legacy))
        print(f"REPRODUCTION GATE {label}: {got[label][0]:.6f} / {got[label][1]:.6f} / {got[label][2]:.6f}")
    expected = {"IG": (53.93, 83.89, 62.85), "Max |z|": (59.55, 80.38, 72.55)}
    passed = len(base) == 327 and base.machine.nunique() == 28 and all(
        abs(a - b) <= 0.01 for label in expected for a, b in zip(got[label], expected[label]))
    print(f"REPRODUCTION GATE segments={len(base)} machines={base.machine.nunique()}")
    print("REPRODUCTION GATE " + ("PASSED" if passed else "FAILED"))
    if not passed:
        raise SystemExit(2)
    return base, {int(e["segment_global_id"]): e for e in rankings}


def seed_ig_rankings(seed: int, entries: list[dict], seed42_entries: dict[int, dict]) -> dict[int, list[int]]:
    if seed == 42:
        return {gid: [int(sensor) - 1 for sensor in entry["ig_ranking_1indexed"]]
                for gid, entry in seed42_entries.items()}

    from src.data_loader import (load_smd_raw, normalize_data,
                                 compute_sensor_correlation_graph, compute_node_features_6d)
    from src.gnn_attribution.model import GATAnomalyDetector
    from src.gnn_attribution.validate_attribution import compute_integrated_gradients_attribution

    torch.set_num_threads(1)
    machines = sorted({entry["machine"] for entry in entries})
    machine_data = {}
    for machine in machines:
        train_raw, test_raw, _, _ = load_smd_raw(str(ROOT / "data" / "raw"), machine)
        train_norm, test_norm, norm_stats = normalize_data(train_raw, test_raw)
        edge_index, edge_attr = compute_sensor_correlation_graph(train_norm, threshold=0.6)
        machine_data[machine] = {"norm_test": test_norm, "norm_stats": norm_stats,
                                 "edge_index": edge_index, "edge_attr": edge_attr}

    segment_objects = []
    for entry in entries:
        md = machine_data[entry["machine"]]
        start = max(0, int(entry["start"] - 15))
        end = min(len(md["norm_test"]), int(entry["end"] + 15) + 1)
        window = md["norm_test"][start:end]
        node_features = compute_node_features_6d(window)
        edge_index = torch.tensor(md["edge_index"], dtype=torch.long)
        edge_attr = torch.tensor(md["edge_attr"], dtype=torch.float32) if md["edge_attr"] is not None else None
        graph = Data(x=torch.tensor(node_features, dtype=torch.float32), edge_index=edge_index, edge_attr=edge_attr)
        segment_objects.append((entry, graph))

    model_by_machine = {}
    for machine in machines:
        checkpoint_path = RB / "checkpoints" / f"{machine}_seed{seed}.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        model = GATAnomalyDetector(in_dim=6, hidden_dim=64, num_heads=4, edge_dim=1, dropout=0.2)
        model.load_state_dict(state)
        model.eval()
        model_by_machine[machine] = model

    result = {}
    for index, (entry, graph) in enumerate(segment_objects, 1):
        scores = compute_integrated_gradients_attribution(model_by_machine[entry["machine"]], graph, steps=50).cpu().numpy()
        result[int(entry["segment_global_id"])] = [int(i) for i in np.argsort(scores)[::-1]]
        if index % 50 == 0 or index == len(segment_objects):
            print(f"Seed {seed}: reconstructed IG rankings {index}/{len(segment_objects)}", flush=True)
    return result


def main() -> None:
    analysis = load_analysis_module()
    entries = json.loads(RANKINGS_PATH.read_text(encoding="utf-8"))
    if len(entries) != 327:
        raise RuntimeError(f"Expected 327 segments in saved Table 5 ranking cache, found {len(entries)}")

    # Required baseline reproduction gate runs before any fusion scores are computed.
    base, entry_by_gid = check_seed42_gate(analysis, entries)
    table5_by_gid = base.set_index("segment_global_id")
    maxz_rankings = {gid: rank_maxz(entry) for gid, entry in entry_by_gid.items()}
    truths = {int(entry["segment_global_id"]): {int(s) - 1 for s in entry["gt_dimensions"]}
              for entry in entries}
    machines = {int(entry["segment_global_id"]): entry["machine"] for entry in entries}

    all_rows = []
    summary_rows = []
    metric_names = ("top5_recall", "top1_hit", "legacy_overlap")
    for seed in SEEDS:
        ig_rankings = seed_ig_rankings(seed, entries, entry_by_gid)
        if len(ig_rankings) != 327:
            raise RuntimeError(f"Seed {seed}: expected IG rankings for all 327 segments, got {len(ig_rankings)}")

        per_segment = []
        for entry in entries:
            gid = int(entry["segment_global_id"])
            ig_rank = ig_rankings[gid]
            maxz_rank = maxz_rankings[gid]
            ig_pos = {sensor: rank for rank, sensor in enumerate(ig_rank, 1)}
            maxz_pos = {sensor: rank for rank, sensor in enumerate(maxz_rank, 1)}
            sensors = range(38)
            scores = {sensor: 1.0 / (RRF_OFFSET + ig_pos[sensor]) +
                      1.0 / (RRF_OFFSET + maxz_pos[sensor]) for sensor in sensors}
            fused_rank = sorted(sensors, key=lambda sensor: (-scores[sensor], sensor))
            truth = truths[gid]
            ranked_ig_metrics = metric_tuple(ig_rank, truth)
            if seed == 42:
                # Table 5's original per-segment IG metrics use its recorded top-k
                # result; retain those values for the official seed-42 comparison.
                ref = table5_by_gid.loc[gid]
                ig_metrics = (float(ref.IG_top1), float(ref.IG_top5), float(ref.IG_legacy))
            else:
                ig_metrics = ranked_ig_metrics
            ref = table5_by_gid.loc[gid]
            maxz_metrics = (float(ref.max_abs_z_top1), float(ref.max_abs_z_top5), float(ref.max_abs_z_legacy))
            fused_metrics = metric_tuple(fused_rank, truth)
            per_segment.append({"segment_global_id": gid, "machine": machines[gid],
                "ig_top1": ig_metrics[0], "ig_top5": ig_metrics[1], "ig_legacy": ig_metrics[2],
                "ranked_ig_top1": ranked_ig_metrics[0], "ranked_ig_top5": ranked_ig_metrics[1], "ranked_ig_legacy": ranked_ig_metrics[2],
                "maxz_top1": maxz_metrics[0], "maxz_top5": maxz_metrics[1], "maxz_legacy": maxz_metrics[2],
                "fusion_top1": fused_metrics[0], "fusion_top5": fused_metrics[1], "fusion_legacy": fused_metrics[2]})
        df = pd.DataFrame(per_segment)

        # Cross-check reconstructed IG ranking metrics against saved outputs for each seed.
        saved_path = SEEDS_OUT / f"attribution_per_segment_seed{seed}.csv"
        saved = pd.read_csv(saved_path).set_index("seg_id")
        reconstructed = df.set_index("segment_global_id")
        matched = 0
        for gid, rec in reconstructed.iterrows():
            old = saved.loc[int(gid)]
            ours = (rec.ranked_ig_top1, rec.ranked_ig_top5, rec.ranked_ig_legacy)
            prior = (float(old.top1), float(old.top5), float(old.legacy))
            if all(abs(a - b) < 1e-9 for a, b in zip(ours, prior)):
                matched += 1
        print(f"Seed {seed} saved-IG metric cross-check: {matched}/327 exact rows")

        for metric, suff in zip(metric_names, ("top5", "top1", "legacy")):
            fusion_col, ig_col, maxz_col = f"fusion_{suff}", f"ig_{suff}", f"maxz_{suff}"
            row = {"seed": seed, "metric": metric,
                   "n_segments": len(df), "n_machines": df.machine.nunique(),
                   "fusion_machine_macro_pct": machine_macro(df, fusion_col),
                   "ig_machine_macro_pct": machine_macro(df, ig_col),
                   "maxz_machine_macro_pct": machine_macro(df, maxz_col)}
            for comparator, col in (("ig", ig_col), ("maxz", maxz_col)):
                diff_col = f"diff_fusion_minus_{comparator}"
                df[diff_col] = df[fusion_col] - df[col]
                point, low, high = bootstrap_diff(df, diff_col)
                row[f"fusion_minus_{comparator}_pp"] = point
                row[f"fusion_minus_{comparator}_ci95_lower_pp"] = low
                row[f"fusion_minus_{comparator}_ci95_upper_pp"] = high
            row["bootstrap_resamples"] = BOOTSTRAPS
            row["bootstrap_seed"] = BOOTSTRAP_SEED
            all_rows.append(row)
            summary_rows.append(row)

    out_path = OUT / "fusion_rrf.csv"
    fieldnames = list(all_rows[0].keys())
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    print("\nRRF fixed recipe: score=1/(60+rank_IG)+1/(60+rank_maxz); fused-score ties: lower sensor index")
    print("seed | metric | fusion % | IG % | max-|z| % | fusion-IG pp [95% CI] | fusion-max-|z| pp [95% CI]")
    for row in summary_rows:
        print(f"{row['seed']} | {row['metric']} | {row['fusion_machine_macro_pct']:.4f} | "
              f"{row['ig_machine_macro_pct']:.4f} | {row['maxz_machine_macro_pct']:.4f} | "
              f"{row['fusion_minus_ig_pp']:+.4f} [{row['fusion_minus_ig_ci95_lower_pp']:+.4f}, {row['fusion_minus_ig_ci95_upper_pp']:+.4f}] | "
              f"{row['fusion_minus_maxz_pp']:+.4f} [{row['fusion_minus_maxz_ci95_lower_pp']:+.4f}, {row['fusion_minus_maxz_ci95_upper_pp']:+.4f}]")
    print(f"WROTE {out_path} rows={len(all_rows)}")


if __name__ == "__main__":
    main()
