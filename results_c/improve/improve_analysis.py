"""Reproducible, read-only analyses for the improve request.

All generated artifacts are written beside this script under results_c/improve.
No rater labels are created or modified; results_b is read-only.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RB = ROOT / "results_b"
sys.path.insert(0, str(ROOT))
SEED = 42
NBOOT = 10_000
EXPECTED = {
    "ig": (83.89, 53.93, 62.84),
    "maxz_top5": 59.55,
}


def log_step(title: str, lines: list[str]) -> None:
    with (OUT / "LOG.md").open("a", encoding="utf-8") as f:
        f.write(f"\n## {title}\n")
        for line in lines:
            f.write(f"- {line}\n")


def ensure_common_gate() -> None:
    p = OUT / "GATE_COMMON.txt"
    if not p.exists() or "COMMON GATE PASSED" not in p.read_text(encoding="utf-8"):
        raise RuntimeError("common gate has not passed; see GATE_COMMON.txt")


def read_cache() -> list[dict]:
    return json.loads((RB / "s3_cached_rankings.json").read_text(encoding="utf-8"))


def metric_tuple(ranking: list[int], truth: set[int]) -> tuple[float, float, float]:
    k = len(truth)
    if not k:
        return np.nan, np.nan, np.nan
    return float(ranking[0] in truth), len(set(ranking[:5]) & truth) / k, len(set(ranking[:k]) & truth) / k


def rank_statistics(entry: dict) -> dict[str, tuple[float, float, float]]:
    truth = {int(v) - 1 for v in entry["gt_dimensions"]}
    x = np.asarray(entry["fine_raw_window"], dtype=float)
    mu = np.asarray(entry["norm_stats_mean"], dtype=float)
    sd = np.asarray(entry["norm_stats_std"], dtype=float)
    sd_adj = np.where(sd < 1e-8, 1.0, sd)
    z = (x - mu) / sd_adj
    z[:, sd_adj == 1.0] = 0.0
    scores = {
        "max_abs_z": np.abs(z).max(axis=0),
        "mean_abs_z": np.abs(z).mean(axis=0),
        "mean_shift": np.abs(x.mean(axis=0) - mu) / sd_adj,
    }
    ret = {}
    for name, score in scores.items():
        ranking = list(np.argsort(-score, kind="stable"))
        ret[name] = metric_tuple(ranking, truth)
    return ret


def machine_macro(values: pd.DataFrame, cols: list[str]) -> pd.Series:
    return values.groupby("machine")[cols].mean().mean()


def cluster_bootstrap(values: pd.DataFrame, value_col: str, *, nboot: int = NBOOT, seed: int = SEED) -> tuple[float, float]:
    by_machine = values.groupby("machine")[value_col].mean()
    arr = by_machine.to_numpy(float)
    if len(arr) < 5:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(arr), size=(nboot, len(arr)))
    boot = arr[idx].mean(axis=1)
    return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def aggregate_attributions(cache: list[dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(RB / "attribution_per_segment.csv")
    igsrc = source[source.method.astype(str).str.strip() == "Integrated_Gradients"]
    records = []
    for e in cache:
        sid = int(e["segment_global_id"])
        r = igsrc.loc[igsrc.segment_global_id == sid]
        if len(r) != 1:
            raise RuntimeError(f"expected one stored IG row for segment {sid}; found {len(r)}")
        r = r.iloc[0]
        mets = rank_statistics(e)
        # The saved IG ranking is the source ranking; score vectors were gate-checked in Part 3.
        records.append({
            "segment_global_id": sid, "machine": e["machine"], "start": int(e["start"]),
            "end": int(e["end"]), "k_gt": len(e["gt_dimensions"]),
            "IG_top1": float(r.top1_hit), "IG_top5": float(r.top5_recall),
            "IG_legacy": float(r.legacy_overlap),
            **{f"{name}_{m}": val for name, tup in mets.items() for m, val in zip(("top1", "top5", "legacy"), tup)},
        })
    return pd.DataFrame(records), source


def part1(cache: list[dict], base: pd.DataFrame) -> None:
    ensure_common_gate()
    gate_lines = ["PART 1 GATE: common gate passed; preparing seed-42 window predictions from the saved detector checkpoints."]
    try:
        import torch
        import torch.nn as nn
        from torch_geometric.data import Data
        from torch_geometric.loader import DataLoader
        from src.data_loader import load_smd_raw, normalize_data, compute_sensor_correlation_graph, compute_node_features_6d
        from src.gnn_attribution.model import GATAnomalyDetector
        from src.gnn_attribution.train import compute_classification_metrics
    except Exception as exc:
        raise RuntimeError(f"PART 1 GATE FAILED: detector inference dependencies unavailable: {type(exc).__name__}: {exc}") from exc

    class FastDataset(torch.utils.data.Dataset):
        def __init__(self, x, edge_index, edge_attr):
            self.x = torch.from_numpy(x).float()
            self.ei = edge_index
            self.ea = edge_attr
        def __len__(self):
            return len(self.x)
        def __getitem__(self, i):
            return Data(x=self.x[i], edge_index=self.ei, edge_attr=self.ea)

    torch.set_num_threads(1)
    dev = torch.device("cpu")
    out_pred = {}
    gate_metrics = []
    machines = sorted({e["machine"] for e in cache})
    cache_by_machine = {m: [e for e in cache if e["machine"] == m] for m in machines}
    metric_dir = RB / "metrics"
    ckpt_dir = RB / "checkpoints"
    for ix, machine in enumerate(machines, 1):
        meta = json.loads((metric_dir / f"{machine}_seed42.json").read_text(encoding="utf-8"))
        ckpt_path = ckpt_dir / f"{machine}_seed42.pt"
        sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
        if sha != meta.get("checkpoint_sha256"):
            raise RuntimeError(f"PART 1 GATE FAILED: checkpoint hash mismatch for {machine}")
        train_raw, test_raw, labels, _ = load_smd_raw(str(ROOT / "data" / "raw"), machine)
        train_norm, test_norm, _ = normalize_data(train_raw, test_raw)
        edge_index, edge_attr = compute_sensor_correlation_graph(train_norm, threshold=0.6)
        total = len(test_norm) - 100 + 1
        train_end = int(total * 0.70)
        val_end = train_end + int(total * 0.15)
        test_slice = slice(val_end, total)
        score_thresh = float(meta["val_optimal_threshold"])
        model = GATAnomalyDetector(in_dim=6, hidden_dim=64, num_heads=4, edge_dim=1, dropout=0.2)
        ckpt = torch.load(ckpt_path, map_location=dev, weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
        model.eval()
        ei = torch.tensor(edge_index, dtype=torch.long)
        ea = torch.tensor(edge_attr, dtype=torch.float32) if edge_attr is not None else None
        prob_chunks = []
        for s in range(0, total, 1000):
            en = min(s + 1000, total)
            raw_chunk = test_norm[s:en + 99]
            windows = np.asarray([raw_chunk[j:j + 100] for j in range(en - s)], dtype=np.float32)
            feats = compute_node_features_6d(windows).astype(np.float32)
            loader = DataLoader(FastDataset(feats, ei, ea), batch_size=64, shuffle=False)
            chunk_p = []
            with torch.no_grad():
                for batch in loader:
                    batch = batch.to(dev)
                    probs, _ = model(x=batch.x, edge_index=batch.edge_index, edge_attr=batch.edge_attr, batch=batch.batch)
                    chunk_p.extend(probs.view(-1).cpu().numpy().tolist())
            prob_chunks.extend(chunk_p)
        probs = np.asarray(prob_chunks, dtype=float)
        if len(probs) != total:
            raise RuntimeError(f"PART 1 GATE FAILED: {machine} got {len(probs)} probabilities for {total} windows")
        y_win = np.asarray([int(np.any(labels[i:i + 100] > 0)) for i in range(total)], dtype=int)
        test_m = compute_classification_metrics(y_win[test_slice], probs[test_slice], threshold=score_thresh)
        ref = meta["test_metrics"]
        for k in ("precision", "recall", "f1", "auroc"):
            if not np.isclose(float(test_m[k]), float(ref[k]), atol=1e-5, rtol=0):
                raise RuntimeError(f"PART 1 GATE FAILED: {machine} reproduced test {k}={test_m[k]:.8f}, saved={ref[k]}")
        gate_metrics.append((machine, float(test_m["f1"]), float(ref["f1"]), len(probs), val_end))
        out_pred[machine] = (probs >= score_thresh, total, val_end)
        print(f"PART 1 inference {ix}/{len(machines)} {machine} windows={total} threshold={score_thresh:.4f}", flush=True)

    gate_lines.extend([f"PART 1 GATE PASSED: detector test metrics reproduced for {len(gate_metrics)}/28 machines.",
                       "Window predictions: `scratch/option_b/s1_detectors.py::run_single_detector_task` split/feature recipe; same `compute_node_features_6d` and saved checkpoint; window probabilities thresholded at `results_b/metrics/<machine>_seed42.json::val_optimal_threshold`. Segment hit = any window whose start index is in inclusive [start,end]."])
    print("\n".join(gate_lines), flush=True)
    rows = []
    for r in base.to_dict("records"):
        pred, total, test_start = out_pred[r["machine"]]
        lo = max(0, int(r["start"]))
        hi = min(total - 1, int(r["end"]))
        hit = bool(hi >= lo and pred[lo:hi + 1].any())
        rows.append({**r, "detector_outcome": "detected" if hit else "missed", "test_block_segment": bool(lo >= test_start and lo < total)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "part1_segment_outcomes.csv", index=False)
    summaries = []
    for subset_name, sub in [("all_327", df), ("test_block_only", df[df.test_block_segment])]:
        for outcome, grp in sub.groupby("detector_outcome", sort=True):
            n_machines = grp.machine.nunique()
            row = {"subset": subset_name, "outcome": outcome, "n_segments": len(grp), "n_machines": n_machines}
            for meth in ("IG", "max_abs_z"):
                for metric in ("top1", "top5", "legacy"):
                    col = f"{meth}_{metric}"
                    row[f"{meth}_{metric}_machine_macro_pct"] = grp.groupby("machine")[col].mean().mean() * 100 if len(grp) else np.nan
            d = grp["IG_top5"] - grp["max_abs_z_top5"]
            row["paired_top5_diff_machine_macro_pp"] = grp.assign(_diff=d).groupby("machine")["_diff"].mean().mean() * 100 if len(grp) else np.nan
            if n_machines >= 5:
                lo, hi = cluster_bootstrap(grp.assign(_diff=d), "_diff")
                row["paired_diff_ci95_lower_pp"], row["paired_diff_ci95_upper_pp"] = lo * 100, hi * 100
                row["ci_note"] = "10,000 machine-cluster bootstrap, seed 42"
            else:
                row["paired_diff_ci95_lower_pp"] = row["paired_diff_ci95_upper_pp"] = np.nan
                row["ci_note"] = "fewer than 5 machines; CI omitted, segment-level values only"
            summaries.append(row)
    sdf = pd.DataFrame(summaries)
    sdf.to_csv(OUT / "part1_detected_missed_summary.csv", index=False)
    print("PART 1 SUMMARY\n" + sdf.to_string(index=False), flush=True)
    log_step("Part 1 — detector outcome split", [*gate_lines, "Wrote part1_segment_outcomes.csv and part1_detected_missed_summary.csv.", sdf.to_string(index=False)])


def part2(cache: list[dict], base: pd.DataFrame) -> None:
    ensure_common_gate()
    if len(base) != 327 or base.machine.nunique() != 28:
        raise RuntimeError(f"PART 2 GATE FAILED: segments={len(base)}, machines={base.machine.nunique()} (expected 327/28)")
    gate = f"PART 2 GATE PASSED: segments={len(base)} machines={base.machine.nunique()} common gate passed."
    print(gate)
    rows = []
    for m, g in base.groupby("machine"):
        diff_pp = (g.IG_top5 - g.max_abs_z_top5).mean() * 100
        if abs(diff_pp) < 1e-10:
            diff_pp = 0.0
        rows.append({"machine": m, "n_segments": len(g), "ig_top5_pct": g.IG_top5.mean() * 100,
                     "max_abs_z_top5_pct": g.max_abs_z_top5.mean() * 100,
                     "difference_ig_minus_maxz_pp": diff_pp})
    pm = pd.DataFrame(rows).sort_values("machine")
    if len(pm) != 28:
        raise RuntimeError(f"PART 2 GATE FAILED: per-machine output has {len(pm)} rows")
    pm.to_csv(OUT / "per_machine_top5.csv", index=False)

    bins = [(1, 2, "1-2"), (3, 5, "3-5"), (6, 10, "6-10"), (11, 10_000, ">10")]
    binrows = []
    for low, high, label in bins:
        g = base[(base.k_gt >= low) & (base.k_gt <= high)].copy()
        if g.empty:
            continue
        row = {"G_bin": label, "n_segments": len(g), "n_machines": g.machine.nunique()}
        for method in ("IG", "max_abs_z"):
            for metric in ("top1", "top5", "legacy"):
                row[f"{method}_{metric}_machine_macro_pct"] = g.groupby("machine")[f"{method}_{metric}"].mean().mean() * 100
        g["diff"] = g.IG_top5 - g.max_abs_z_top5
        row["paired_top5_diff_machine_macro_pp"] = g.groupby("machine")["diff"].mean().mean() * 100
        if row["n_machines"] >= 5:
            lo, hi = cluster_bootstrap(g, "diff")
            row["diff_ci95_lower_pp"], row["diff_ci95_upper_pp"] = lo * 100, hi * 100
            row["ci_note"] = "10,000 machine-cluster bootstrap, seed 42"
        else:
            row["diff_ci95_lower_pp"] = row["diff_ci95_upper_pp"] = np.nan
            row["ci_note"] = "fewer than 5 machines; CI omitted, segment-level group values"
        binrows.append(row)
    bdf = pd.DataFrame(binrows)
    bdf.to_csv(OUT / "part2_by_ground_truth_size.csv", index=False)
    print("PART 2 PER MACHINE\n" + pm.to_string(index=False), flush=True)
    print("PART 2 BY |G|\n" + bdf.to_string(index=False), flush=True)
    log_step("Part 2 — heterogeneity", [gate, "Wrote per_machine_top5.csv and part2_by_ground_truth_size.csv.", "Per-machine table:", pm.to_string(index=False), "By-|G| table:", bdf.to_string(index=False)])


def prefix_lengths(entry: dict, ranking: np.ndarray) -> tuple[int, int]:
    gt = {int(x) for x in entry["gt_dimensions"]}
    n = len(gt)
    pos = {int(sensor): i + 1 for i, sensor in enumerate(ranking)}
    p100 = max(pos[x] for x in gt)
    need80 = max(1, int(math.ceil(0.8 * n)))
    seen = 0
    p80 = len(ranking)
    for i, sensor in enumerate(ranking, 1):
        if int(sensor) in gt:
            seen += 1
            if seen >= need80:
                p80 = i
                break
    return p100, p80


def conformal_quantile_prefix(s: np.ndarray, alpha: float) -> int:
    n = len(s)
    q = min(1.0, max(0.0, math.ceil((n + 1) * (1 - alpha)) / n))
    return int(np.quantile(s, q, method="higher"))


def threshold_from_t(t: np.ndarray, alpha: float) -> float:
    k = math.floor(alpha * (len(t) + 1))
    if k < 1:
        return 0.0
    return float(np.sort(t)[k - 1])


def part3(cache: list[dict]) -> None:
    ensure_common_gate()
    n_match = 0
    vals = []
    for e in cache:
        a = np.asarray(e["ig_scores"], dtype=float)
        ranking = np.argsort(a)[::-1]  # Stored order is generated by argsort(scores)[::-1].
        stored = np.asarray(e["ig_ranking_1indexed"], dtype=int) - 1
        n_match += int(np.array_equal(ranking, stored))
        vals.append(float(a.sum()))
    score_gate = f"PART 3 SCORE GATE: normalized IG score vectors={len(vals)}, sum range=[{min(vals):.9f},{max(vals):.9f}], full-ranking matches={n_match}/327 (need >=325)."
    print(score_gate)
    if n_match < 325 or not (np.all(np.isfinite(vals)) and min(vals) > 0.999 and max(vals) < 1.001):
        raise RuntimeError("PART 3 SCORE GATE FAILED: stored IG scores do not reproduce normalized stored rankings.")

    prefixes = []
    for e in cache:
        ranking = np.asarray(e["ig_ranking_1indexed"], dtype=int)
        p100, p80 = prefix_lengths(e, ranking)
        prefixes.append({"machine": e["machine"], "prefix_100": p100, "prefix_80": p80})
    pfx = pd.DataFrame(prefixes)
    ref = pd.read_csv(RB / "conformal_summary.csv")
    rng = np.random.RandomState(SEED)
    rows_pre = []
    for rho, col in [(1.0, "prefix_100"), (0.8, "prefix_80")]:
        scores = pfx[col].to_numpy()
        for alpha in (0.1, 0.2):
            covs = []
            for _ in range(1000):
                perm = rng.permutation(len(scores))
                cal, test = perm[:len(scores)//2], perm[len(scores)//2:]
                q = conformal_quantile_prefix(scores[cal], alpha)
                covs.append(np.mean(scores[test] <= q))
            loo_cov = []
            for machine in sorted(pfx.machine.unique()):
                tr = pfx[pfx.machine != machine][col].to_numpy()
                te = pfx[pfx.machine == machine][col].to_numpy()
                q = conformal_quantile_prefix(tr, alpha)
                loo_cov.extend((te <= q).tolist())
            match = ref[(ref.containment_rho == rho) & (ref.alpha == alpha)]
            q_ref = int(match.iloc[0].full_pool_q_hat)
            q_full = conformal_quantile_prefix(scores, alpha)
            rows_pre.append({"rho": rho, "alpha": alpha, "q_hat": q_full, "reference_q_hat": q_ref,
                             "held_out_coverage_pct": np.mean(covs)*100, "reference_coverage_pct": float(match.iloc[0].held_out_coverage_mean)*100,
                             "loo_coverage_pct": np.mean(loo_cov)*100})
    pfxdf = pd.DataFrame(rows_pre)
    values_ok = len(pfxdf) == 4 and all(abs(r.held_out_coverage_pct - r.reference_coverage_pct) <= 0.05 and r.q_hat == r.reference_q_hat for r in pfxdf.itertuples())
    prefix_gate = "PART 3 PREFIX GATE " + ("PASSED" if values_ok else "FAILED") + ": " + ", ".join(f"rho={r.rho:g},alpha={r.alpha:g} {r.held_out_coverage_pct:.2f}% (reference {r.reference_coverage_pct:.2f}%), q={r.q_hat}/{r.reference_q_hat}" for r in pfxdf.itertuples())
    print(prefix_gate)
    if not values_ok:
        raise RuntimeError("PART 3 PREFIX GATE FAILED: saved prefix-coverage results were not reproduced.")

    score_rows = []
    a_all = [np.asarray(e["ig_scores"], dtype=float) for e in cache]
    g_all = [{int(x)-1 for x in e["gt_dimensions"]} for e in cache]
    machines = np.asarray([e["machine"] for e in cache])
    n = len(cache)
    rng_s = np.random.RandomState(SEED)
    for rho in (1.0, 0.8):
        t = []
        needed = []
        for a, g in zip(a_all, g_all):
            need = int(math.ceil(rho * len(g)))
            needed.append(need)
            t.append(float(np.sort(a[list(g)])[::-1][need-1]))
        t = np.asarray(t)
        # One shared seed-42 stream, 1000 splits, matching the project prefix protocol.
        for alpha in (0.1, 0.2):
            coverages, sizes = [], []
            for _ in range(1000):
                perm = rng_s.permutation(n)
                cal, test = perm[:n//2], perm[n//2:]
                th = threshold_from_t(t[cal], alpha)
                cov, sz = [], []
                for j in test:
                    mask = a_all[j] >= th
                    cov.append(len(g_all[j] & set(np.flatnonzero(mask))) >= needed[j])
                    sz.append(int(mask.sum()))
                coverages.append(float(np.mean(cov)))
                sizes.extend(sz)
            loo_hits = []
            for machine in sorted(set(machines)):
                cal = np.flatnonzero(machines != machine)
                test = np.flatnonzero(machines == machine)
                th = threshold_from_t(t[cal], alpha)
                for j in test:
                    mask_sensors = set(np.flatnonzero(a_all[j] >= th))
                    loo_hits.append(len(g_all[j] & mask_sensors) >= needed[j])
            q_full = threshold_from_t(t, alpha)
            score_rows.append({"rho": rho, "alpha": alpha, "calibration_threshold": q_full,
                               "held_out_coverage_mean_pct": np.mean(coverages)*100,
                               "held_out_coverage_sd_pct": np.std(coverages, ddof=1)*100,
                               "mean_set_size": np.mean(sizes), "median_set_size": np.median(sizes),
                               "set_size_q1": np.percentile(sizes, 25), "set_size_q3": np.percentile(sizes, 75),
                               "loo_pooled_coverage_pct": np.mean(loo_hits)*100})
    scdf = pd.DataFrame(score_rows)
    cmp = pfxdf.merge(scdf, on=["rho", "alpha"], suffixes=("_prefix", "_score"), validate="one_to_one")
    cmp["prefix_set_size_sensors"] = cmp["q_hat"]
    cmp = cmp.rename(columns={"q_hat": "prefix_q_hat", "held_out_coverage_pct": "prefix_held_out_coverage_pct",
                              "loo_coverage_pct": "prefix_loo_pooled_coverage_pct"})
    cmp = cmp[["rho", "alpha", "prefix_q_hat", "prefix_set_size_sensors", "prefix_held_out_coverage_pct",
               "prefix_loo_pooled_coverage_pct", "calibration_threshold", "held_out_coverage_mean_pct",
               "mean_set_size", "median_set_size", "set_size_q1", "set_size_q3", "loo_pooled_coverage_pct"]]
    scdf.to_csv(OUT / "part3_score_threshold_conformal.csv", index=False)
    pfxdf.to_csv(OUT / "part3_prefix_gate_reproduction.csv", index=False)
    cmp.to_csv(OUT / "part3_prefix_vs_score_comparison.csv", index=False)
    print(score_gate); print(prefix_gate)
    print("PART 3 SCORE-THRESHOLD RESULTS\n" + scdf.to_string(index=False), flush=True)
    print("PART 3 PREFIX VS SCORE COMPARISON\n" + cmp.to_string(index=False), flush=True)
    log_step("Part 3 — score-threshold conformal sets", [score_gate, prefix_gate, "Wrote part3_score_threshold_conformal.csv, part3_prefix_gate_reproduction.csv, and part3_prefix_vs_score_comparison.csv.", scdf.to_string(index=False), "Prefix comparison:", cmp.to_string(index=False)])


def part4(base: pd.DataFrame, attr_source: pd.DataFrame) -> None:
    ensure_common_gate()
    methods = {
        "GNNExplainer": "GNNExplainer",
        "attention": "Layer2_Attention",
    }
    rows = []
    attr = attr_source.copy()
    for disp, method in methods.items():
        sub = attr[attr.method.astype(str).str.strip() == method]
        for r in sub.itertuples():
            rows.append({"segment_global_id": int(r.segment_global_id), "machine": r.machine,
                         "method": disp, "top1": float(r.top1_hit), "top5": float(r.top5_recall), "legacy": float(r.legacy_overlap)})
    for method in ("max_abs_z", "mean_abs_z", "mean_shift"):
        for r in base.itertuples():
            rows.append({"segment_global_id": int(r.segment_global_id), "machine": r.machine,
                         "method": method, "top1": float(getattr(r, method+"_top1")),
                         "top5": float(getattr(r, method+"_top5")), "legacy": float(getattr(r, method+"_legacy"))})
    long = pd.DataFrame(rows)
    wide = long.pivot(index=["segment_global_id", "machine"], columns="method", values="top5").reset_index()
    ig = base[["segment_global_id", "machine", "IG_top5"]]
    wide = wide.merge(ig, on=["segment_global_id", "machine"], validate="one_to_one")
    comps = ["GNNExplainer", "attention", "max_abs_z", "mean_abs_z", "mean_shift"]
    machines = sorted(wide.machine.unique())
    if len(machines) != 28 or any(wide[m].isna().any() for m in comps):
        raise RuntimeError("PART 4 GATE FAILED: missing one or more comparison rows across 28 machines")
    rng = np.random.default_rng(SEED)
    bidx = rng.integers(0, len(machines), size=(NBOOT, len(machines)))
    out = []
    for method in comps:
        pair = wide[["machine", "IG_top5", method]].copy()
        pair["diff"] = pair.IG_top5 - pair[method]
        dm = pair.groupby("machine")["diff"].mean().reindex(machines).to_numpy()
        boot = dm[bidx].mean(axis=1)
        p_raw = max(1/NBOOT, 2*min(np.mean(boot <= 0), np.mean(boot >= 0)))
        out.append({"comparison": f"IG minus {method}", "IG_top5_machine_macro_pct": wide.groupby("machine").IG_top5.mean().mean()*100,
                    "other_top5_machine_macro_pct": wide.groupby("machine")[method].mean().mean()*100,
                    "difference_pp": dm.mean()*100, "bootstrap_ci95_lower_pp": np.percentile(boot, 2.5)*100,
                    "bootstrap_ci95_upper_pp": np.percentile(boot, 97.5)*100, "p_raw": p_raw})
    odf = pd.DataFrame(out).sort_values("p_raw").reset_index(drop=True)
    m = len(odf)
    adj = []
    prev = 0.0
    for i, p in enumerate(odf.p_raw, 1):
        prev = max(prev, min(1.0, (m-i+1)*float(p)))
        adj.append(prev)
    odf["p_holm"] = adj
    odf["significant_holm_0_05"] = odf.p_holm <= 0.05
    odf.to_csv(OUT / "part4_holm_paired_top5.csv", index=False)
    print(f"PART 4 GATE PASSED: all five paired comparisons complete for {len(machines)} machines; {NBOOT} cluster resamples, seed {SEED}.")
    print("PART 4 RESULTS\n" + odf.to_string(index=False), flush=True)
    log_step("Part 4 — Holm correction", ["PART 4 GATE PASSED: 28 machines and all five comparison rows present.", "Bootstrap p = 2*min(P(d<=0),P(d>=0)); floor=1/10,000; Holm family size=5.", "Wrote part4_holm_paired_top5.csv.", odf.to_string(index=False)])


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if n == 0:
        return np.nan, np.nan
    p = k / n
    d = 1 + z*z/n
    c = (p + z*z/(2*n))/d
    h = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return max(0.0, c-h), min(1.0, c+h)


def rule_lexicon_assessment(explanation: str, lex: re.Pattern) -> list[dict]:
    """Return lexicon hits attached to any stated sensor magnitude with |z| < 2."""
    def lex_hits(text: str) -> list[str]:
        hits = []
        for match in lex.finditer(text):
            prefix = text[max(0, match.start() - 16):match.start()]
            if re.search(r"(?:\bno\s+|\bnot\s+|\bnon[-\s]?|\bwithout\s*)$", prefix, re.I):
                continue
            hits.append(match.group(0).lower())
        return sorted(set(hits))

    sensor_re = re.compile(r"Sensors?[_\s]*(\d+)", re.I)
    z_re = re.compile(r"(?<![\w.])([+\-−]?\d+(?:\.\d+)?|±\s*\d+(?:\.\d+)?)\s*(?:std|σ)\b", re.I)
    sensors = list(sensor_re.finditer(explanation))
    mentions = []
    for i, sensor_match in enumerate(sensors):
        stop = sensors[i + 1].start() if i + 1 < len(sensors) else len(explanation)
        tail = explanation[sensor_match.end():stop]
        z_matches = list(z_re.finditer(tail))
        # More than one magnitude after a sensor name is not attributable without
        # inventing an ordering (for example, "Sensor_1 and Sensor_2 shifts +x and +y").
        signed_values = re.findall(r"(?<![\w.])(?:[+\-−]|±)\s*\d+(?:\.\d+)?", tail)
        pre_value = tail[:z_matches[0].start()] if z_matches else tail
        unlabelled_sensor_list = bool(re.search(r"(?:,\s*\d+|\band\s+\d+)\b", pre_value, re.I))
        if len(z_matches) != 1 or len(signed_values) > 1 or unlabelled_sensor_list:
            continue
        z_match = z_matches[0]
        raw_z = z_match.group(1).replace("−", "-").replace("±", "").strip()
        z_value = abs(float(raw_z)) if raw_z else 0.0
        local = explanation[sensor_match.start():stop]
        mentions.append({"sensor": int(sensor_match.group(1)), "z": z_value,
                         "start": sensor_match.start(), "stop": stop,
                         "z_end": sensor_match.end() + z_match.end(),
                         "wording": local.strip(),
                         "hits": lex_hits(local)})

    # A predicate after a coordinated sensor list ("... and Sensor_34 show significant shifts")
    # applies to the listed sensors. Propagate only that shared trailing predicate;
    # otherwise keep descriptors within each sensor's own span.
    shared = {}
    boundaries = [m.end() for m in re.finditer(r"(?<!\d)[.!?](?:[\"”])?(?=\s|$)", explanation)]
    sent_starts = [0] + boundaries
    sent_ends = boundaries + [len(explanation)]
    for sent_start, sent_end in zip(sent_starts, sent_ends):
        sentence = explanation[sent_start:sent_end]
        contained = [m for m in mentions if sent_start <= m["start"] < sent_end]
        if len(contained) < 2:
            continue
        last = max(contained, key=lambda x: x["start"])
        tail = explanation[last["z_end"]:sent_end]
        tail_hits = lex_hits(tail)
        if tail_hits and re.search(r"\b(?:show|shows|exhibit|exhibits|indicate|indicates|represent|are)\b", tail, re.I):
            for mention in contained:
                shared[(mention["sensor"], mention["start"])] = (tail_hits, sentence.strip())

    findings = []
    for mention in mentions:
        if mention["z"] >= 2:
            continue
        hits = mention["hits"]
        wording = mention["wording"]
        if not hits and (mention["sensor"], mention["start"]) in shared:
            hits, wording = shared[(mention["sensor"], mention["start"])]
        if hits:
            findings.append({"sensor": mention["sensor"], "z": mention["z"],
                             "wording": wording, "lexicon_hits": "; ".join(hits)})
    return findings


def part5() -> None:
    ensure_common_gate()
    eval_path = RB / "final" / "faithfulness_results_v2.csv"
    strict_path = RB / "final" / "faithfulness_results.csv"
    cache_path = RB / "final" / "faithfulness_cache.jsonl"
    eval_df = pd.read_csv(eval_path)
    strict_df = pd.read_csv(strict_path)
    # The existing strict scorer's overall verdict; select the 80 records the v2 pipeline evaluated structurally.
    selected = eval_df[eval_df.struct_status.astype(str) == "tested"].copy()
    if len(selected) != 80:
        raise RuntimeError(f"PART 5 GATE FAILED: structural evaluated responses={len(selected)}, expected 80")
    strict_df = strict_df[["segment_global_id", "machine", "overall_faithful"]]
    selected = selected.merge(strict_df, on=["segment_global_id", "machine"], how="left", validate="one_to_one")
    if selected.overall_faithful.isna().any():
        raise RuntimeError(f"PART 5 GATE FAILED: missing strict-scoring verdicts for {selected.overall_faithful.isna().sum()} selected responses")
    cache = {}
    for line in cache_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        key = item["cache_key"]
        if key.endswith("_orig"):
            match = re.search(r"_(machine-\d+-\d+)_(\d+)_orig$", key)
            if match:
                cache[(match.group(1), int(match.group(2)))] = item["response"]
    if any((r.machine, int(r.segment_global_id)) not in cache for r in selected.itertuples()):
        raise RuntimeError("PART 5 GATE FAILED: at least one selected response is missing from faithfulness_cache.jsonl")
    gate = f"PART 5 GATE PASSED: selected={len(selected)} evaluated responses; strict verdict joined for all 80."
    print(gate)

    lexicon = ["significant", "substantial", "large", "massive", "extreme", "strong", "primary driver", "dominant", "far outside"]
    lex = re.compile(r"\b(?:significant|substantial|large|massive|extreme|strong|primary\s+drivers?|dominant|far\s+outside)\b", re.I)
    id_to_audit = {}
    audit_path = RB / "final" / "audit_key.csv"
    if audit_path.exists():
        ak = pd.read_csv(audit_path)
        id_to_audit = { (str(r.machine), int(r.segment_global_id)): int(r.audit_id) for r in ak.itertuples() }
    responses = []
    examples = []
    for r in selected.itertuples():
        response = cache[(r.machine, int(r.segment_global_id))]
        try:
            parsed = json.loads(response)
            explanation = str(parsed.get("explanation", ""))
        except Exception:
            explanation = response
        findings = rule_lexicon_assessment(explanation, lex)
        over = bool(findings)
        audit_id = id_to_audit.get((r.machine, int(r.segment_global_id)), int(r.segment_global_id))
        responses.append({"audit_id": audit_id, "segment_global_id": int(r.segment_global_id), "machine": r.machine,
                          "strict_scorer_verdict": str(r.overall_faithful), "overstated": over,
                          "flagged_sensor_details": json.dumps(findings, ensure_ascii=False)})
        for f in findings:
            examples.append({"audit_id": audit_id, "segment_global_id": int(r.segment_global_id), "machine": r.machine,
                             "strict_scorer_verdict": str(r.overall_faithful), "sensor": f["sensor"], "z": f["z"],
                             "wording": f["wording"], "lexicon_hits": f["lexicon_hits"]})
    rdf = pd.DataFrame(responses)
    rdf.to_csv(OUT / "part5_response_flags.csv", index=False)
    groups = []
    for label, g in [("all", rdf), *[(str(v), rdf[rdf.strict_scorer_verdict.astype(str).str.lower() == str(v).lower()]) for v in sorted(rdf.strict_scorer_verdict.astype(str).unique())]]:
        k = int(g.overstated.sum()); n = len(g); lo, hi = wilson(k, n)
        groups.append({"strict_scorer_verdict": label, "n_responses": n, "overstated_n": k,
                       "overstated_share_pct": 100*k/n if n else np.nan,
                       "wilson95_lower_pct": lo*100, "wilson95_upper_pct": hi*100})
    sdf = pd.DataFrame(groups)
    sdf.to_csv(OUT / "part5_exaggeration_summary.csv", index=False)
    exdf = pd.DataFrame(examples)
    if len(exdf):
        exdf = exdf.drop_duplicates(subset=["audit_id", "sensor", "wording"]).sort_values(["audit_id", "sensor"]).head(10)
    else:
        exdf = pd.DataFrame(columns=["audit_id", "segment_global_id", "machine", "strict_scorer_verdict", "sensor", "z", "wording", "lexicon_hits"])
    exdf.to_csv(OUT / "part5_examples.csv", index=False)
    import inspect
    (OUT / "part5_lexicon_and_function.txt").write_text(
        "LEXICON (case-insensitive): " + ", ".join(lexicon) + "\n\n"
        "MAGNITUDE REGEXES:\n"
        + r"Sensors?[_\s]*(\d+)" + "\n"
        + r"(?<![\w.])([+\-−]?\d+(?:\.\d+)?|±\s*\d+(?:\.\d+)?)\s*(?:std|σ)\b" + "\n\n"
        "FUNCTION CODE:\n" + inspect.getsource(rule_lexicon_assessment), encoding="utf-8")
    print("PART 5 RESULTS\n" + sdf.to_string(index=False), flush=True)
    print("PART 5 EXAMPLES\n" + (exdf.to_string(index=False) if len(exdf) else "No flagged examples."), flush=True)
    log_step("Part 5 — fixed-lexicon exaggeration scan", [gate, f"Lexicon: {', '.join(lexicon)}.", "Wrote part5_response_flags.csv, part5_exaggeration_summary.csv, part5_examples.csv, and part5_lexicon_and_function.txt.", sdf.to_string(index=False), "Examples:", exdf.to_string(index=False) if len(exdf) else "No flagged examples."])


def main() -> int:
    ensure_common_gate()
    cache = read_cache()
    base, attr = aggregate_attributions(cache)
    print(f"Loaded {len(cache)} cached segments, {base.machine.nunique()} machines.", flush=True)
    parts = [("Part 1", lambda: part1(cache, base)),
             ("Part 2", lambda: part2(cache, base)),
             ("Part 3", lambda: part3(cache)),
             ("Part 4", lambda: part4(base, attr)),
             ("Part 5", part5)]
    requested = None
    if len(sys.argv) > 1:
        arg = sys.argv[1]
        if arg.startswith("--parts="):
            requested = {x.strip().lower() for x in arg.split("=", 1)[1].split(",") if x.strip()}
        else:
            raise RuntimeError("usage: improve_analysis.py [--parts=1,2,3,4,5]")
    if requested is not None:
        parts = [(name, fn) for name, fn in parts if name.split()[-1].lower() in requested]
    failures = []
    for name, fn in parts:
        log_step(name + " — started", ["Common gate PASSED; part-specific gate will run before this part's calculations."])
        try:
            fn()
        except Exception as exc:
            message = f"{name} FAILED: {type(exc).__name__}: {exc}"
            failures.append(message)
            print(message, flush=True)
            log_step(name + " — failed", [message, traceback.format_exc().strip()])
    if failures:
        print("ANALYSIS COMPLETED WITH PART FAILURES:\n" + "\n".join(failures), flush=True)
        return 1
    print("ALL PARTS COMPLETED", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
