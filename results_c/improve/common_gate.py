"""Read-only common reproduction gate for the improve analyses."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RANKINGS_PATH = ROOT / "results_b" / "s3_cached_rankings.json"
ATTR_PATH = ROOT / "results_b" / "attribution_per_segment.csv"


def rank_metrics(ranking: list[int], truth: set[int]) -> tuple[float, float, float]:
    k = len(truth)
    if not k:
        return float("nan"), float("nan"), float("nan")
    return (
        float(ranking[0] in truth),
        len(set(ranking[:5]) & truth) / k,
        len(set(ranking[:k]) & truth) / k,
    )


def max_abs_z_rank(entry: dict) -> list[int]:
    x = np.asarray(entry["fine_raw_window"], dtype=float)
    mu = np.asarray(entry["norm_stats_mean"], dtype=float)
    sd = np.asarray(entry["norm_stats_std"], dtype=float)
    sd_adj = np.where(sd < 1e-8, 1.0, sd)
    z = (x - mu) / sd_adj
    z[:, sd_adj == 1.0] = 0.0
    return list(np.argsort(-np.abs(z).max(axis=0), kind="stable"))


def main() -> int:
    raw = json.loads(RANKINGS_PATH.read_text(encoding="utf-8"))
    if len(raw) != 327:
        raise RuntimeError(f"common gate failed: cached segment count={len(raw)}, expected=327")

    attr = pd.read_csv(ATTR_PATH)
    ig = attr[attr["method"].astype(str).str.strip() == "Integrated_Gradients"].copy()
    if ig["segment_global_id"].nunique() != 327:
        raise RuntimeError(
            f"common gate failed: seed-42 IG unique segments={ig['segment_global_id'].nunique()}, expected=327"
        )
    ig_mm = ig.groupby("machine")[["top1_hit", "top5_recall", "legacy_overlap"]].mean().mean() * 100

    max_rows = []
    for entry in raw:
        truth = {int(s) - 1 for s in entry["gt_dimensions"]}
        max_rows.append((entry["machine"], *rank_metrics(max_abs_z_rank(entry), truth)))
    max_df = pd.DataFrame(max_rows, columns=["machine", "top1", "top5", "legacy"])
    max_mm = max_df.groupby("machine")[["top1", "top5", "legacy"]].mean().mean() * 100

    lines = [
        f"COMMON GATE segments={len(raw)} machines={ig['machine'].nunique()} IG_segments={ig['segment_global_id'].nunique()}",
        f"COMMON GATE IG machine-macro top1={ig_mm['top1_hit']:.2f} top5={ig_mm['top5_recall']:.2f} legacy={ig_mm['legacy_overlap']:.2f} (targets 83.89/53.93/62.84; tolerance 0.05)",
        f"COMMON GATE max|z| machine-macro top5={max_mm['top5']:.2f} (target 59.55)",
    ]
    passed = (
        all(abs(float(ig_mm[k]) - t) <= 0.05 for k, t in [("top1_hit", 83.89), ("top5_recall", 53.93), ("legacy_overlap", 62.84)])
        and abs(float(max_mm["top5"]) - 59.55) <= 0.01
        and ig["machine"].nunique() == 28
    )
    lines.append(f"COMMON GATE {'PASSED' if passed else 'FAILED'}")
    text = "\n".join(lines) + "\n"
    (OUT / "GATE_COMMON.txt").write_text(text, encoding="utf-8")
    print(text, end="")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
