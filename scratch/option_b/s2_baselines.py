"""Stage S2: Same-Split Baselines (Isolation Forest & Max-|z-score|) on Identical Purged Splits."""

import os
import sys
import json
import time
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

sys.path.insert(0, os.path.abspath("."))
from src.data_loader import (
    load_smd_raw, normalize_data, create_sliding_windows, compute_node_features_6d
)
from src.gnn_attribution.train import compute_classification_metrics, find_optimal_threshold
from scratch.option_b.s1_detectors import compute_event_metrics, cluster_bootstrap_ci

def run_single_machine_baselines(machine: str, seed: int = 42) -> List[Dict[str, Any]]:
    """Run Isolation Forest and Max-|z-score| on a single machine under purged split."""
    # 1. Load data
    train_raw, test_raw, test_labels_raw, interp_labels = load_smd_raw("data/raw", machine)
    train_norm, test_norm, norm_stats = normalize_data(train_raw, test_raw)
    
    # 2. Sliding windows
    # 2. Sliding windows - memory efficient
    total_windows = len(test_norm) - 100 + 1
    test_win_labels = np.array([
        1 if np.any(test_labels_raw[i : i + 100] > 0) else 0
        for i in range(total_windows)
    ], dtype=np.int64)
    
    train_end = int(total_windows * 0.70)
    val_end = train_end + int(total_windows * 0.15)
    
    # 3. Purged split
    train_slice = slice(0, train_end - 99)
    val_slice = slice(train_end, val_end - 99)
    test_slice = slice(val_end, total_windows)
    
    y_train = test_win_labels[train_slice]
    y_val = test_win_labels[val_slice]
    y_test = test_win_labels[test_slice]
    
    is_degenerate = (np.sum(y_val) == 0) or (np.sum(y_test) == 0)
    
    # Precompute features in chunks
    chunk_size = 1000
    feat_list = []
    max_z_list = []
    for s in range(0, total_windows, chunk_size):
        e = min(s + chunk_size, total_windows)
        s_data = test_norm[s : e + 99]
        w_chunk = np.array([s_data[i : i + 100] for i in range(e - s)], dtype=np.float32)
        feat_list.append(compute_node_features_6d(w_chunk))
        max_z_list.append(np.max(np.abs(w_chunk), axis=(1, 2)))
        del w_chunk, s_data
        
    node_feats_6d = np.concatenate(feat_list, axis=0) # [total_windows, 38, 6]
    flat_feats = node_feats_6d.reshape(total_windows, -1) # [total_windows, 228]
    max_z_all = np.concatenate(max_z_list, axis=0)
    del feat_list, max_z_list, node_feats_6d
    
    X_train_flat = flat_feats[train_slice]
    X_val_flat = flat_feats[val_slice]
    X_test_flat = flat_feats[test_slice]
    
    max_z_val = max_z_all[val_slice]
    max_z_test = max_z_all[test_slice]
    
    results = []
    test_win_indices = np.arange(test_slice.start, test_slice.stop)
    
    # -------------------------------------------------------------
    # Baseline 1: Isolation Forest
    # -------------------------------------------------------------
    iso_forest = IsolationForest(n_estimators=100, random_state=seed, n_jobs=1)
    iso_forest.fit(X_train_flat)
    
    # Anomaly score: -score_samples (higher = more anomalous)
    val_scores_raw = -iso_forest.score_samples(X_val_flat)
    test_scores_raw = -iso_forest.score_samples(X_test_flat)
    
    # Normalize scores to [0, 1] range for threshold search
    min_s = min(np.min(val_scores_raw), np.min(test_scores_raw))
    max_s = max(np.max(val_scores_raw), np.max(test_scores_raw))
    val_scores_iso = (val_scores_raw - min_s) / (max_s - min_s + 1e-8)
    test_scores_iso = (test_scores_raw - min_s) / (max_s - min_s + 1e-8)
    
    # Find threshold maximizing F1 on val
    candidate_thresholds = np.quantile(val_scores_iso, np.linspace(0.01, 0.99, 99))
    opt_t_iso = 0.5
    best_f1_iso = -1.0
    for t in candidate_thresholds:
        m = compute_classification_metrics(y_val, val_scores_iso, threshold=float(t))
        if m["f1"] > best_f1_iso:
            best_f1_iso = m["f1"]
            opt_t_iso = float(t)
            
    m_test_iso = compute_classification_metrics(y_test, test_scores_iso, threshold=opt_t_iso)
    auprc_iso = float(average_precision_score(y_test, test_scores_iso)) if np.sum(y_test) > 0 else float("nan")
    m_test_iso["auprc"] = auprc_iso
    
    y_pred_iso = (test_scores_iso >= opt_t_iso).astype(int)
    event_m_iso = compute_event_metrics(y_test, y_pred_iso, test_win_indices, window_size=100)
    
    # Oracle test F1
    oracle_t_iso = 0.5
    best_of1_iso = -1.0
    for t in np.quantile(test_scores_iso, np.linspace(0.01, 0.99, 99)):
        m = compute_classification_metrics(y_test, test_scores_iso, threshold=float(t))
        if m["f1"] > best_of1_iso:
            best_of1_iso = m["f1"]
            oracle_t_iso = float(t)
    oracle_m_iso = compute_classification_metrics(y_test, test_scores_iso, threshold=oracle_t_iso)
    oracle_m_iso["auprc"] = auprc_iso
    oracle_event_iso = compute_event_metrics(y_test, (test_scores_iso >= oracle_t_iso).astype(int), test_win_indices)
    
    results.append({
        "baseline_name": "Isolation_Forest",
        "machine": machine,
        "seed": seed,
        "is_degenerate": is_degenerate,
        "val_opt_threshold": opt_t_iso,
        "precision": m_test_iso["precision"],
        "recall": m_test_iso["recall"],
        "f1": m_test_iso["f1"],
        "auroc": m_test_iso["auroc"],
        "auprc": auprc_iso,
        "tp": m_test_iso["tp"],
        "fp": m_test_iso["fp"],
        "fn": m_test_iso["fn"],
        "tn": m_test_iso["tn"],
        "event_recall": event_m_iso["event_recall"],
        "false_alarm_events": event_m_iso["false_alarm_events"],
        "oracle_f1": oracle_m_iso["f1"],
        "oracle_precision": oracle_m_iso["precision"],
        "oracle_recall": oracle_m_iso["recall"]
    })
    
    # -------------------------------------------------------------
    # Baseline 2: Max-|z-score|
    # -------------------------------------------------------------
    # Normalize max_z to [0, 1] range for thresholding
    min_z = min(np.min(max_z_val), np.min(max_z_test))
    max_z = max(np.max(max_z_val), np.max(max_z_test))
    val_scores_z = (max_z_val - min_z) / (max_z - min_z + 1e-8)
    test_scores_z = (max_z_test - min_z) / (max_z - min_z + 1e-8)
    
    candidate_thresholds_z = np.quantile(val_scores_z, np.linspace(0.01, 0.99, 99))
    opt_t_z = 0.5
    best_f1_z = -1.0
    for t in candidate_thresholds_z:
        m = compute_classification_metrics(y_val, val_scores_z, threshold=float(t))
        if m["f1"] > best_f1_z:
            best_f1_z = m["f1"]
            opt_t_z = float(t)
            
    m_test_z = compute_classification_metrics(y_test, test_scores_z, threshold=opt_t_z)
    auprc_z = float(average_precision_score(y_test, test_scores_z)) if np.sum(y_test) > 0 else float("nan")
    m_test_z["auprc"] = auprc_z
    
    y_pred_z = (test_scores_z >= opt_t_z).astype(int)
    event_m_z = compute_event_metrics(y_test, y_pred_z, test_win_indices, window_size=100)
    
    # Oracle test F1
    oracle_t_z = 0.5
    best_of1_z = -1.0
    for t in np.quantile(test_scores_z, np.linspace(0.01, 0.99, 99)):
        m = compute_classification_metrics(y_test, test_scores_z, threshold=float(t))
        if m["f1"] > best_of1_z:
            best_of1_z = m["f1"]
            oracle_t_z = float(t)
    oracle_m_z = compute_classification_metrics(y_test, test_scores_z, threshold=oracle_t_z)
    oracle_m_z["auprc"] = auprc_z
    oracle_event_z = compute_event_metrics(y_test, (test_scores_z >= oracle_t_z).astype(int), test_win_indices)
    
    results.append({
        "baseline_name": "Max_Z_Score",
        "machine": machine,
        "seed": seed,
        "is_degenerate": is_degenerate,
        "val_opt_threshold": opt_t_z,
        "precision": m_test_z["precision"],
        "recall": m_test_z["recall"],
        "f1": m_test_z["f1"],
        "auroc": m_test_z["auroc"],
        "auprc": auprc_z,
        "tp": m_test_z["tp"],
        "fp": m_test_z["fp"],
        "fn": m_test_z["fn"],
        "tn": m_test_z["tn"],
        "event_recall": event_m_z["event_recall"],
        "false_alarm_events": event_m_z["false_alarm_events"],
        "oracle_f1": oracle_m_z["f1"],
        "oracle_precision": oracle_m_z["precision"],
        "oracle_recall": oracle_m_z["recall"]
    })
    
    return results

def run_all_baselines():
    print("=" * 70)
    print("STAGE S2 — Same-Split Baselines (Isolation Forest & Max-|z-score|)")
    print("=" * 70)
    
    t0 = time.time()
    
    machines = []
    for i in range(1, 9):
        machines.append(f"machine-1-{i}")
    for i in range(1, 10):
        machines.append(f"machine-2-{i}")
    for i in range(1, 12):
        machines.append(f"machine-3-{i}")
        
    all_baseline_rows = []
    
    for idx, mid in enumerate(machines, 1):
        print(f"[{idx}/28] Running baselines on {mid} ...")
        rows = run_single_machine_baselines(mid, seed=42)
        all_baseline_rows.extend(rows)
        for r in rows:
            print(f"  -> {r['baseline_name']}: F1={r['f1']*100:.2f}%, AUROC={r['auroc']*100:.2f}%, AUPRC={r['auprc']*100:.2f}% (Oracle F1={r['oracle_f1']*100:.2f}%)")
            
    df_baselines = pd.DataFrame(all_baseline_rows)
    out_csv = "results_b/baselines.csv"
    df_baselines.to_csv(out_csv, index=False)
    print(f"\nSaved baselines to {out_csv}")
    
    # Compute macro and micro averages across non-degenerate machines
    summary_rows = []
    for b_name in ["Isolation_Forest", "Max_Z_Score"]:
        b_df = df_baselines[df_baselines["baseline_name"] == b_name]
        non_deg = b_df[~b_df["is_degenerate"]]
        
        f1_m, f1_sd, f1_l, f1_u = cluster_bootstrap_ci(dict(zip(non_deg["machine"], non_deg["f1"])))
        p_m, p_sd, _, _ = cluster_bootstrap_ci(dict(zip(non_deg["machine"], non_deg["precision"])))
        r_m, r_sd, _, _ = cluster_bootstrap_ci(dict(zip(non_deg["machine"], non_deg["recall"])))
        auc_m, auc_sd, _, _ = cluster_bootstrap_ci(dict(zip(non_deg["machine"], non_deg["auroc"])))
        auprc_m, auprc_sd, _, _ = cluster_bootstrap_ci(dict(zip(non_deg["machine"], non_deg["auprc"])))
        erec_m, erec_sd, _, _ = cluster_bootstrap_ci(dict(zip(non_deg["machine"], non_deg["event_recall"])))
        of1_m, of1_sd, _, _ = cluster_bootstrap_ci(dict(zip(non_deg["machine"], non_deg["oracle_f1"])))
        
        sum_tp = int(non_deg["tp"].sum())
        sum_fp = int(non_deg["fp"].sum())
        sum_fn = int(non_deg["fn"].sum())
        sum_tn = int(non_deg["tn"].sum())
        
        micro_p = float(sum_tp / (sum_tp + sum_fp)) if (sum_tp + sum_fp) > 0 else 0.0
        micro_r = float(sum_tp / (sum_tp + sum_fn)) if (sum_tp + sum_fn) > 0 else 0.0
        micro_f1 = float(2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0
        
        summary_rows.append({
            "baseline_name": b_name,
            "macro_f1": f1_m,
            "macro_f1_sd": f1_sd,
            "macro_f1_95ci": f"[{f1_l:.4f}, {f1_u:.4f}]",
            "macro_precision": p_m,
            "macro_recall": r_m,
            "macro_auroc": auc_m,
            "macro_auprc": auprc_m,
            "macro_event_recall": erec_m,
            "oracle_macro_f1": of1_m,
            "micro_f1": micro_f1,
            "micro_precision": micro_p,
            "micro_recall": micro_r
        })
        
    sum_df = pd.DataFrame(summary_rows)
    sum_csv = "results_b/baselines_summary.csv"
    sum_df.to_csv(sum_csv, index=False)
    print(f"Saved baseline summaries to {sum_csv}")
    
    elapsed = time.time() - t0
    print(f"Stage S2 Baselines completed in {elapsed:.2f}s ({elapsed/60.0:.2f} min)")
    
    with open("results_b/LOG.md", "a", encoding="utf-8") as f:
        f.write(f"\n## Stage S2 — Same-Split Baselines Completed\n")
        f.write(f"- Wall-clock: {elapsed:.2f}s\n")
        f.write(f"- Outputs: `results_b/baselines.csv`, `results_b/baselines_summary.csv`\n")

if __name__ == "__main__":
    run_all_baselines()
