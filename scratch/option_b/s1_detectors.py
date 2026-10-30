"""Stage S1: Per-Machine Anomaly Detectors with Purge Gap, Multi-Seed, Leakage Checks, and Aggregates."""

import os
import sys
import glob
import json
import time
import hashlib
import datetime
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath("."))
from src.data_loader import (
    load_smd_raw, normalize_data, create_sliding_windows,
    compute_sensor_correlation_graph, build_pyg_dataset,
    compute_node_features_6d, PyGDataFallback
)
from src.gnn_attribution.model import GATAnomalyDetector
from src.gnn_attribution.train import (
    train_gnn_model, evaluate_model, find_optimal_threshold,
    compute_classification_metrics
)

def get_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def extract_events_from_labels(labels: np.ndarray) -> List[Tuple[int, int]]:
    """Extract maximal consecutive runs of 1s as (start_idx, end_idx) inclusive."""
    events = []
    in_event = False
    start = 0
    for idx, val in enumerate(labels):
        if val > 0:
            if not in_event:
                start = idx
                in_event = True
        else:
            if in_event:
                events.append((start, idx - 1))
                in_event = False
    if in_event:
        events.append((start, len(labels) - 1))
    return events

def compute_event_metrics(
    y_test_windows: np.ndarray,
    y_pred_windows: np.ndarray,
    test_window_start_timesteps: np.ndarray,
    window_size: int = 100
) -> Dict[str, Any]:
    """Compute event-level detection metrics based on window predictions.
    
    An event is detected if at least one predicted-positive test window overlaps it.
    False alarm events are maximal runs of predicted-positive windows that overlap no anomaly.
    """
    n_windows = len(y_test_windows)
    if n_windows == 0:
        return {
            "n_events_test": 0,
            "events_detected": 0,
            "event_recall": float("nan"),
            "false_alarm_events": 0
        }
    
    # Ground truth events in test window series
    gt_events = extract_events_from_labels(y_test_windows)
    n_events_test = len(gt_events)
    
    # Event detection check
    events_detected = 0
    for (e_start, e_end) in gt_events:
        # Check if any window in [e_start, e_end] is predicted positive
        if np.any(y_pred_windows[e_start : e_end + 1] == 1):
            events_detected += 1
            
    event_recall = float(events_detected / n_events_test) if n_events_test > 0 else float("nan")
    
    # False alarm events: maximal runs of predicted-positive windows
    pred_runs = extract_events_from_labels(y_pred_windows)
    false_alarm_events = 0
    for (p_start, p_end) in pred_runs:
        # Does this run overlap any gt_event?
        overlap = False
        for (e_start, e_end) in gt_events:
            if not (p_end < e_start or p_start > e_end):
                overlap = True
                break
        if not overlap:
            false_alarm_events += 1
            
    return {
        "n_events_test": n_events_test,
        "events_detected": events_detected,
        "event_recall": event_recall,
        "false_alarm_events": false_alarm_events
    }

def run_single_detector_task(args: Dict[str, Any]) -> Dict[str, Any]:
    """Worker function to train and evaluate a single detector on a machine and seed."""
    machine = args["machine"]
    seed = args["seed"]
    split_mode = args.get("split_mode", "purged") # 'purged' or 'legacy'
    ablation_2d = args.get("ablation_2d", False)
    
    suffix = f"_{split_mode}" if split_mode == "legacy" else ""
    ablation_suffix = "_2d" if ablation_2d else ""
    run_id = f"{machine}_seed{seed}{suffix}{ablation_suffix}"
    
    metrics_path = f"results_b/metrics/{run_id}.json"
    ckpt_path = f"results_b/checkpoints/{run_id}.pt"
    
    # Rule 6: Resumability check
    if os.path.exists(metrics_path) and os.path.exists(ckpt_path):
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                saved_metrics = json.load(f)
            current_hash = get_file_sha256(ckpt_path)
            if saved_metrics.get("checkpoint_sha256") == current_hash:
                print(f"[{run_id}] Resumed from existing metrics ({metrics_path}).")
                return saved_metrics
        except Exception:
            pass # re-run if corrupt
            
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    
    t_start = time.time()
    
    # 1. Load data
    train_raw, test_raw, test_labels_raw, interp_labels = load_smd_raw("data/raw", machine)
    train_norm, test_norm, norm_stats = normalize_data(train_raw, test_raw)
    
    # 2. Graph from machine's own train block (|r| >= 0.6)
    edge_index, edge_attr = compute_sensor_correlation_graph(train_norm, threshold=0.6)
    
    # 3. Sliding windows (T=100, stride=1) - memory efficient
    total_windows = len(test_norm) - 100 + 1
    test_win_labels = np.array([
        1 if np.any(test_labels_raw[i : i + 100] > 0) else 0
        for i in range(total_windows)
    ], dtype=np.int64)
    
    train_end = int(total_windows * 0.70)
    val_end = train_end + int(total_windows * 0.15)
    
    # 4. Leakage check and split slicing
    if split_mode == "purged":
        # Purge last 99 windows of train and val
        train_slice = slice(0, train_end - 99)
        val_slice = slice(train_end, val_end - 99)
        test_slice = slice(val_end, total_windows)
        
        last_train_w_end = (train_end - 100) + 99
        first_val_w_start = train_end
        last_val_w_end = (val_end - 100) + 99
        first_test_w_start = val_end
        
        overlap_train_val = max(0, last_train_w_end - first_val_w_start + 1)
        overlap_val_test = max(0, last_val_w_end - first_test_w_start + 1)
        leak_pass = (overlap_train_val == 0) and (overlap_val_test == 0)
    else:
        # Legacy unpurged split
        train_slice = slice(0, train_end)
        val_slice = slice(train_end, val_end)
        test_slice = slice(val_end, total_windows)
        
        last_train_w_end = (train_end - 1) + 99
        first_val_w_start = train_end
        last_val_w_end = (val_end - 1) + 99
        first_test_w_start = val_end
        
        overlap_train_val = max(0, last_train_w_end - first_val_w_start + 1)
        overlap_val_test = max(0, last_val_w_end - first_test_w_start + 1)
        leak_pass = False # Legacy split intentionally leaks 99 timesteps
        
    leakage_info = {
        "machine": machine,
        "seed": seed,
        "split_mode": split_mode,
        "total_windows": total_windows,
        "last_train_window_end": int(last_train_w_end),
        "first_val_window_start": int(first_val_w_start),
        "overlap_train_val": int(overlap_train_val),
        "last_val_window_end": int(last_val_w_end),
        "first_test_window_start": int(first_test_w_start),
        "overlap_val_test": int(overlap_val_test),
        "status": "pass" if leak_pass else ("leak_legacy" if split_mode == "legacy" else "leak_fail")
    }
    
    # 5. Build features in memory-efficient 1,000-window chunks (prevents RAM spikes)
    chunk_size = 1000
    feat_chunks = []
    for s in range(0, total_windows, chunk_size):
        e = min(s + chunk_size, total_windows)
        s_data = test_norm[s : e + 99]
        w_chunk = np.array([s_data[i : i + 100] for i in range(e - s)], dtype=np.float32)
        if ablation_2d:
            m = np.mean(w_chunk, axis=1)
            sd = np.std(w_chunk, axis=1)
            feat_chunks.append(np.stack([m, sd], axis=-1).astype(np.float32))
        else:
            feat_chunks.append(compute_node_features_6d(w_chunk))
        del w_chunk, s_data
        
    node_feats = np.concatenate(feat_chunks, axis=0)
    del feat_chunks
    in_dim = 2 if ablation_2d else 6
        
    ei_ten = torch.tensor(edge_index, dtype=torch.long)
    ea_ten = torch.tensor(edge_attr, dtype=torch.float32) if edge_attr is not None else None
    
    class FastPyGDataset(torch.utils.data.Dataset):
        def __init__(self, x_arr, y_arr, ei, ea):
            self.x = torch.from_numpy(x_arr).float()
            self.y = torch.from_numpy(y_arr).long()
            self.ei = ei
            self.ea = ea
        def __len__(self):
            return len(self.y)
        def __getitem__(self, idx):
            from torch_geometric.data import Data as PyGData
            return PyGData(x=self.x[idx], edge_index=self.ei, edge_attr=self.ea, y=self.y[idx : idx + 1])
            
    train_graphs = FastPyGDataset(node_feats[train_slice], test_win_labels[train_slice], ei_ten, ea_ten)
    val_graphs = FastPyGDataset(node_feats[val_slice], test_win_labels[val_slice], ei_ten, ea_ten)
    test_graphs = FastPyGDataset(node_feats[test_slice], test_win_labels[test_slice], ei_ten, ea_ten)
    del node_feats
    import gc
    gc.collect()
    
    train_labels = test_win_labels[train_slice]
    val_labels = test_win_labels[val_slice]
    test_labels = test_win_labels[test_slice]
    
    is_degenerate = (np.sum(val_labels) == 0) or (np.sum(test_labels) == 0)
    
    # 6. Model definition & training
    model = GATAnomalyDetector(
        in_dim=in_dim,
        hidden_dim=64,
        num_heads=4,
        edge_dim=1,
        dropout=0.2
    )
    
    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False)
    
    cfg = {
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs": 20,
        "patience": 10,
        "device": "cpu"
    }
    
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    if os.path.exists(ckpt_path) and os.path.getsize(ckpt_path) > 0:
        try:
            ckpt_data = torch.load(ckpt_path, map_location="cpu")
            if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data:
                model.load_state_dict(ckpt_data["model_state_dict"])
            else:
                model.load_state_dict(ckpt_data)
            trained_model = model
            print(f"[{run_id}] Reusing existing checkpoint from {ckpt_path}.")
        except Exception as e:
            print(f"[{run_id}] Could not load checkpoint ({e}), retraining...")
            trained_model = train_gnn_model(model, train_loader, val_loader, cfg, checkpoint_path=ckpt_path)
    else:
        trained_model = train_gnn_model(model, train_loader, val_loader, cfg, checkpoint_path=ckpt_path)
    ckpt_hash = get_file_sha256(ckpt_path)
    
    # 7. Threshold search on val split
    criterion = nn.BCEWithLogitsLoss()
    _, _, val_true, val_prob = evaluate_model(trained_model, val_loader, criterion, torch.device("cpu"), threshold=0.5)
    
    if np.sum(val_true) > 0:
        opt_thresh, opt_val_m = find_optimal_threshold(val_true, val_prob)
    else:
        opt_thresh = 0.50
        opt_val_m = compute_classification_metrics(val_true, val_prob, threshold=0.50)
        
    # 8. Evaluate test set at val-optimal threshold
    test_loss, test_m, y_true, y_prob = evaluate_model(
        trained_model, test_loader, criterion, torch.device("cpu"), threshold=opt_thresh
    )
    
    # AUPRC on test set
    if np.sum(y_true) > 0:
        auprc = float(average_precision_score(y_true, y_prob))
    else:
        auprc = float("nan")
    test_m["auprc"] = auprc
    
    # 9. Oracle best-F1 on test set
    if np.sum(y_true) > 0:
        oracle_thresh, oracle_m = find_optimal_threshold(y_true, y_prob)
        oracle_m["auprc"] = auprc
    else:
        oracle_thresh = 0.50
        oracle_m = compute_classification_metrics(y_true, y_prob, threshold=0.50)
        oracle_m["auprc"] = float("nan")
        
    # 10. Event-level metrics
    y_pred_test = (y_prob >= opt_thresh).astype(int)
    test_win_indices = np.arange(test_slice.start, test_slice.stop)
    event_m = compute_event_metrics(y_true, y_pred_test, test_win_indices, window_size=100)
    
    # Oracle event-level metrics
    y_pred_oracle = (y_prob >= oracle_thresh).astype(int)
    oracle_event_m = compute_event_metrics(y_true, y_pred_oracle, test_win_indices, window_size=100)
    
    wall_clock = time.time() - t_start
    
    result = {
        "machine": machine,
        "seed": seed,
        "split_mode": split_mode,
        "ablation_2d": ablation_2d,
        "run_id": run_id,
        "is_degenerate": bool(is_degenerate),
        "wall_clock_seconds": round(wall_clock, 2),
        "checkpoint_path": ckpt_path,
        "checkpoint_sha256": ckpt_hash,
        "n_train_windows": len(train_graphs),
        "n_val_windows": len(val_graphs),
        "n_test_windows": len(test_graphs),
        "val_anomaly_ratio": float(np.mean(val_labels)),
        "test_anomaly_ratio": float(np.mean(test_labels)),
        "val_optimal_threshold": float(opt_thresh),
        "val_metrics_at_optimal_threshold": opt_val_m,
        "test_metrics": {
            "precision": float(test_m["precision"]),
            "recall": float(test_m["recall"]),
            "f1": float(test_m["f1"]),
            "auroc": float(test_m["auroc"]),
            "auprc": float(auprc),
            "tp": int(test_m["tp"]),
            "fp": int(test_m["fp"]),
            "fn": int(test_m["fn"]),
            "tn": int(test_m["tn"]),
            "threshold": float(opt_thresh),
            **event_m
        },
        "oracle_test_metrics": {
            "precision": float(oracle_m["precision"]),
            "recall": float(oracle_m["recall"]),
            "f1": float(oracle_m["f1"]),
            "auroc": float(oracle_m["auroc"]),
            "auprc": float(auprc),
            "tp": int(oracle_m["tp"]),
            "fp": int(oracle_m["fp"]),
            "fn": int(oracle_m["fn"]),
            "tn": int(oracle_m["tn"]),
            "threshold": float(oracle_thresh),
            **oracle_event_m
        },
        "leakage_info": leakage_info
    }
    
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)
        
    print(f"[{run_id}] Finished in {wall_clock:.1f}s | Val-Opt Thresh={opt_thresh:.2f} | Test F1={test_m['f1']:.4f} | AUROC={test_m['auroc']:.4f} | AUPRC={auprc:.4f} | Deconstruct={is_degenerate}")
    return result

def cluster_bootstrap_ci(
    per_machine_metric_dict: Dict[str, float],
    n_resamples: int = 10000,
    seed: int = 42
) -> Tuple[float, float, float, float]:
    """Cluster bootstrap over machines to calculate mean, std, and 95% CI."""
    machines = list(per_machine_metric_dict.keys())
    values = np.array([per_machine_metric_dict[m] for m in machines if not np.isnan(per_machine_metric_dict[m])])
    
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
        
    rng = np.random.RandomState(seed)
    n_m = len(values)
    bootstrap_means = []
    
    for _ in range(n_resamples):
        idx = rng.choice(n_m, size=n_m, replace=True)
        bootstrap_means.append(np.mean(values[idx]))
        
    bootstrap_means = np.array(bootstrap_means)
    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1)) if n_m > 1 else 0.0
    ci_lower = float(np.percentile(bootstrap_means, 2.5))
    ci_upper = float(np.percentile(bootstrap_means, 97.5))
    
    return mean_val, std_val, ci_lower, ci_upper

def build_detection_summary(all_results: List[Dict[str, Any]]):
    """Aggregate per-machine metrics into detection_summary.csv and related summary tables."""
    print("\n" + "=" * 70)
    print("STAGE S1.7 — Detection Summary & Aggregates")
    print("=" * 70)
    
    # 1. Leakage checks table
    leakage_rows = [r["leakage_info"] for r in all_results if "leakage_info" in r]
    leak_df = pd.DataFrame(leakage_rows).drop_duplicates(subset=["machine", "split_mode"])
    leak_csv_path = "results_b/leakage_checks.csv"
    leak_df.to_csv(leak_csv_path, index=False)
    print(f"Saved leakage checks to {leak_csv_path} (Total checks: {len(leak_df)})")
    
    # Check for any unexpected leakage
    failed_leaks = leak_df[(leak_df["split_mode"] == "purged") & (leak_df["status"] != "pass")]
    if len(failed_leaks) > 0:
        print(f"WARNING: Found {len(failed_leaks)} machines with leakage failures under purged split:")
        print(failed_leaks[["machine", "status", "overlap_train_val", "overlap_val_test"]])
    else:
        print("Leakage verification PASSED: 0 overlap timesteps across all machines under purged split.")
        
    # 2. Build rows for detection table
    table_rows = []
    for r in all_results:
        m = r["machine"]
        s = r["seed"]
        mode = r["split_mode"]
        abl = "2D" if r.get("ablation_2d") else "6D"
        tm = r["test_metrics"]
        om = r["oracle_test_metrics"]
        
        table_rows.append({
            "run_id": r["run_id"],
            "machine": m,
            "seed": s,
            "feature_dim": abl,
            "split_mode": mode,
            "is_degenerate": r["is_degenerate"],
            "wall_clock_s": r["wall_clock_seconds"],
            "val_opt_threshold": tm["threshold"],
            "precision": tm["precision"],
            "recall": tm["recall"],
            "f1": tm["f1"],
            "auroc": tm["auroc"],
            "auprc": tm["auprc"],
            "tp": tm["tp"],
            "fp": tm["fp"],
            "fn": tm["fn"],
            "tn": tm["tn"],
            "n_events_test": tm["n_events_test"],
            "events_detected": tm["events_detected"],
            "event_recall": tm["event_recall"],
            "false_alarm_events": tm["false_alarm_events"],
            "oracle_threshold": om["threshold"],
            "oracle_precision": om["precision"],
            "oracle_recall": om["recall"],
            "oracle_f1": om["f1"],
            "oracle_tp": om["tp"],
            "oracle_fp": om["fp"],
            "oracle_fn": om["fn"],
            "oracle_tn": om["tn"],
            "oracle_events_detected": om["events_detected"],
            "oracle_event_recall": om["event_recall"],
            "oracle_false_alarm_events": om["false_alarm_events"]
        })
        
    det_df = pd.DataFrame(table_rows)
    det_csv_path = "results_b/detection_per_machine.csv"
    det_df.to_csv(det_csv_path, index=False)
    print(f"Saved per-machine detection metrics to {det_csv_path}")
    
    # 3. Macro and Micro Summaries across non-degenerate machines for seed 42 (and multi-seed)
    summary_records = []
    
    # Filter groups
    configs_to_summarize = [
        ("6D_GAT_Purged_Seed42", det_df[(det_df["feature_dim"] == "6D") & (det_df["split_mode"] == "purged") & (det_df["seed"] == 42)]),
        ("6D_GAT_Purged_MultiSeed", det_df[(det_df["feature_dim"] == "6D") & (det_df["split_mode"] == "purged")]),
        ("2D_GAT_Purged_Seed42", det_df[(det_df["feature_dim"] == "2D") & (det_df["split_mode"] == "purged") & (det_df["seed"] == 42)]),
        ("6D_GAT_Legacy_m11", det_df[(det_df["machine"] == "machine-1-1") & (det_df["split_mode"] == "legacy") & (det_df["seed"] == 42)])
    ]
    
    for name, group in configs_to_summarize:
        if len(group) == 0:
            continue
            
        non_deg = group[~group["is_degenerate"]]
        n_total = len(group["machine"].unique())
        n_non_deg = len(non_deg["machine"].unique())
        deg_machines = list(group[group["is_degenerate"]]["machine"].unique())
        
        # Macro averages on non-degenerate (group by machine to handle multi-seed cleanly)
        m_grouped = non_deg.groupby("machine")[["f1", "precision", "recall", "auroc", "auprc", "event_recall", "oracle_f1"]].mean()
        macro_f1, macro_f1_sd, f1_ci_l, f1_ci_u = cluster_bootstrap_ci(m_grouped["f1"].to_dict())
        macro_p, macro_p_sd, p_ci_l, p_ci_u = cluster_bootstrap_ci(m_grouped["precision"].to_dict())
        macro_r, macro_r_sd, r_ci_l, r_ci_u = cluster_bootstrap_ci(m_grouped["recall"].to_dict())
        macro_auc, macro_auc_sd, auc_ci_l, auc_ci_u = cluster_bootstrap_ci(m_grouped["auroc"].to_dict())
        macro_auprc, macro_auprc_sd, auprc_ci_l, auprc_ci_u = cluster_bootstrap_ci(m_grouped["auprc"].to_dict())
        macro_erec, macro_erec_sd, erec_ci_l, erec_ci_u = cluster_bootstrap_ci(m_grouped["event_recall"].to_dict())
        
        # Oracle macro F1
        oracle_f1, oracle_f1_sd, of1_ci_l, of1_ci_u = cluster_bootstrap_ci(m_grouped["oracle_f1"].to_dict())
        
        # Micro metrics (summed confusion matrix across non-degenerate)
        sum_tp = int(non_deg["tp"].sum())
        sum_fp = int(non_deg["fp"].sum())
        sum_fn = int(non_deg["fn"].sum())
        sum_tn = int(non_deg["tn"].sum())
        
        micro_p = float(sum_tp / (sum_tp + sum_fp)) if (sum_tp + sum_fp) > 0 else 0.0
        micro_r = float(sum_tp / (sum_tp + sum_fn)) if (sum_tp + sum_fn) > 0 else 0.0
        micro_f1 = float(2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0
        
        summary_records.append({
            "model_config": name,
            "n_machines_total": n_total,
            "n_machines_non_degenerate": n_non_deg,
            "degenerate_machines": ", ".join(deg_machines) if deg_machines else "none",
            "macro_f1_mean": macro_f1,
            "macro_f1_sd": macro_f1_sd,
            "macro_f1_95ci": f"[{f1_ci_l:.4f}, {f1_ci_u:.4f}]",
            "macro_precision_mean": macro_p,
            "macro_precision_sd": macro_p_sd,
            "macro_recall_mean": macro_r,
            "macro_recall_sd": macro_r_sd,
            "macro_auroc_mean": macro_auc,
            "macro_auroc_sd": macro_auc_sd,
            "macro_auprc_mean": macro_auprc,
            "macro_auprc_sd": macro_auprc_sd,
            "macro_event_recall_mean": macro_erec,
            "macro_event_recall_sd": macro_erec_sd,
            "oracle_macro_f1_mean": oracle_f1,
            "oracle_macro_f1_sd": oracle_f1_sd,
            "micro_precision": micro_p,
            "micro_recall": micro_r,
            "micro_f1": micro_f1,
            "sum_tp": sum_tp,
            "sum_fp": sum_fp,
            "sum_fn": sum_fn,
            "sum_tn": sum_tn
        })
        
    summary_df = pd.DataFrame(summary_records)
    summary_csv_path = "results_b/detection_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Saved detection summary to {summary_csv_path}")
    
    # Print clean table
    print("\n--- DETECTION BENCHMARK SUMMARY ---")
    for _, row in summary_df.iterrows():
        print(f"\nConfiguration: {row['model_config']} (N={row['n_machines_non_degenerate']}/{row['n_machines_total']})")
        print(f"  Macro F1:           {row['macro_f1_mean']*100:.2f}% +/- {row['macro_f1_sd']*100:.2f}%  95% CI: {row['macro_f1_95ci']}")
        print(f"  Macro Precision:    {row['macro_precision_mean']*100:.2f}% +/- {row['macro_precision_sd']*100:.2f}%")
        print(f"  Macro Recall:       {row['macro_recall_mean']*100:.2f}% +/- {row['macro_recall_sd']*100:.2f}%")
        print(f"  Macro AUROC:        {row['macro_auroc_mean']*100:.2f}% +/- {row['macro_auroc_sd']*100:.2f}%")
        print(f"  Macro AUPRC:        {row['macro_auprc_mean']*100:.2f}% +/- {row['macro_auprc_sd']*100:.2f}%")
        print(f"  Macro Event Recall: {row['macro_event_recall_mean']*100:.2f}%")
        print(f"  Oracle Macro F1:    {row['oracle_macro_f1_mean']*100:.2f}%")
        print(f"  Micro F1:           {row['micro_f1']*100:.2f}% (TP={row['sum_tp']}, FP={row['sum_fp']}, FN={row['sum_fn']}, TN={row['sum_tn']})")
        if row['degenerate_machines'] != "none":
            print(f"  Excluded Degenerate Machines: {row['degenerate_machines']}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--machine", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_mode", type=str, default="purged")
    parser.add_argument("--ablation_2d", action="store_true")
    args = parser.parse_args()
    
    if args.machine is not None:
        run_single_detector_task({
            "machine": args.machine,
            "seed": args.seed,
            "split_mode": args.split_mode,
            "ablation_2d": args.ablation_2d
        })
