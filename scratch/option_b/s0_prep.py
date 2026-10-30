"""Stage S0: Inventory, Config Freeze, Timing, and Gate G0 Plan Selection."""

import os
import sys
import glob
import json
import time
import hashlib
import datetime
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.abspath("."))
from src.data_loader import load_smd_raw, parse_interpretation_labels

def get_file_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def count_events(binary_labels: np.ndarray) -> int:
    """Count maximal runs of consecutive anomalous timesteps (y == 1)."""
    in_event = False
    count = 0
    for val in binary_labels:
        if val > 0:
            if not in_event:
                count += 1
                in_event = True
        else:
            in_event = False
    return count

def run_s0_inventory():
    print("=" * 70)
    print("STAGE S0.1 — Machine Inventory & Assertion")
    print("=" * 70)
    
    expected_machines = []
    for i in range(1, 9):
        expected_machines.append(f"machine-1-{i}")
    for i in range(1, 10):
        expected_machines.append(f"machine-2-{i}")
    for i in range(1, 12):
        expected_machines.append(f"machine-3-{i}")
    
    assert len(expected_machines) == 28, f"Expected 28 machines, got {len(expected_machines)}"
    print(f"Verified 28 machine IDs: {expected_machines[0]} .. {expected_machines[-1]}")
    
    raw_dir = "data/raw"
    interp_dir = os.path.join(raw_dir, "SMD", "interpretation_label")
    test_label_dir = os.path.join(raw_dir, "SMD", "test_label")
    
    inventory_rows = []
    total_segments = 0
    
    for mid in expected_machines:
        interp_path = os.path.join(interp_dir, f"{mid}.txt")
        test_label_path = os.path.join(test_label_dir, f"{mid}.txt")
        
        assert os.path.exists(interp_path), f"Missing interp label file: {interp_path}"
        assert os.path.exists(test_label_path), f"Missing test label file: {test_label_path}"
        
        segments = parse_interpretation_labels(interp_path)
        labels = np.loadtxt(test_label_path, delimiter=",", dtype=np.int64)
        
        n_timesteps = len(labels)
        n_anomalous_timesteps = int(np.sum(labels > 0))
        n_events = count_events(labels)
        n_interp_segments = len(segments)
        total_segments += n_interp_segments
        
        inventory_rows.append({
            "machine": mid,
            "n_timesteps": n_timesteps,
            "n_anomalous_timesteps": n_anomalous_timesteps,
            "n_events": n_events,
            "n_interp_segments": n_interp_segments
        })
    
    print(f"Total interpretation segments found across all 28 machines: {total_segments}")
    assert total_segments == 327, f"Expected exactly 327 segments, found {total_segments}"
    print("Assertion PASSED: Exactly 28 machines and exactly 327 interpretation segments.")
    
    inv_df = pd.DataFrame(inventory_rows)
    inv_path = "results_b/inventory.csv"
    inv_df.to_csv(inv_path, index=False)
    print(f"Saved inventory to {inv_path}")
    
    return expected_machines, inv_df

def run_s0_config_freeze():
    print("\n" + "=" * 70)
    print("STAGE S0.2 — Config Freeze")
    print("=" * 70)
    
    config = {
        "dataset_name": "SMD",
        "dataset_path": "data/raw",
        "num_machines": 28,
        "num_sensors": 38,
        "window_size": 100,
        "window_stride": 1,
        "purge_gap_windows": 99,
        "split_ratio": [0.70, 0.15, 0.15],
        "graph_rule": {
            "type": "absolute_correlation",
            "threshold": 0.6,
            "source": "machine_own_train_block"
        },
        "node_features": {
            "type": "6-D",
            "dimensions": [
                "mean",
                "std",
                "skewness",
                "kurtosis",
                "max_abs_dev",
                "fft_dominant_freq_magnitude"
            ]
        },
        "model": {
            "architecture": "GATAnomalyDetector",
            "in_dim": 6,
            "hidden_dim": 64,
            "num_heads": 4,
            "num_layers": 2,
            "edge_dim": 1,
            "dropout": 0.2
        },
        "training": {
            "batch_size": 64,
            "learning_rate": 0.001,
            "weight_decay": 0.0001,
            "epochs": 20,
            "patience": 10,
            "loss": "BCEWithLogitsLoss",
            "optimizer": "Adam",
            "device": "cpu",
            "early_stopping_target": "val_loss"
        },
        "detection_threshold_search": {
            "min": 0.01,
            "max": 0.99,
            "step": 0.01,
            "criterion": "max_val_f1"
        },
        "attribution": {
            "fine_window_margin": 15,
            "ig_steps": 50,
            "ig_baseline": "zero",
            "gnnexplainer_epochs": 60,
            "gnnexplainer_lr": 0.05
        },
        "conformal": {
            "alpha_values": [0.10, 0.20],
            "recall_targets": [1.0, 0.8],
            "quantile_rule": "ceil((n+1)(1-alpha))/n"
        },
        "abstention_gate": {
            "explanation_top_k": 5,
            "min_topk_mass": 0.70,
            "min_sensor_std": 0.001,
            "tau_grid": [round(t * 0.05, 2) for t in range(20)]
        },
        "llm": {
            "model_name": "meta/llama-3.1-70b-instruct",
            "temperature": 0.0,
            "max_tokens": 600,
            "max_retries": 3,
            "backoff_base": 2.0
        }
    }
    
    cfg_path = "results_b/config_frozen.json"
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
        
    cfg_sha256 = get_file_sha256(cfg_path)
    print(f"Saved frozen config to {cfg_path}")
    print(f"SHA-256: {cfg_sha256}")
    
    # Save hash in companion file
    with open("results_b/config_frozen.sha256", "w", encoding="utf-8") as f:
        f.write(cfg_sha256 + "\n")
        
    return config, cfg_sha256

if __name__ == "__main__":
    machines, inv_df = run_s0_inventory()
    config, cfg_sha256 = run_s0_config_freeze()
