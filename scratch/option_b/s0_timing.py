"""Stage S0.3 & S0.4: Timing on machine-1-1 and GATE G0 Plan Selection."""

import os
import sys
import time
import json
import torch
import numpy as np
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data as PyGData

sys.path.insert(0, os.path.abspath("."))
from src.data_loader import (
    load_smd_raw, normalize_data, create_sliding_windows,
    compute_sensor_correlation_graph, build_pyg_dataset,
    compute_node_features_6d
)
from src.gnn_attribution.model import GATAnomalyDetector
from src.gnn_attribution.train import train_gnn_model, evaluate_model, find_optimal_threshold
from src.gnn_attribution.validate_attribution import compute_integrated_gradients_attribution

def run_timing_machine_1_1():
    print("=" * 70)
    print("STAGE S0.3 — Timing Run on machine-1-1")
    print("=" * 70)
    
    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_num_threads(1)
    
    t_start_total = time.time()
    
    # 1. Load data
    t0 = time.time()
    train_raw, test_raw, test_labels_raw, interp_labels = load_smd_raw("data/raw", "machine-1-1")
    train_norm, test_norm, norm_stats = normalize_data(train_raw, test_raw)
    
    # 2. Graph from train block
    edge_index, edge_attr = compute_sensor_correlation_graph(train_norm, threshold=0.6)
    
    # 3. Sliding windows
    test_wins, test_win_labels = create_sliding_windows(test_norm, test_labels_raw, window_size=100, stride=1)
    
    # 4. PyG dataset
    pyg_dataset = build_pyg_dataset(test_wins, test_win_labels, edge_index, edge_attr)
    
    # 5. Purged split
    n = len(pyg_dataset)
    train_end = int(n * 0.70)
    val_end = train_end + int(n * 0.15)
    
    # Purge last 99 windows of train and val
    train_graphs = pyg_dataset[:train_end - 99]
    val_graphs = pyg_dataset[train_end:val_end - 99]
    test_graphs = pyg_dataset[val_end:]
    
    train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=64, shuffle=False)
    
    prep_time = time.time() - t0
    print(f"Data prep + windowing + graph construction: {prep_time:.2f}s")
    print(f"Purged sizes: Train={len(train_graphs)}, Val={len(val_graphs)}, Test={len(test_graphs)}")
    
    # 6. Train model
    model = GATAnomalyDetector(in_dim=6, hidden_dim=64, num_heads=4, edge_dim=1, dropout=0.2)
    cfg = {
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs": 20,
        "patience": 10,
        "device": "cpu"
    }
    
    t_train_start = time.time()
    timing_ckpt_path = "results_b/timing_ckpt.pt"
    trained_model = train_gnn_model(model, train_loader, val_loader, cfg, checkpoint_path=timing_ckpt_path)
    train_time = time.time() - t_train_start
    print(f"\nTraining time (20 epochs max / early stopping): {train_time:.2f}s")
    
    # 7. Evaluate test set
    t_eval_start = time.time()
    criterion = torch.nn.BCEWithLogitsLoss()
    _, _, val_true, val_prob = evaluate_model(trained_model, val_loader, criterion, torch.device("cpu"), threshold=0.5)
    best_thresh, _ = find_optimal_threshold(val_true, val_prob)
    test_loss, test_metrics, _, _ = evaluate_model(trained_model, test_loader, criterion, torch.device("cpu"), threshold=best_thresh)
    eval_time = time.time() - t_eval_start
    print(f"Validation threshold search + Test eval time: {eval_time:.2f}s")
    print(f"Sample test metrics: F1={test_metrics['f1']:.4f}, AUROC={test_metrics['auroc']:.4f}")
    
    time_per_detector_run = train_time + eval_time + prep_time
    print(f"Total time for one full detector run: {time_per_detector_run:.2f}s ({time_per_detector_run / 60.0:.2f} min)")
    
    # 8. Time 5 IG attributions
    print("\nTiming 5 Integrated Gradients attributions...")
    t_ig_start = time.time()
    ei_ten = torch.tensor(edge_index, dtype=torch.long)
    ea_ten = torch.tensor(edge_attr, dtype=torch.float32) if edge_attr is not None else None
    
    sample_segments = interp_labels[:5]
    for seg in sample_segments:
        s_idx = max(0, int(seg["start"] - 15))
        e_idx = min(len(test_norm), int(seg["end"] + 15) + 1)
        fine_win = test_norm[s_idx:e_idx]
        node_feats = compute_node_features_6d(fine_win)
        data = PyGData(x=torch.tensor(node_feats, dtype=torch.float32), edge_index=ei_ten, edge_attr=ea_ten)
        _ = compute_integrated_gradients_attribution(trained_model, data, steps=50)
        
    ig_5_time = time.time() - t_ig_start
    ig_time_per_segment = ig_5_time / len(sample_segments)
    total_ig_time_327 = ig_time_per_segment * 327
    print(f"5 IG attributions took: {ig_5_time:.2f}s ({ig_time_per_segment:.3f}s per segment)")
    print(f"Projected time for all 327 segments (IG): {total_ig_time_327:.2f}s ({total_ig_time_327 / 60.0:.2f} min)")
    
    # Clean up timing ckpt
    if os.path.exists(timing_ckpt_path):
        os.remove(timing_ckpt_path)
        
    return time_per_detector_run, ig_time_per_segment

def run_s0_gate_g0(time_per_detector_run, ig_time_per_segment):
    print("\n" + "=" * 70)
    print("STAGE S0.4 — GATE G0 Plan Selection")
    print("=" * 70)
    
    cpu_count = os.cpu_count() or 4
    n_workers = max(1, cpu_count - 1)
    print(f"CPU count: {cpu_count}, Using N_WORKERS = {n_workers}")
    
    n_machines = 28
    n_segments = 327
    
    # Full Plan: 3 seeds on all 28 machines = 84 detector runs
    runs_full = n_machines * 3
    detector_hours_full = (runs_full * time_per_detector_run) / (n_workers * 3600.0)
    # Attribution: 327 segments evaluated on seed 42 detector
    attribution_hours = (n_segments * ig_time_per_segment * 2.0) / 3600.0  # + attention/gnnexplainer margin
    total_hours_full = detector_hours_full + attribution_hours
    
    # Reduced Plan: seed 42 on 28 machines + seeds 43, 44 on 6 machines = 28 + 12 = 40 detector runs
    runs_reduced = n_machines + 6 * 2
    detector_hours_reduced = (runs_reduced * time_per_detector_run) / (n_workers * 3600.0)
    total_hours_reduced = detector_hours_reduced + attribution_hours
    
    # Seed 42 only: 28 detector runs
    runs_s42_only = n_machines
    detector_hours_s42 = (runs_s42_only * time_per_detector_run) / (n_workers * 3600.0)
    
    print(f"Time per run: {time_per_detector_run:.1f}s")
    print(f"Projected FULL Plan hours (84 runs, {n_workers} workers): {total_hours_full:.2f} h")
    print(f"Projected REDUCED Plan hours (40 runs, {n_workers} workers): {total_hours_reduced:.2f} h")
    print(f"Projected Seed-42-only hours (28 runs, {n_workers} workers): {detector_hours_s42:.2f} h")
    
    if detector_hours_s42 > 24.0:
        raise RuntimeError(f"Projected seed-42-only time ({detector_hours_s42:.2f} h) exceeds 24 h! Stopping per Rule S0.4.")
        
    if total_hours_full <= 12.0:
        chosen_plan = "FULL"
        plan_desc = "Plan FULL: seeds {42, 43, 44} on all 28 machines (84 detector runs)."
        seeds_per_machine = {m: [42, 43, 44] for m in [f"machine-1-{i}" for i in range(1, 9)] + [f"machine-2-{i}" for i in range(1, 10)] + [f"machine-3-{i}" for i in range(1, 12)]}
    else:
        chosen_plan = "REDUCED"
        multi_seed_machines = ["machine-1-1", "machine-1-5", "machine-2-1", "machine-2-5", "machine-3-1", "machine-3-5"]
        plan_desc = f"Plan REDUCED: seed 42 on all 28 machines; seeds {{43, 44}} only on {', '.join(multi_seed_machines)} (40 detector runs total)."
        seeds_per_machine = {}
        all_m = [f"machine-1-{i}" for i in range(1, 9)] + [f"machine-2-{i}" for i in range(1, 10)] + [f"machine-3-{i}" for i in range(1, 12)]
        for m in all_m:
            if m in multi_seed_machines:
                seeds_per_machine[m] = [42, 43, 44]
            else:
                seeds_per_machine[m] = [42]
                
    print(f"\n>>> CHOSEN PLAN: {chosen_plan} <<<")
    print(plan_desc)
    
    # Write results_b/PLAN.md
    plan_md = f"""# EXECUTION PLAN (GATE G0)

- **Selected Plan**: `{chosen_plan}`
- **Plan Description**: {plan_desc}
- **Hardware Profile**: CPU count = {cpu_count}, Process Pool Workers (`N_WORKERS`) = {n_workers}
- **Benchmark Timing on machine-1-1**:
  - Time per detector run (data prep + 20-epoch training + val search + test eval): `{time_per_detector_run:.2f}s` ({time_per_detector_run/60.0:.2f} min)
  - Time per IG attribution segment: `{ig_time_per_segment:.4f}s`
- **Projections**:
  - Projected FULL Plan Wall-Clock: `{total_hours_full:.2f} hours`
  - Projected REDUCED Plan Wall-Clock: `{total_hours_reduced:.2f} hours`
  - Projected Seed-42-Only Wall-Clock: `{detector_hours_s42:.2f} hours`
- **Decision Rule Triggered**:
  - `total_hours_full <= 12.0 h`: {'YES (Plan FULL selected)' if chosen_plan == 'FULL' else 'NO (Plan REDUCED selected)'}
- **Multi-Seed Coverage**:
  - Total detector runs planned: {sum(len(v) for v in seeds_per_machine.values())}
  - Seed list per machine saved in `results_b/plan_seeds.json`
"""
    with open("results_b/PLAN.md", "w", encoding="utf-8") as f:
        f.write(plan_md)
        
    with open("results_b/plan_seeds.json", "w", encoding="utf-8") as f:
        json.dump({
            "plan": chosen_plan,
            "projected_hours_full": total_hours_full,
            "projected_hours_reduced": total_hours_reduced,
            "time_per_detector_run_s": time_per_detector_run,
            "ig_time_per_segment_s": ig_time_per_segment,
            "n_workers": n_workers,
            "seeds_per_machine": seeds_per_machine
        }, f, indent=4)
        
    print(f"Written plan to results_b/PLAN.md and results_b/plan_seeds.json")
    return chosen_plan

if __name__ == "__main__":
    t_run, t_ig = run_timing_machine_1_1()
    plan = run_s0_gate_g0(t_run, t_ig)
