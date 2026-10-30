"""Stage 2: Fine-grained Anomaly Attribution for Multivariate Time-Series.

Computes targeted sensor attribution over tight, fine-grained anomaly windows
without retraining the GNN, avoiding feature dilution on short anomaly spikes.
"""

import os
import warnings
from typing import Dict, Any, List, Tuple, Optional, Union
import numpy as np
import yaml
import torch
import torch.nn as nn

try:
    from torch_geometric.data import Data as PyGData
except ImportError:
    PyGData = None

from src.data_loader import load_dataset, create_pyg_data_object, compute_node_features_6d
from src.gnn_attribution.model import GATAnomalyDetector, get_node_attribution

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def attribute_anomaly(
    model: nn.Module,
    raw_test_data: np.ndarray,
    anomaly_start: int,
    anomaly_end: int,
    edge_index: Any,
    edge_attr: Any,
    fine_window_margin: int = 15,
    top_k: Optional[int] = None
) -> Dict[str, Any]:
    """Extract fine-grained per-sensor attribution for an anomaly segment.

    Extracts a tight window around [anomaly_start, anomaly_end] with margin
    `fine_window_margin`, computes [mean, std] per-sensor node features, constructs
    a PyG Data object with the static graph structure, and queries the frozen GNN.

    Args:
        model: Trained GATAnomalyDetector / SensorGNN model (evaluated in eval mode).
        raw_test_data: 2D numpy array [num_timesteps, num_sensors] (normalized or raw).
        anomaly_start: Start timestep index of the anomaly (0-indexed).
        anomaly_end: End timestep index of the anomaly (inclusive).
        edge_index: Graph connectivity tensor/array [2, num_edges].
        edge_attr: Edge attribute tensor/array [num_edges, edge_dim].
        fine_window_margin: Number of timesteps to expand on both ends (default 15).
        top_k: Optional number of top attributed sensor indices to retrieve.

    Returns:
        Dict containing:
            - 'attribution_scores': 1D Tensor of length [num_sensors] summing to 1.0.
            - 'top_k_indices_0based': List of top-K sensor indices (0 to num_sensors - 1).
            - 'top_k_dims_1based': List of top-K 1-indexed sensor dimensions (1 to num_sensors).
            - 'window_range': Tuple (start_idx, end_idx) of the extracted slice.
            - 'window_length': Length in timesteps of the fine window.
            - 'anomaly_prob': Model anomaly probability on this fine window.
    """
    model.eval()
    device = next(model.parameters()).device

    # 1. Extract fine window
    total_len = len(raw_test_data)
    start_idx = max(0, int(anomaly_start - fine_window_margin))
    end_idx = min(total_len, int(anomaly_end + fine_window_margin) + 1)
    fine_window = raw_test_data[start_idx:end_idx]

    # 2. Compute per-sensor 6-D node features over fine window
    node_features = compute_node_features_6d(fine_window)  # [num_sensors, 6]

    # 3. Build PyG Data object
    x_tensor = torch.tensor(node_features, dtype=torch.float32, device=device)
    if isinstance(edge_index, np.ndarray):
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long, device=device)
    elif isinstance(edge_index, torch.Tensor):
        edge_index_tensor = edge_index.to(device)
    else:
        edge_index_tensor = torch.tensor(edge_index, dtype=torch.long, device=device)

    if edge_attr is not None:
        if isinstance(edge_attr, np.ndarray):
            edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32, device=device)
        elif isinstance(edge_attr, torch.Tensor):
            edge_attr_tensor = edge_attr.to(device)
        else:
            edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float32, device=device)
    else:
        edge_attr_tensor = None

    if PyGData is not None:
        data = PyGData(x=x_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)
    else:
        data = create_pyg_data_object(x=x_tensor, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor)

    # 4. Query frozen model for attention weights and anomaly score
    with torch.no_grad():
        probs, _ = model(data.x, data.edge_index, edge_attr=data.edge_attr)
        attr_scores = get_node_attribution(model, data)  # [num_sensors]

    # 5. Extract top-K attribution
    num_sensors = node_features.shape[0]
    k = top_k if top_k is not None else num_sensors
    k = min(max(1, k), num_sensors)

    topk_vals, topk_idx = torch.topk(attr_scores, k=k)
    topk_idx_0based = topk_idx.cpu().numpy().tolist()
    topk_dims_1based = [int(idx + 1) for idx in topk_idx_0based]

    return {
        "attribution_scores": attr_scores,
        "top_k_indices_0based": topk_idx_0based,
        "top_k_dims_1based": topk_dims_1based,
        "window_range": (start_idx, end_idx),
        "window_length": int(end_idx - start_idx),
        "anomaly_prob": float(probs.item()) if hasattr(probs, "item") else float(probs)
    }


def compare_coarse_vs_fine_attribution(
    model: nn.Module,
    norm_test_data: np.ndarray,
    test_pyg_dataset: List[Any],
    interpretation_labels: List[Dict[str, Any]],
    edge_index: Any,
    edge_attr: Any,
    window_size: int = 100,
    fine_window_margin: int = 15
) -> List[Dict[str, Any]]:
    """Compare attribution overlap % between coarse (100-timestep) and fine-grained windows."""
    results = []

    for idx, seg in enumerate(interpretation_labels, 1):
        start = seg["start"]
        end = seg["end"]
        gt_dims = set(seg["dimensions"])  # 1-indexed in SMD (1 to 38)
        k = len(gt_dims)
        seg_len = end - start + 1

        # --- Coarse Window Attribution (original 100-step sliding window) ---
        mid_idx = min(max(0, (start + end) // 2 - (window_size // 2)), len(test_pyg_dataset) - 1)
        coarse_sample = test_pyg_dataset[mid_idx]
        coarse_attr = get_node_attribution(model, coarse_sample)
        coarse_topk_idx = torch.topk(coarse_attr, k=k).indices.cpu().numpy()
        coarse_topk_dims = set(coarse_topk_idx + 1)
        coarse_overlap = gt_dims.intersection(coarse_topk_dims)
        coarse_ratio = (len(coarse_overlap) / len(gt_dims)) * 100.0 if len(gt_dims) > 0 else 0.0

        # --- Fine Window Attribution (tight margin around actual segment) ---
        fine_res = attribute_anomaly(
            model=model,
            raw_test_data=norm_test_data,
            anomaly_start=start,
            anomaly_end=end,
            edge_index=edge_index,
            edge_attr=edge_attr,
            fine_window_margin=fine_window_margin,
            top_k=k
        )
        fine_topk_dims = set(fine_res["top_k_dims_1based"])
        fine_overlap = gt_dims.intersection(fine_topk_dims)
        fine_ratio = (len(fine_overlap) / len(gt_dims)) * 100.0 if len(gt_dims) > 0 else 0.0

        diff = fine_ratio - coarse_ratio

        results.append({
            "segment_id": idx,
            "timesteps": f"{start}-{end}",
            "seg_len": seg_len,
            "fine_win_len": fine_res["window_length"],
            "gt_dims": sorted(list(gt_dims)),
            "k": k,
            "coarse_topk": sorted(list(coarse_topk_dims)),
            "coarse_overlap": sorted(list(coarse_overlap)),
            "coarse_overlap_count": len(coarse_overlap),
            "coarse_ratio": coarse_ratio,
            "fine_topk": sorted(list(fine_topk_dims)),
            "fine_overlap": sorted(list(fine_overlap)),
            "fine_overlap_count": len(fine_overlap),
            "fine_ratio": fine_ratio,
            "diff": diff
        })

    return results


def print_comparison_table(results: List[Dict[str, Any]], fine_window_margin: int) -> None:
    """Print a side-by-side comparison table of coarse vs fine attribution overlap."""
    print("\n" + "=" * 96)
    print(f"ATTRIBUTION COMPARISON: COARSE (100-step) vs. FINE-GRAINED (Margin={fine_window_margin})")
    print("=" * 96)
    print(
        f"{'Seg':<4} | {'Timesteps':<13} | {'Len':<5} | {'FineLen':<8} | {'GT Count':<8} | "
        f"{'Coarse Overlap':<15} | {'Fine Overlap':<15} | {'Gain':<8}"
    )
    print("-" * 96)

    total_coarse_hits = 0
    total_fine_hits = 0
    total_gt = 0
    coarse_total_overlap = 0
    fine_total_overlap = 0

    for r in results:
        gain_str = f"+{r['diff']:.1f}%" if r["diff"] > 0 else f"{r['diff']:.1f}%"
        print(
            f"{r['segment_id']:<4} | {r['timesteps']:<13} | {r['seg_len']:<5} | {r['fine_win_len']:<8} | {r['k']:<8} | "
            f"{r['coarse_overlap_count']}/{r['k']} ({r['coarse_ratio']:>5.1f}%)   | "
            f"{r['fine_overlap_count']}/{r['k']} ({r['fine_ratio']:>5.1f}%)   | {gain_str:<8}"
        )
        if r["coarse_overlap_count"] > 0:
            total_coarse_hits += 1
        if r["fine_overlap_count"] > 0:
            total_fine_hits += 1
        total_gt += r["k"]
        coarse_total_overlap += r["coarse_overlap_count"]
        fine_total_overlap += r["fine_overlap_count"]

    print("-" * 96)
    coarse_macro_avg = np.mean([r["coarse_ratio"] for r in results])
    fine_macro_avg = np.mean([r["fine_ratio"] for r in results])
    coarse_micro_avg = (coarse_total_overlap / total_gt) * 100.0 if total_gt > 0 else 0.0
    fine_micro_avg = (fine_total_overlap / total_gt) * 100.0 if total_gt > 0 else 0.0

    print(f"Segment Hit Rate (>0 overlap): Coarse = {total_coarse_hits}/{len(results)} | Fine = {total_fine_hits}/{len(results)}")
    print(f"Macro Average Overlap Rate:    Coarse = {coarse_macro_avg:.2f}% | Fine = {fine_macro_avg:.2f}% (Delta: {fine_macro_avg - coarse_macro_avg:+.2f}%)")
    print(f"Micro (Total Dims) Overlap:    Coarse = {coarse_total_overlap}/{total_gt} ({coarse_micro_avg:.2f}%) | Fine = {fine_total_overlap}/{total_gt} ({fine_micro_avg:.2f}%) (Delta: {fine_micro_avg - coarse_micro_avg:+.2f}%)")
    print("=" * 96 + "\n")


if __name__ == "__main__":
    config_path = "configs/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get("device", "cpu"))
    checkpoint_path = "results/gnn_checkpoint.pt"
    fine_margin = int(cfg.get("fine_window_margin", 15))

    print("=" * 80)
    print(f"Running Fine-Grained Attribution Benchmark (Machine: {cfg['machine_id']}, Margin: {fine_margin})")
    print("=" * 80)

    # 1. Load data
    train_wins, (test_wins, test_win_labels), interp_labels, meta = load_dataset(
        dataset_name=cfg.get("dataset_name", "SMD"),
        raw_dir=cfg.get("dataset_path", "data/raw"),
        machine_id=cfg.get("machine_id", "machine-1-1"),
        window_size=cfg.get("window_size", 100),
        stride=cfg.get("window_stride", 1),
        corr_threshold=cfg.get("edge_corr_threshold", 0.6)
    )

    from src.data_loader import build_pyg_dataset
    test_pyg_dataset = build_pyg_dataset(test_wins, test_win_labels, meta["edge_index"], meta["edge_attr"])

    # 2. Load trained GNN model
    model = GATAnomalyDetector(
        in_dim=6,
        hidden_dim=cfg.get("gnn_hidden_dim", 64),
        num_heads=cfg.get("gnn_num_heads", 4),
        edge_dim=1,
        dropout=0.2
    ).to(device)

    if os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"Loaded trained GNN checkpoint from epoch {ckpt.get('epoch', 'N/A')}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Using uninitialized weights.")

    model.eval()

    # 3. Compare coarse vs fine attribution on the 8 interpretation-labeled anomaly segments
    comparison_results = compare_coarse_vs_fine_attribution(
        model=model,
        norm_test_data=meta["norm_test"],
        test_pyg_dataset=test_pyg_dataset,
        interpretation_labels=interp_labels,
        edge_index=meta["edge_index"],
        edge_attr=meta["edge_attr"],
        window_size=cfg.get("window_size", 100),
        fine_window_margin=fine_margin
    )

    # 4. Print comparison table
    print_comparison_table(comparison_results, fine_window_margin=fine_margin)

    # Detailed inspection per segment
    print("\n" + "=" * 80)
    print("DETAILED PER-SEGMENT ATTRIBUTION BREAKDOWN")
    print("=" * 80)
    for r in comparison_results:
        print(f"--- Segment {r['segment_id']} (Timesteps {r['timesteps']}, Length={r['seg_len']}) ---")
        print(f"  Ground Truth Root-Causes ({r['k']} dims): {r['gt_dims']}")
        print(f"  Coarse Top-{r['k']} Attributed Dims:      {r['coarse_topk']}")
        print(f"  Coarse Overlap ({r['coarse_overlap_count']}/{r['k']}):                 {r['coarse_overlap'] if r['coarse_overlap'] else 'None'}")
        print(f"  Fine-Grained Top-{r['k']} Attributed Dims: {r['fine_topk']}")
        print(f"  Fine Overlap ({r['fine_overlap_count']}/{r['k']}):                   {r['fine_overlap'] if r['fine_overlap'] else 'None'}")
        print(f"  Attribution Improvement:              {r['coarse_ratio']:.1f}% -> {r['fine_ratio']:.1f}% ({'+' if r['diff']>0 else ''}{r['diff']:.1f}%)\n")
