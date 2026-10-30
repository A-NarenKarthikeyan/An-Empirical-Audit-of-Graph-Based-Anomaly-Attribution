"""Stage 2 Attribution Validation: Multi-Method Sensor Attribution Benchmark.

Compares GAT Attention Attribution against two independent post-hoc explainers:
1. GNNExplainer (node importance mask optimization via mutual information)
2. Integrated Gradients (path-integrated feature gradients with respect to node features)

Evaluates on all 8 ground-truth SMD anomaly segments across both coarse (100-step)
and fine-grained (margin=15) windows, reporting ground-truth overlap % and pairwise
Jaccard agreements.
"""

import os
import time
import warnings
from typing import Dict, Any, List, Tuple, Optional, Set
import numpy as np
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.data import Data as PyGData
except ImportError:
    PyGData = None

from src.data_loader import load_dataset, create_pyg_data_object, build_pyg_dataset, compute_node_features_6d
from src.gnn_attribution.model import GATAnomalyDetector, get_node_attribution
from src.gnn_attribution.attribution import attribute_anomaly

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def compute_attention_attribution(model: nn.Module, data: Any) -> torch.Tensor:
    """Extract normalized per-sensor attribution scores from GAT layer 2 attention."""
    return get_node_attribution(model, data)


def compute_gnnexplainer_attribution(
    model: nn.Module,
    data: Any,
    epochs: int = 60,
    lr: float = 0.05,
    lambda1: float = 0.005,  # L1 mask sparsity penalty
    lambda2: float = 0.01    # Mask entropy penalty
) -> torch.Tensor:
    """Compute per-node importance using GNNExplainer optimization.

    Optimizes a continuous node mask M in [0, 1]^N to maximize mutual information
    with the frozen GNN prediction while encouraging sparsity and binary masks.

    Args:
        model: Frozen GAT anomaly detection model.
        data: PyG Data object.
        epochs: Number of optimization iterations (default 60).
        lr: Learning rate for mask parameter (default 0.05).
        lambda1: Sparsity regularization weight.
        lambda2: Entropy regularization weight.

    Returns:
        1D Tensor of shape [num_nodes] containing importance scores summing to 1.0.
    """
    model.eval()
    device = data.x.device
    num_nodes = data.x.size(0)

    # 1. Target prediction on original input
    with torch.no_grad():
        orig_prob, orig_logit = model(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=getattr(data, "edge_attr", None)
        )
        target_prob = orig_prob.detach()

    # 2. Learnable node mask parameter gamma (initialized so sigmoid(gamma) ~ 0.5)
    node_mask_param = nn.Parameter(torch.randn(num_nodes, 1, device=device) * 0.1)
    optimizer = torch.optim.Adam([node_mask_param], lr=lr)

    bce_loss_fn = nn.BCELoss()

    # 3. Mask optimization loop
    for _ in range(epochs):
        optimizer.zero_grad()
        # Node mask M in (0, 1)
        node_mask = torch.sigmoid(node_mask_param)  # [num_nodes, 1]
        masked_x = data.x * node_mask

        pred_prob, _ = model(
            x=masked_x,
            edge_index=data.edge_index,
            edge_attr=getattr(data, "edge_attr", None)
        )

        # Cross-entropy / prediction retention loss
        pred_loss = bce_loss_fn(pred_prob.view(-1), target_prob.view(-1))

        # Regularization: L1 sparsity
        l1_loss = lambda1 * node_mask.sum()

        # Regularization: Entropy loss to push mask values toward 0 or 1
        eps = 1e-6
        entropy = - (node_mask * torch.log(node_mask + eps) + (1 - node_mask) * torch.log(1 - node_mask + eps))
        ent_loss = lambda2 * entropy.mean()

        total_loss = pred_loss + l1_loss + ent_loss
        total_loss.backward()
        optimizer.step()

    # 4. Extract final node scores
    with torch.no_grad():
        final_mask = torch.sigmoid(node_mask_param).squeeze(-1)  # [num_nodes]
        score_sum = final_mask.sum()
        if score_sum > 0:
            norm_scores = final_mask / score_sum
        else:
            norm_scores = torch.ones_like(final_mask) / num_nodes

    return norm_scores.detach()


def compute_integrated_gradients_attribution(
    model: nn.Module,
    data: Any,
    steps: int = 50,
    baseline: str = "zero"
) -> torch.Tensor:
    """Compute per-sensor attribution using Integrated Gradients on input node features.

    Integrates gradients along the straight-line path from baseline to input x:
    IG(x) = (x - baseline) * (1/M) sum_{m=1}^M grad_x(model(x_m))

    Args:
        model: Frozen GAT anomaly detection model.
        data: PyG Data object.
        steps: Number of Riemann interpolation steps (default 50).
        baseline: Baseline choice ("zero" or "mean").

    Returns:
        1D Tensor of shape [num_nodes] containing importance scores summing to 1.0.
    """
    model.eval()
    x = data.x.clone().detach()
    device = x.device
    num_nodes = x.size(0)

    if baseline == "zero":
        x_base = torch.zeros_like(x, device=device)
    else:
        x_base = x.mean(dim=0, keepdim=True).expand_as(x)

    diff = x - x_base
    accum_grads = torch.zeros_like(x, device=device)

    # Path integration loop
    for step in range(1, steps + 1):
        alpha = float(step) / float(steps)
        x_step = (x_base + alpha * diff).requires_grad_(True)

        probs, logits = model(
            x=x_step,
            edge_index=data.edge_index,
            edge_attr=getattr(data, "edge_attr", None)
        )

        # Target anomaly logit
        target = logits.sum()
        grad = torch.autograd.grad(target, x_step)[0]
        accum_grads += grad.detach()

    avg_grads = accum_grads / float(steps)
    ig_matrix = diff * avg_grads  # [num_nodes, in_dim]

    # Aggregate across feature dimensions (L2 norm per sensor)
    node_ig = torch.norm(ig_matrix, p=2, dim=-1)  # [num_nodes]

    # Normalize to sum to 1.0
    total_ig = node_ig.sum()
    if total_ig > 0:
        norm_scores = node_ig / total_ig
    else:
        norm_scores = torch.ones_like(node_ig) / num_nodes

    return norm_scores.detach()


def get_topk_sensor_dimensions(scores: torch.Tensor, k: int) -> Set[int]:
    """Retrieve the set of 1-indexed top-K sensor dimensions from a score tensor."""
    k = min(max(1, k), scores.size(0))
    topk_idx = torch.topk(scores, k=k).indices.cpu().numpy()
    return set([int(idx + 1) for idx in topk_idx])


def compute_jaccard_similarity(set_a: Set[int], set_b: Set[int]) -> float:
    """Compute Jaccard similarity (Intersection over Union) between two sensor sets."""
    union = set_a.union(set_b)
    if not union:
        return 1.0
    return float(len(set_a.intersection(set_b)) / len(union))


def validate_all_attribution_methods(
    model: nn.Module,
    norm_test_data: np.ndarray,
    test_pyg_dataset: List[Any],
    interpretation_labels: List[Dict[str, Any]],
    edge_index: Any,
    edge_attr: Any,
    window_size: int = 100,
    fine_window_margin: int = 15
) -> List[Dict[str, Any]]:
    """Run Attention, GNNExplainer, and IG on coarse and fine windows for all 8 segments."""
    device = next(model.parameters()).device
    results = []

    for idx, seg in enumerate(interpretation_labels, 1):
        start = seg["start"]
        end = seg["end"]
        gt_dims = set(seg["dimensions"])
        k = len(gt_dims)
        seg_len = end - start + 1

        for win_type in ["Coarse (100-step)", f"Fine (Margin={fine_window_margin})"]:
            # 1. Construct or fetch PyG Data object
            if "Coarse" in win_type:
                mid_idx = min(max(0, (start + end) // 2 - (window_size // 2)), len(test_pyg_dataset) - 1)
                data_sample = test_pyg_dataset[mid_idx].to(device)
                win_len = window_size
            else:
                start_idx = max(0, int(start - fine_window_margin))
                end_idx = min(len(norm_test_data), int(end + fine_window_margin) + 1)
                fine_win = norm_test_data[start_idx:end_idx]
                node_feat = compute_node_features_6d(fine_win)
                x_ten = torch.tensor(node_feat, dtype=torch.float32, device=device)
                ei_ten = torch.tensor(edge_index, dtype=torch.long, device=device)
                ea_ten = torch.tensor(edge_attr, dtype=torch.float32, device=device) if edge_attr is not None else None
                data_sample = PyGData(x=x_ten, edge_index=ei_ten, edge_attr=ea_ten)
                win_len = end_idx - start_idx

            # 2. Compute attributions using all 3 methods
            # Method 1: GAT Attention
            attn_scores = compute_attention_attribution(model, data_sample)
            attn_topk = get_topk_sensor_dimensions(attn_scores, k=k)
            attn_overlap = gt_dims.intersection(attn_topk)
            attn_overlap_pct = (len(attn_overlap) / k) * 100.0 if k > 0 else 0.0

            # Method 2: GNNExplainer
            gnnexp_scores = compute_gnnexplainer_attribution(model, data_sample, epochs=60)
            gnnexp_topk = get_topk_sensor_dimensions(gnnexp_scores, k=k)
            gnnexp_overlap = gt_dims.intersection(gnnexp_topk)
            gnnexp_overlap_pct = (len(gnnexp_overlap) / k) * 100.0 if k > 0 else 0.0

            # Method 3: Integrated Gradients
            ig_scores = compute_integrated_gradients_attribution(model, data_sample, steps=50)
            ig_topk = get_topk_sensor_dimensions(ig_scores, k=k)
            ig_overlap = gt_dims.intersection(ig_topk)
            ig_overlap_pct = (len(ig_overlap) / k) * 100.0 if k > 0 else 0.0

            # 3. Pairwise Jaccard agreements
            jaccard_attn_gnnexp = compute_jaccard_similarity(attn_topk, gnnexp_topk)
            jaccard_attn_ig = compute_jaccard_similarity(attn_topk, ig_topk)
            jaccard_gnnexp_ig = compute_jaccard_similarity(gnnexp_topk, ig_topk)

            results.append({
                "segment_id": idx,
                "timesteps": f"{start}-{end}",
                "seg_len": seg_len,
                "win_type": win_type,
                "win_len": win_len,
                "gt_dims": sorted(list(gt_dims)),
                "k": k,
                # Top-K sets
                "attn_topk": sorted(list(attn_topk)),
                "gnnexp_topk": sorted(list(gnnexp_topk)),
                "ig_topk": sorted(list(ig_topk)),
                # Overlaps with GT
                "attn_overlap_count": len(attn_overlap),
                "attn_overlap_pct": attn_overlap_pct,
                "gnnexp_overlap_count": len(gnnexp_overlap),
                "gnnexp_overlap_pct": gnnexp_overlap_pct,
                "ig_overlap_count": len(ig_overlap),
                "ig_overlap_pct": ig_overlap_pct,
                # Pairwise Jaccard
                "jaccard_attn_gnnexp": jaccard_attn_gnnexp,
                "jaccard_attn_ig": jaccard_attn_ig,
                "jaccard_gnnexp_ig": jaccard_gnnexp_ig,
            })

    return results


def print_validation_summary_table(results: List[Dict[str, Any]]) -> None:
    """Print the complete attribution validation summary table and headline agreement scores."""
    print("\n" + "=" * 115)
    print("STAGE 2 ATTRIBUTION VALIDATION: 3-METHOD BENCHMARK (Attention vs. GNNExplainer vs. Integrated Gradients)")
    print("=" * 115)
    print(
        f"{'Seg':<4} | {'Window Mode':<18} | {'Len':<5} | {'GT(K)':<6} | "
        f"{'Attn Overlap':<13} | {'GNNExp Overlap':<15} | {'IG Overlap':<12} | "
        f"{'Jacc(Attn,GNN)':<14} | {'Jacc(Attn,IG)':<13} | {'Jacc(GNN,IG)':<13}"
    )
    print("-" * 115)

    for r in results:
        print(
            f"{r['segment_id']:<4} | {r['win_type']:<18} | {r['win_len']:<5} | {r['k']:<6} | "
            f"{r['attn_overlap_count']}/{r['k']} ({r['attn_overlap_pct']:>5.1f}%) | "
            f"{r['gnnexp_overlap_count']}/{r['k']} ({r['gnnexp_overlap_pct']:>5.1f}%) | "
            f"{r['ig_overlap_count']}/{r['k']} ({r['ig_overlap_pct']:>5.1f}%) | "
            f"{r['jaccard_attn_gnnexp']:>10.3f}     | {r['jaccard_attn_ig']:>9.3f}     | {r['jaccard_gnnexp_ig']:>9.3f}"
        )

    print("-" * 115)

    # Compute averages
    avg_attn_overlap = np.mean([r["attn_overlap_pct"] for r in results])
    avg_gnnexp_overlap = np.mean([r["gnnexp_overlap_pct"] for r in results])
    avg_ig_overlap = np.mean([r["ig_overlap_pct"] for r in results])

    avg_jacc_attn_gnn = np.mean([r["jaccard_attn_gnnexp"] for r in results])
    avg_jacc_attn_ig = np.mean([r["jaccard_attn_ig"] for r in results])
    avg_jacc_gnn_ig = np.mean([r["jaccard_gnnexp_ig"] for r in results])

    print("\n" + "=" * 80)
    print("HEADLINE SUMMARY STATISTICS ACROSS ALL 16 EVALUATION RUNS (8 Segments x 2 Windows)")
    print("=" * 80)
    print(f"Mean Ground-Truth Overlap:")
    print(f"  - GAT Attention Attribution:    {avg_attn_overlap:.2f}%")
    print(f"  - GNNExplainer Attribution:     {avg_gnnexp_overlap:.2f}%")
    print(f"  - Integrated Gradients (IG):    {avg_ig_overlap:.2f}%")
    print(f"\nOverall Pairwise Jaccard Agreement (IoU between Top-K sensor sets):")
    print(f"  - Attention vs. GNNExplainer:   {avg_jacc_attn_gnn:.4f} ({avg_jacc_attn_gnn*100:.2f}%)")
    print(f"  - Attention vs. IG:             {avg_jacc_attn_ig:.4f} ({avg_jacc_attn_ig*100:.2f}%)")
    print(f"  - GNNExplainer vs. IG:          {avg_jacc_gnn_ig:.4f} ({avg_jacc_gnn_ig*100:.2f}%)")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    config_path = "configs/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    device = torch.device(cfg.get("device", "cpu"))
    checkpoint_path = "results/gnn_checkpoint.pt"
    fine_margin = int(cfg.get("fine_window_margin", 15))

    print("=" * 80)
    print(f"Attribution Validation Pipeline: Machine={cfg['machine_id']}, Margin={fine_margin}, Device={device}")
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

    test_pyg_dataset = build_pyg_dataset(test_wins, test_win_labels, meta["edge_index"], meta["edge_attr"])

    # 2. Load trained frozen GNN
    model = GATAnomalyDetector(
        in_dim=2,
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
        print(f"Warning: Checkpoint not found at {checkpoint_path}.")

    model.eval()

    # 3. Run validation benchmark
    t0 = time.time()
    results = validate_all_attribution_methods(
        model=model,
        norm_test_data=meta["norm_test"],
        test_pyg_dataset=test_pyg_dataset,
        interpretation_labels=interp_labels,
        edge_index=meta["edge_index"],
        edge_attr=meta["edge_attr"],
        window_size=cfg.get("window_size", 100),
        fine_window_margin=fine_margin
    )
    elapsed = time.time() - t0
    print(f"Validation benchmark completed in {elapsed:.2f}s.")

    # 4. Print summary table
    print_validation_summary_table(results)
