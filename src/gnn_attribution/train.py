"""Stage 2: Training and evaluation script for GAT Anomaly Attribution Model."""

import os
import time
import warnings
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from src.data_loader import load_dataset, build_pyg_dataset
from src.gnn_attribution.model import GATAnomalyDetector, get_node_attribution

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


def stratified_train_val_split(
    labels: np.ndarray,
    val_size: float = 0.15,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform a contiguous time-block split into train and validation sets to prevent data leakage.
    (Name kept as stratified_train_val_split for backward compatibility).

    Args:
        labels: 1D array of binary class labels.
        val_size: Fraction of samples to allocate to validation split.
        seed: Ignored.

    Returns:
        train_indices, val_indices as numpy arrays.
    """
    n = len(labels)
    split_idx = int(n * (1.0 - val_size))
    
    train_indices = np.arange(0, split_idx)
    val_indices = np.arange(split_idx, n)
    
    return train_indices, val_indices


def stratified_train_val_cal_split(
    labels: np.ndarray,
    val_size: float = 0.15,
    cal_size: float = 0.15,
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform a contiguous 3-way time-block split into train, val, and cal sets to prevent data leakage.
    (Name kept as stratified_train_val_cal_split for backward compatibility).

    Args:
        labels: 1D array of binary class labels.
        val_size: Fraction of total samples for validation (default 0.15).
        cal_size: Fraction of total samples for calibration (default 0.15).
        seed: Ignored.

    Returns:
        train_indices, val_indices, cal_indices as disjoint numpy arrays.
    """
    n = len(labels)
    train_end = int(n * (1.0 - val_size - cal_size))
    val_end = train_end + int(n * val_size)
    
    train_indices = np.arange(0, train_end)
    val_indices = np.arange(train_end, val_end)
    cal_indices = np.arange(val_end, n)
    
    return train_indices, val_indices, cal_indices


def compute_classification_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """Compute Precision, Recall, F1, and AUROC from true labels and predicted probabilities."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))

    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # AUROC computation using rank-sums (Mann-Whitney U statistic)
    pos_mask = (y_true == 1)
    n_pos = np.sum(pos_mask)
    n_neg = len(y_true) - n_pos

    if n_pos == 0 or n_neg == 0:
        auroc = 0.5
    else:
        from scipy.stats import rankdata
        ranks = rankdata(y_prob)
        sum_pos_ranks = np.sum(ranks[pos_mask])
        u_stat = sum_pos_ranks - n_pos * (n_pos + 1) / 2.0
        auroc = float(u_stat / (n_pos * n_neg))

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    candidates: np.ndarray = None
) -> Tuple[float, Dict[str, float]]:
    """Find the classification threshold that maximises F1 on a given set (e.g. val split).

    Args:
        y_true: 1-D array of ground-truth binary labels.
        y_prob: 1-D array of predicted anomaly probabilities in [0, 1].
        candidates: Array of threshold values to search. Defaults to 0.01..0.99 step 0.01.

    Returns:
        best_threshold: Threshold value that maximised F1.
        best_metrics: Metrics dict from compute_classification_metrics at that threshold.
    """
    if candidates is None:
        candidates = np.arange(0.01, 1.00, 0.01)

    best_f1 = -1.0
    best_threshold = 0.5
    best_metrics: Dict[str, float] = {}

    for tau in candidates:
        m = compute_classification_metrics(y_true, y_prob, threshold=float(tau))
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_threshold = float(tau)
            best_metrics = m

    return best_threshold, best_metrics


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    threshold: float = 0.5
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray]:
    """Evaluate model on a DataLoader, returning loss, metrics, true labels, and predicted probabilities.

    Args:
        model: Trained GATAnomalyDetector.
        data_loader: DataLoader to evaluate on.
        criterion: Loss function.
        device: Torch device.
        threshold: Classification threshold applied to predicted probabilities. Default 0.5.
                   For final test evaluation, pass the val-optimal threshold found by
                   find_optimal_threshold() on the validation split.

    Returns:
        avg_loss, metrics dict, y_true array, y_prob array.
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            y = batch.y.float().view(-1)
            probs, logits = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch
            )
            loss = criterion(logits.view(-1), y)
            total_loss += loss.item() * batch.num_graphs

            all_preds.extend(probs.view(-1).cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    avg_loss = total_loss / len(data_loader.dataset)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    metrics = compute_classification_metrics(all_targets, all_preds, threshold=threshold)
    return avg_loss, metrics, all_targets, all_preds



def train_gnn_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: Dict[str, Any],
    checkpoint_path: str = "results/gnn_checkpoint.pt"
) -> nn.Module:
    """Train GAT model with BCE loss, Adam optimizer, and early stopping on validation loss."""
    device = torch.device(config.get("device", "cpu"))
    model = model.to(device)

    lr = float(config.get("learning_rate", 0.001))
    weight_decay = float(config.get("weight_decay", 1e-4))
    epochs = int(config.get("epochs", 20))
    patience = int(config.get("patience", config.get("early_stopping_patience", 10)))

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    print(f"\nStarting GNN Training on {device}...")
    print(f"Hyperparameters: epochs={epochs}, batch_size={train_loader.batch_size}, lr={lr}, weight_decay={weight_decay}, patience={patience}")
    print("-" * 75)
    print(f"{'Epoch':<8} | {'Train Loss':<12} | {'Val Loss':<12} | {'Val F1':<10} | {'Val AUROC':<10} | {'Time':<8}")
    print("-" * 75)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        model.train()
        train_loss = 0.0

        for batch in train_loader:
            batch = batch.to(device)
            y = batch.y.float().view(-1)

            optimizer.zero_grad()
            probs, logits = model(
                x=batch.x,
                edge_index=batch.edge_index,
                edge_attr=batch.edge_attr,
                batch=batch.batch
            )
            loss = criterion(logits.view(-1), y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * batch.num_graphs

        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_val_loss, val_metrics, _, _ = evaluate_model(model, val_loader, criterion, device)
        epoch_time = time.time() - start_time

        print(
            f"{epoch:<8} | {avg_train_loss:<12.4f} | {avg_val_loss:<12.4f} | "
            f"{val_metrics['f1']:<10.4f} | {val_metrics['auroc']:<10.4f} | {epoch_time:.1f}s"
        )

        # Early stopping check
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": best_val_loss,
                "val_metrics": val_metrics,
                "config": config
            }, checkpoint_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n[Early Stopping] No improvement in validation loss for {patience} consecutive epochs. Stopping at epoch {epoch}.")
                break

    # Load best weights
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"\n[Checkpoint] Loaded best model from epoch {checkpoint['epoch']} with Val Loss: {checkpoint['val_loss']:.4f}")

    return model


def evaluate_interpretation_segments(
    model: nn.Module,
    test_pyg_dataset: List[Any],
    interpretation_labels: List[Dict[str, Any]],
    window_size: int = 100
) -> None:
    """Evaluate sensor attribution overlap with ground-truth root causes for anomaly segments."""
    print("\n" + "=" * 80)
    print("ATTRIBUTION SANITY CHECK: 8 INTERPRETATION-LABELED ANOMALY SEGMENTS")
    print("=" * 80)

    model.eval()
    total_segments = len(interpretation_labels)
    hits = 0

    for idx, seg in enumerate(interpretation_labels, 1):
        start = seg["start"]
        end = seg["end"]
        gt_dims = set(seg["dimensions"])  # 1-indexed in SMD (1 to 38)
        k = len(gt_dims)

        # Pick window corresponding to the anomaly segment (midpoint)
        mid_idx = min(max(0, (start + end) // 2 - (window_size // 2)), len(test_pyg_dataset) - 1)
        sample_data = test_pyg_dataset[mid_idx]

        # Compute per-sensor attribution scores
        attr_scores = get_node_attribution(model, sample_data)
        topk_indices = torch.topk(attr_scores, k=k).indices.cpu().numpy()
        # Convert 0-indexed node IDs (0..37) to 1-indexed SMD dimensions (1..38)
        topk_dims_1based = set(topk_indices + 1)

        overlap = gt_dims.intersection(topk_dims_1based)
        overlap_ratio = len(overlap) / len(gt_dims) if len(gt_dims) > 0 else 0.0
        if len(overlap) > 0:
            hits += 1

        print(f"Segment {idx} (Timesteps {start}-{end}, Length {end - start + 1}):")
        print(f"  Ground-truth root-cause dimensions (1-indexed): {sorted(list(gt_dims))}")
        print(f"  Model Top-{k} attributed dimensions (1-indexed):   {sorted(list(topk_dims_1based))}")
        print(f"  Overlapping dimensions ({len(overlap)}/{len(gt_dims)}):                       {sorted(list(overlap)) if overlap else 'None'}")
        print(f"  Overlap / Hit Ratio:                            {overlap_ratio * 100:.1f}%\n")

    print(f"Summary: {hits}/{total_segments} anomaly segments had overlapping root-cause dimensions in model Top-K attribution.")


if __name__ == "__main__":
    # Load configuration
    config_path = "configs/config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    else:
        cfg = {}

    dataset_name = cfg.get("dataset_name", "SMD")
    raw_dir = cfg.get("dataset_path", "data/raw")
    machine_id = cfg.get("machine_id", "machine-1-1")
    window_size = int(cfg.get("window_size", 100))
    stride = int(cfg.get("window_stride", 1))
    corr_threshold = float(cfg.get("edge_corr_threshold", cfg.get("corr_threshold", 0.6)))
    batch_size = int(cfg.get("batch_size", 64))
    hidden_dim = int(cfg.get("gnn_hidden_dim", 64))
    num_heads = int(cfg.get("gnn_num_heads", 4))
    device_str = cfg.get("device", "cpu")
    device = torch.device(device_str)

    print("=" * 80)
    print(f"GNN Training Pipeline: Dataset={dataset_name}, Machine={machine_id}, Device={device}")
    print("=" * 80)

    # 1. Load dataset and create sliding windows
    print("\n[Step 1] Loading raw data & generating sliding windows...")
    train_wins, (test_wins, test_win_labels), interp_labels, meta = load_dataset(
        dataset_name=dataset_name,
        raw_dir=raw_dir,
        machine_id=machine_id,
        window_size=window_size,
        stride=stride,
        corr_threshold=corr_threshold
    )

    # 2. Build PyG datasets
    print("[Step 2] Constructing PyTorch Geometric graph datasets...")
    test_pyg_dataset = build_pyg_dataset(
        windows=test_wins,
        labels=test_win_labels,
        edge_index=meta["edge_index"],
        edge_attr=meta["edge_attr"]
    )
    print(f"Total labeled test graphs: {len(test_pyg_dataset)} (Anomaly ratio: {np.mean(test_win_labels)*100:.2f}%)")

    # 3. Contiguous 70/15/15 train/validation/test split
    print("\n[Step 3] Creating 70/15/15 contiguous train/validation/test split...")
    train_idx, val_idx, test_idx = stratified_train_val_cal_split(test_win_labels, val_size=0.15, cal_size=0.15, seed=42)
    train_graphs = [test_pyg_dataset[i] for i in train_idx]
    val_graphs = [test_pyg_dataset[i] for i in val_idx]
    test_graphs = [test_pyg_dataset[i] for i in test_idx]

    train_anomaly_ratio = np.mean([g.y.item() for g in train_graphs]) * 100
    val_anomaly_ratio = np.mean([g.y.item() for g in val_graphs]) * 100
    test_anomaly_ratio = np.mean([g.y.item() for g in test_graphs]) * 100
    print(f"Train split: {len(train_graphs)} graphs (Anomaly ratio: {train_anomaly_ratio:.2f}%)")
    print(f"Val split:   {len(val_graphs)} graphs (Anomaly ratio: {val_anomaly_ratio:.2f}%)")
    print(f"Test split:  {len(test_graphs)} graphs (Anomaly ratio: {test_anomaly_ratio:.2f}%)")

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    # 4. Instantiate model
    print("\n[Step 4] Initializing GATAnomalyDetector...")
    model = GATAnomalyDetector(
        in_dim=6,
        hidden_dim=hidden_dim,
        num_heads=num_heads,
        edge_dim=1,
        dropout=0.2
    ).to(device)

    # 5. Train model
    checkpoint_path = "results/gnn_checkpoint.pt"
    model = train_gnn_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg,
        checkpoint_path=checkpoint_path
    )

    # 6. Find optimal threshold on val split, then evaluate test split with it
    criterion = nn.BCEWithLogitsLoss()

    print("\n" + "=" * 80)
    print("THRESHOLD SELECTION (val split)")
    print("=" * 80)
    _, _, val_true, val_prob = evaluate_model(model, val_loader, criterion, device, threshold=0.5)
    opt_threshold, opt_val_metrics = find_optimal_threshold(val_true, val_prob)
    print(f"Val-optimal threshold (maximises F1 on val split): tau = {opt_threshold:.2f}")
    print(f"Val F1 at tau={opt_threshold:.2f}: {opt_val_metrics['f1']*100:.2f}%  "
          f"(P={opt_val_metrics['precision']*100:.2f}%, R={opt_val_metrics['recall']*100:.2f}%)")
    print("[Policy: threshold selected on val split, applied unchanged to test split]")

    print("\n" + "=" * 80)
    print("FINAL TEST EVALUATION METRICS")
    print("=" * 80)
    test_loss, test_metrics, y_true, y_prob = evaluate_model(
        model, test_loader, criterion, device, threshold=opt_threshold
    )

    print(f"Classification threshold (val-optimal): tau = {opt_threshold:.2f}")
    print(f"Test Loss:      {test_loss:.4f}")
    print(f"Precision:      {test_metrics['precision']:.4f} ({test_metrics['precision']*100:.2f}%)")
    print(f"Recall:         {test_metrics['recall']:.4f} ({test_metrics['recall']*100:.2f}%)")
    print(f"F1 Score:       {test_metrics['f1']:.4f} ({test_metrics['f1']*100:.2f}%)")
    print(f"AUROC:          {test_metrics['auroc']:.4f} ({test_metrics['auroc']*100:.2f}%)")
    print(f"Confusion:      TP={test_metrics['tp']}, FP={test_metrics['fp']}, FN={test_metrics['fn']}, TN={test_metrics['tn']}")

    # Save metrics + threshold to file so other scripts can dynamically reference them
    import json
    os.makedirs("results", exist_ok=True)
    test_metrics["threshold"] = opt_threshold
    with open("results/smd_test_metrics.json", "w") as f:
        json.dump(test_metrics, f, indent=4)

    # 7. Attribution check on 8 interpretation-labeled segments
    evaluate_interpretation_segments(
        model=model,
        test_pyg_dataset=test_pyg_dataset,
        interpretation_labels=interp_labels,
        window_size=window_size
    )

    print("\n[SUCCESS] Stage 2 GNN Training and Attribution Evaluation completed successfully.")
