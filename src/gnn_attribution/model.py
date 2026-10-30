"""Stage 2: Graph Attention Network (GAT) for sensor anomaly detection and attribution."""

import warnings
from typing import Tuple, Optional, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


class GATAnomalyDetector(nn.Module):
    """Graph Attention Network (GAT) for multivariate time-series anomaly detection.

    Consumes PyG Data objects with:
        - x: Node feature tensor of shape [num_sensors, in_dim] (e.g. [38, 2] for mean, std)
        - edge_index: Graph connectivity tensor of shape [2, num_edges] (e.g. [2, 318])
        - edge_attr: Edge attribute tensor of shape [num_edges, edge_dim] (e.g. [318, 1])
        - y: Window anomaly binary label tensor of shape [batch_size]

    Architecture:
        1. Layer 1: GATConv (in_dim -> hidden_dim * num_heads) with edge_attr support.
        2. Non-linearity (ELU) + Dropout.
        3. Layer 2: GATConv (hidden_dim * num_heads -> hidden_dim) with edge_attr support.
        4. Global Mean Pooling to aggregate sensor embeddings into a graph embedding.
        5. MLP Head (Linear -> ReLU -> Dropout -> Linear) producing a single anomaly score (sigmoid output).
    """

    def __init__(
        self,
        in_dim: int = 6,
        hidden_dim: int = 64,
        num_heads: int = 4,
        edge_dim: int = 1,
        dropout: float = 0.2,
        mlp_hidden_dim: Optional[int] = None
    ):
        super(GATAnomalyDetector, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.edge_dim = edge_dim
        self.dropout = dropout
        mlp_hidden = mlp_hidden_dim or (hidden_dim // 2)

        # Layer 1: Multi-head GAT
        self.conv1 = GATConv(
            in_channels=in_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout
        )

        # Layer 2: Single-head GAT
        self.conv2 = GATConv(
            in_channels=hidden_dim * num_heads,
            out_channels=hidden_dim,
            heads=1,
            concat=False,
            edge_dim=edge_dim,
            dropout=dropout
        )

        # MLP classifier head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(mlp_hidden, 1)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass for single graph or batched graphs.

        Args:
            x: Node features of shape [total_nodes, in_dim].
            edge_index: Edge index of shape [2, total_edges].
            edge_attr: Optional edge attributes of shape [total_edges, edge_dim].
            batch: Optional batch assignment vector of shape [total_nodes].
            return_attention: If True, also returns layer 2 attention weights.

        Returns:
            probs: Anomaly probabilities in [0, 1] of shape [batch_size].
            logits: Raw anomaly logits of shape [batch_size].
            (Optional) attention: Tuple of (edge_index_with_loops, alpha_layer2) if return_attention=True.
        """
        # Layer 1
        h1 = self.conv1(x, edge_index, edge_attr=edge_attr)
        h1 = F.elu(h1)
        h1 = F.dropout(h1, p=self.dropout, training=self.training)

        # Layer 2
        if return_attention:
            h2, (edge_index_layer2, alpha2) = self.conv2(
                h1, edge_index, edge_attr=edge_attr, return_attention_weights=True
            )
        else:
            h2 = self.conv2(h1, edge_index, edge_attr=edge_attr)
            edge_index_layer2, alpha2 = None, None

        # Global Mean Pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        graph_emb = global_mean_pool(h2, batch)

        # MLP Head
        logits = self.mlp(graph_emb).squeeze(-1)
        probs = torch.sigmoid(logits)

        if return_attention:
            return probs, logits, (edge_index_layer2, alpha2)
        return probs, logits


# Alias for backward compatibility with pipeline imports
SensorGNN = GATAnomalyDetector


def get_node_attribution(
    model: nn.Module,
    data: Any
) -> torch.Tensor:
    """Extract normalized per-sensor attribution scores from the last GATConv layer.

    Args:
        model: Trained GATAnomalyDetector or SensorGNN model.
        data: Single PyG Data object (x: [38, 2], edge_index: [2, E], edge_attr: [E, 1]).

    Returns:
        A 1D tensor of shape [num_sensors] containing attribution scores normalized to sum to 1.
    """
    model.eval()
    with torch.no_grad():
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, "edge_attr", None)
        batch = getattr(data, "batch", None)

        probs, logits, (ei_out, alpha) = model(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            batch=batch,
            return_attention=True
        )

        # alpha has shape [num_edges_with_self_loops, heads]
        if alpha.dim() > 1:
            alpha_mean = alpha.mean(dim=-1)
        else:
            alpha_mean = alpha

        num_nodes = x.size(0)
        node_scores = torch.zeros(num_nodes, device=x.device, dtype=torch.float32)
        # Aggregate attention received by target node (ei_out[1])
        node_scores = node_scores.scatter_add(0, ei_out[1], alpha_mean)

        # Normalize to sum to 1
        total_score = node_scores.sum()
        if total_score > 0:
            norm_scores = node_scores / total_score
        else:
            norm_scores = torch.ones(num_nodes, device=x.device) / num_nodes

    return norm_scores


if __name__ == "__main__":
    print("Testing GATAnomalyDetector...")
    model = GATAnomalyDetector(in_dim=2, hidden_dim=64, num_heads=4, edge_dim=1)
    
    # Create sample Data object
    from torch_geometric.data import Data
    x = torch.randn(38, 2)
    edge_index = torch.randint(0, 38, (2, 318))
    edge_attr = torch.randn(318, 1)
    y = torch.tensor([1], dtype=torch.long)
    sample_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    
    probs, logits = model(sample_data.x, sample_data.edge_index, edge_attr=sample_data.edge_attr)
    print(f"Forward output prob: {probs.item():.4f}, logit: {logits.item():.4f}")
    
    attr = get_node_attribution(model, sample_data)
    print(f"Attribution scores shape: {attr.shape}, sum: {attr.sum().item():.4f}")
    print("GATAnomalyDetector module verified successfully.")
