"""Stage 1: Data loader and sensor correlation graph builder for multivariate time series datasets (SMD, MSL, SMAP, PSM)."""

import os
import warnings
from typing import Dict, Tuple, Optional, List, Any, Callable
import numpy as np
import yaml

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from torch_geometric.data import Data as PyGData
    PYG_AVAILABLE = True
except ImportError:
    PyGData = None
    PYG_AVAILABLE = False


class PyGDataFallback:
    """Lightweight Data object mimicking torch_geometric.data.Data."""
    def __init__(
        self,
        x: Optional[Any] = None,
        edge_index: Optional[Any] = None,
        edge_attr: Optional[Any] = None,
        y: Optional[Any] = None,
        **kwargs
    ):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self) -> str:
        parts = []
        if self.x is not None:
            shape = list(self.x.shape) if hasattr(self.x, 'shape') else len(self.x)
            parts.append(f"x={shape}")
        if self.edge_index is not None:
            shape = list(self.edge_index.shape) if hasattr(self.edge_index, 'shape') else len(self.edge_index)
            parts.append(f"edge_index={shape}")
        if self.edge_attr is not None:
            shape = list(self.edge_attr.shape) if hasattr(self.edge_attr, 'shape') else len(self.edge_attr)
            parts.append(f"edge_attr={shape}")
        if self.y is not None:
            shape = list(self.y.shape) if hasattr(self.y, 'shape') else '1'
            parts.append(f"y={shape}")
        return f"Data({', '.join(parts)})"


def create_pyg_data_object(x, edge_index, edge_attr=None, y=None):
    if PYG_AVAILABLE and PyGData is not None:
        return PyGData(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)
    return PyGDataFallback(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def compute_sensor_correlation_graph(
    data: np.ndarray,
    threshold: float = 0.6
) -> Tuple[Any, Any]:
    """Compute adjacency matrix and edge index based on sensor correlation matrix.

    Args:
        data: Multivariate time series data of shape (time_steps, num_sensors).
        threshold: Absolute correlation threshold to form an edge.

    Returns:
        edge_index: Tensor or ndarray of shape (2, num_edges).
        edge_attr: Tensor or ndarray of shape (num_edges, 1) representing correlation weights.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        corr_matrix = np.corrcoef(data, rowvar=False)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Thresholding
    adj = np.where(np.abs(corr_matrix) >= threshold, corr_matrix, 0.0)
    np.fill_diagonal(adj, 0.0)  # Remove self loops for explicit handling if needed
    
    source_nodes, target_nodes = np.nonzero(adj)
    
    if TORCH_AVAILABLE and torch is not None:
        edge_index = torch.tensor(np.vstack([source_nodes, target_nodes]), dtype=torch.long)
        edge_attr = torch.tensor(adj[source_nodes, target_nodes], dtype=torch.float).unsqueeze(1)
    else:
        edge_index = np.vstack([source_nodes, target_nodes])
        edge_attr = adj[source_nodes, target_nodes][:, np.newaxis]
    
    return edge_index, edge_attr


def normalize_data(
    train_data: np.ndarray,
    test_data: Optional[np.ndarray] = None,
    eps: float = 1e-8
) -> Tuple[np.ndarray, Optional[np.ndarray], Dict[str, np.ndarray]]:
    """Normalize multivariate time series using z-score per channel, fit on train only.

    Args:
        train_data: Training data array of shape (N_train, num_sensors).
        test_data: Optional test data array of shape (N_test, num_sensors).
        eps: Small constant to avoid division by zero for constant features.

    Returns:
        train_norm: Z-score normalized training data.
        test_norm: Z-score normalized test data (if provided).
        stats: Dictionary containing 'mean' and 'std' computed from train_data.
    """
    mean = np.mean(train_data, axis=0)
    std = np.std(train_data, axis=0)
    
    # Avoid zero standard deviation
    std_adj = np.where(std < eps, 1.0, std)
    
    train_norm = (train_data - mean) / std_adj
    test_norm = (test_data - mean) / std_adj if test_data is not None else None
    
    # Where std is effectively zero in train data, constant channel -> set to 0.0
    zero_std_mask = std < eps
    if np.any(zero_std_mask):
        train_norm[:, zero_std_mask] = 0.0
        if test_norm is not None:
            test_norm[:, zero_std_mask] = 0.0
            
    stats = {"mean": mean, "std": std}
    return train_norm, test_norm, stats


def create_sliding_windows(
    data: np.ndarray,
    labels: Optional[np.ndarray] = None,
    window_size: int = 100,
    stride: int = 1
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Generate sliding window time-series snippets.

    Args:
        data: Time series array of shape (N, num_sensors).
        labels: Optional 1D binary anomaly labels of shape (N,).
        window_size: Number of time steps in each sliding window.
        stride: Stride / step size between consecutive windows.

    Returns:
        windows: Array of shape (num_windows, window_size, num_sensors).
        window_labels: Optional array of shape (num_windows,) where a window is
                       labeled as 1 if any time step in the window is anomalous.
    """
    if len(data) < window_size:
        raise ValueError(f"Data length ({len(data)}) must be >= window_size ({window_size}).")
        
    windows = []
    window_labels = [] if labels is not None else None
    
    for i in range(0, len(data) - window_size + 1, stride):
        windows.append(data[i : i + window_size])
        if labels is not None:
            # Anomaly in window if any timestamp is anomalous (> 0)
            window_labels.append(1 if np.any(labels[i : i + window_size] > 0) else 0)
            
    windows_arr = np.array(windows, dtype=np.float32)
    labels_arr = np.array(window_labels, dtype=np.int64) if window_labels is not None else None
    
    return windows_arr, labels_arr


def compute_node_features_6d(window_data: np.ndarray) -> np.ndarray:
    """Compute 6-dimensional feature vector per sensor for a window or batch of windows.

    Features:
        1. Mean
        2. Standard deviation
        3. Skewness (Fisher-Pearson coefficient)
        4. Kurtosis (excess kurtosis)
        5. Max absolute deviation from window mean
        6. Dominant FFT frequency magnitude (amplitude of highest non-DC component)

    Args:
        window_data: Array of shape (window_size, num_sensors) or (num_windows, window_size, num_sensors).

    Returns:
        Node features of shape (num_sensors, 6) or (num_windows, num_sensors, 6), dtype float32.
    """
    is_3d = (window_data.ndim == 3)
    axis = 1 if is_3d else 0
    T = window_data.shape[axis]

    mean = np.mean(window_data, axis=axis)
    std = np.std(window_data, axis=axis)
    std_safe = np.where(std < 1e-7, 1.0, std)

    if is_3d:
        diff = window_data - mean[:, np.newaxis, :]
        std_expand = std_safe[:, np.newaxis, :]
    else:
        diff = window_data - mean[np.newaxis, :]
        std_expand = std_safe[np.newaxis, :]

    z = diff / std_expand
    skew = np.mean(z**3, axis=axis)
    kurt = np.mean(z**4, axis=axis) - 3.0
    max_dev = np.max(np.abs(diff), axis=axis)

    # Dominant non-DC FFT frequency magnitude
    fft_vals = np.fft.rfft(window_data, axis=axis)
    mags = np.abs(fft_vals) / float(T)
    if is_3d:
        non_dc = mags[:, 1:, :]
    else:
        non_dc = mags[1:, :]

    if non_dc.shape[axis] > 0:
        fft_dominant = np.max(non_dc, axis=axis)
    else:
        fft_dominant = np.zeros_like(mean)

    feats = np.stack([mean, std, skew, kurt, max_dev, fft_dominant], axis=-1).astype(np.float32)
    # Guard against NaN/Inf
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    return feats


def build_pyg_dataset(
    windows: np.ndarray,
    labels: Optional[np.ndarray],
    edge_index: Any,
    edge_attr: Optional[Any] = None
) -> List[Any]:
    """Convert sliding windows into a list of PyTorch Geometric Data objects.

    Node features per window:
        Aggregates each sensor's values within that window into a 6-dim feature vector:
        [mean, std, skewness, kurtosis, max_abs_dev, fft_dominant_freq_magnitude]
        (node feature matrix of shape: num_sensors x 6).
    Target:
        Attaches the window's anomaly label (1 for anomaly, 0 for normal) as data.y.
    Graph Structure:
        Reuses the static sensor correlation graph (edge_index, edge_attr) for every window.

    Args:
        windows: Array of shape (num_windows, window_size, num_sensors).
        labels: Optional 1D array of anomaly labels with shape (num_windows,).
        edge_index: Graph connectivity in COO format with shape (2, num_edges).
        edge_attr: Optional edge attributes/weights with shape (num_edges, 1).

    Returns:
        List of PyG Data objects, one per window.
    """
    if len(windows) == 0:
        return []

    # Chunked computation of 6-D node features to avoid huge RAM allocations
    num_windows = len(windows)
    chunk_size = 10000
    node_features_list = []
    for start in range(0, num_windows, chunk_size):
        end = min(start + chunk_size, num_windows)
        node_features_list.append(compute_node_features_6d(windows[start:end]))
    node_features = np.concatenate(node_features_list, axis=0)

    # Convert edge_index and edge_attr to PyTorch tensors if torch is available
    if TORCH_AVAILABLE and torch is not None:
        if isinstance(edge_index, torch.Tensor):
            edge_index_t = edge_index.clone().detach().long()
        else:
            edge_index_t = torch.tensor(edge_index, dtype=torch.long)

        if edge_attr is not None:
            if isinstance(edge_attr, torch.Tensor):
                edge_attr_t = edge_attr.clone().detach().float()
            else:
                edge_attr_t = torch.tensor(edge_attr, dtype=torch.float32)
        else:
            edge_attr_t = None
    else:
        edge_index_t = edge_index
        edge_attr_t = edge_attr

    dataset = []
    num_windows = len(windows)

    for i in range(num_windows):
        if TORCH_AVAILABLE and torch is not None:
            x_i = torch.from_numpy(node_features[i]).float()
            if labels is not None:
                y_i = torch.tensor([labels[i]], dtype=torch.long)
            else:
                y_i = torch.tensor([0], dtype=torch.long)
            data_obj = create_pyg_data_object(
                x=x_i,
                edge_index=edge_index_t,
                edge_attr=edge_attr_t,
                y=y_i
            )
        else:
            x_i = node_features[i]
            y_i = int(labels[i]) if labels is not None else 0
            data_obj = create_pyg_data_object(
                x=x_i,
                edge_index=edge_index_t,
                edge_attr=edge_attr_t,
                y=y_i
            )
        dataset.append(data_obj)

    return dataset


def parse_interpretation_labels(file_path: str) -> List[Dict[str, Any]]:
    """Parse SMD interpretation label text file.

    Format per line: <start>-<end>:<dim1>,<dim2>,...
    Example: 15849-16368:1,9,10,12,13,14,15

    Args:
        file_path: Path to the interpretation label file.

    Returns:
        List of dictionaries with segment details:
        [
            {
                "start": int,
                "end": int,
                "dimensions": List[int]
            },
            ...
        ]
    """
    segments = []
    if not os.path.exists(file_path):
        return segments

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            
            segment_part, dims_part = line.split(":", 1)
            if "-" not in segment_part:
                continue
            start_str, end_str = segment_part.split("-", 1)
            
            dims = [int(d.strip()) for d in dims_part.split(",") if d.strip().isdigit()]
            segments.append({
                "start": int(start_str.strip()),
                "end": int(end_str.strip()),
                "dimensions": dims
            })
            
    return segments


def load_smd_raw(
    raw_dir: str = "data/raw",
    machine_id: str = "machine-1-1"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Load raw SMD dataset files for a specific machine.

    Args:
        raw_dir: Root directory of raw datasets.
        machine_id: Machine identifier (e.g. 'machine-1-1').

    Returns:
        train_data: Raw training data array of shape (N_train, 38).
        test_data: Raw test data array of shape (N_test, 38).
        test_labels: Raw point-level test anomaly labels of shape (N_test,).
        interpretation_labels: Parsed root-cause dimension labels for anomaly segments.
    """
    smd_dir = os.path.join(raw_dir, "SMD")
    filename = f"{machine_id}.txt"
    
    train_path = os.path.join(smd_dir, "train", filename)
    test_path = os.path.join(smd_dir, "test", filename)
    test_label_path = os.path.join(smd_dir, "test_label", filename)
    interp_label_path = os.path.join(smd_dir, "interpretation_label", filename)

    # Confirm machine_id type
    if isinstance(machine_id, list):
        print(f"[load_smd_raw] Warning: machine_id is a list: {machine_id}. Using first element.")
        machine_id = machine_id[0]
    
    print(f"[load_smd_raw] Loading SMD raw files for machine_id='{machine_id}' (type: {type(machine_id).__name__})")
    print(f"  - Train Path:          {os.path.abspath(train_path)}")
    print(f"  - Test Path:           {os.path.abspath(test_path)}")
    print(f"  - Test Label Path:     {os.path.abspath(test_label_path)}")
    print(f"  - Interp Label Path:   {os.path.abspath(interp_label_path)}")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"SMD train file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"SMD test file not found: {test_path}")
    if not os.path.exists(test_label_path):
        raise FileNotFoundError(f"SMD test_label file not found: {test_label_path}")
        
    train_data = np.loadtxt(train_path, delimiter=",", dtype=np.float32)
    print(f"[load_smd_raw] Raw train array shape immediately after reading: {train_data.shape} (Rows: {train_data.shape[0]}, Channels: {train_data.shape[1]})")

    test_data = np.loadtxt(test_path, delimiter=",", dtype=np.float32)
    test_labels = np.loadtxt(test_label_path, delimiter=",", dtype=np.int64)
    print(f"[load_smd_raw] Raw test array shape: {test_data.shape}, Raw test labels shape: {test_labels.shape}")
    
    interpretation_labels = parse_interpretation_labels(interp_label_path)
    print(f"[load_smd_raw] Loaded {len(interpretation_labels)} interpretation-labeled anomaly segments from {os.path.basename(interp_label_path)}")
    
    return train_data, test_data, test_labels, interpretation_labels

def load_psm_raw(
    raw_dir: str = "data/raw",
    **kwargs
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """Load raw PSM dataset files.

    Args:
        raw_dir: Root directory of raw datasets.

    Returns:
        train_data: Raw training data array of shape (N_train, 25).
        test_data: Raw test data array of shape (N_test, 25).
        test_labels: Raw point-level test anomaly labels of shape (N_test,).
        interpretation_labels: Empty list (PSM doesn't have interpretation labels in this format).
    """
    import pandas as pd
    psm_dir = os.path.join(raw_dir, "PSM")
    train_path = os.path.join(psm_dir, "train.csv")
    test_path = os.path.join(psm_dir, "test.csv")
    test_label_path = os.path.join(psm_dir, "test_label.csv")

    if not os.path.exists(train_path):
        raise FileNotFoundError(f"PSM train file not found: {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"PSM test file not found: {test_path}")
    if not os.path.exists(test_label_path):
        raise FileNotFoundError(f"PSM test_label file not found: {test_label_path}")

    # PSM CSVs have a header row and the first column is 'timestamp_(min)'. 
    # We skip the first column to get the 25 feature columns.
    train_data = pd.read_csv(train_path).iloc[:, 1:].values.astype(np.float32)
    test_data = pd.read_csv(test_path).iloc[:, 1:].values.astype(np.float32)
    test_labels = pd.read_csv(test_label_path).iloc[:, 1].values.astype(np.int64)
    
    return train_data, test_data, test_labels, []


# Dataset loader registry for extensibility (PSM, MSL, SMAP, etc.)
DATASET_LOADERS: Dict[str, Callable] = {
    "SMD": load_smd_raw,
    "PSM": load_psm_raw,
}


def register_dataset_loader(dataset_name: str, loader_fn: Callable) -> None:
    """Register a new raw dataset loader function.
    
    Args:
        dataset_name: Identifier for the dataset (e.g. 'PSM', 'MSL', 'SMAP').
        loader_fn: Function with signature (raw_dir, **kwargs) -> (train_raw, test_raw, test_labels, interpretation_labels).
    """
    DATASET_LOADERS[dataset_name.upper()] = loader_fn


def load_dataset(
    dataset_name: str = "SMD",
    raw_dir: str = "data/raw",
    machine_id: str = "machine-1-1",
    window_size: int = 100,
    stride: int = 1,
    corr_threshold: float = 0.6,
    **kwargs
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray], List[Dict[str, Any]], Dict[str, Any]]:
    """Load and process time series datasets into normalized windowed representations and graph context.

    Args:
        dataset_name: Dataset name ('SMD', 'PSM', 'MSL', 'SMAP').
        raw_dir: Path to raw datasets directory.
        machine_id: Machine / entity identifier for datasets with multiple entities (e.g. SMD).
        window_size: Size of sliding windows.
        stride: Step stride for sliding windows.
        corr_threshold: Sensor correlation threshold for edge construction.
        **kwargs: Additional dataset-specific arguments.

    Returns:
        train_windows: Windowed training data array of shape (N_train_win, window_size, num_sensors).
        test_windows_and_labels: Tuple of (test_windows, test_window_labels) where:
            test_windows has shape (N_test_win, window_size, num_sensors)
            test_window_labels has shape (N_test_win,)
        interpretation_labels: List of ground-truth anomaly segment interpretation labels.
        meta_info: Dictionary containing normalization stats, sensor graph edges, shapes, and raw data references.
    """
    dataset_key = dataset_name.upper()
    if dataset_key not in DATASET_LOADERS:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Currently supported: {list(DATASET_LOADERS.keys())}. "
            f"Register a custom loader via register_dataset_loader()."
        )
        
    loader_fn = DATASET_LOADERS[dataset_key]
    
    # Load raw data
    if dataset_key == "SMD":
        train_raw, test_raw, test_labels_raw, interpretation_labels = loader_fn(
            raw_dir=raw_dir,
            machine_id=machine_id
        )
    else:
        train_raw, test_raw, test_labels_raw, interpretation_labels = loader_fn(
            raw_dir=raw_dir,
            **kwargs
        )
        
    # Z-score normalization fitted on train only
    train_norm, test_norm, norm_stats = normalize_data(train_raw, test_raw)
    
    # Sliding windows
    train_windows, _ = create_sliding_windows(
        data=train_norm,
        labels=None,
        window_size=window_size,
        stride=stride
    )
    
    test_windows, test_window_labels = create_sliding_windows(
        data=test_norm,
        labels=test_labels_raw,
        window_size=window_size,
        stride=stride
    )
    
    # Compute sensor correlation graph from train data
    edge_index, edge_attr = compute_sensor_correlation_graph(
        data=train_norm,
        threshold=corr_threshold
    )
    
    meta_info = {
        "dataset_name": dataset_name,
        "machine_id": machine_id if dataset_key == "SMD" else None,
        "window_size": window_size,
        "stride": stride,
        "corr_threshold": corr_threshold,
        "num_sensors": train_raw.shape[1],
        "train_raw_shape": train_raw.shape,
        "test_raw_shape": test_raw.shape,
        "train_windows_shape": train_windows.shape,
        "test_windows_shape": test_windows.shape,
        "raw_train": train_raw,
        "raw_test": test_raw,
        "raw_test_labels": test_labels_raw,
        "norm_train": train_norm,
        "norm_test": test_norm,
        "norm_stats": norm_stats,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
    }
    
    return train_windows, (test_windows, test_window_labels), interpretation_labels, meta_info


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
    window_size = cfg.get("window_size", 100)
    stride = cfg.get("window_stride", 1)
    corr_threshold = cfg.get("edge_corr_threshold", cfg.get("corr_threshold", 0.6))

    print("=" * 60)
    print(f"Loading SMD Dataset for entity: {machine_id}")
    print(f"Config: dataset={dataset_name}, raw_dir={raw_dir}, window_size={window_size}, stride={stride}")
    print("=" * 60)

    train_wins, (test_wins, test_win_labels), interp_labels, meta = load_dataset(
        dataset_name=dataset_name,
        raw_dir=raw_dir,
        machine_id=machine_id,
        window_size=window_size,
        stride=stride,
        corr_threshold=corr_threshold
    )

    raw_test_labels = meta["raw_test_labels"]
    point_anomaly_ratio = np.mean(raw_test_labels) * 100
    window_anomaly_ratio = np.mean(test_win_labels) * 100

    print("\n--- Data Shapes ---")
    print(f"Train Raw Shape:          {meta['train_raw_shape']}")
    print(f"Test Raw Shape:           {meta['test_raw_shape']}")
    print(f"Train Windows Shape:      {train_wins.shape}")
    print(f"Test Windows Shape:       {test_wins.shape}")
    print(f"Test Window Labels Shape: {test_win_labels.shape}")

    print("\n--- Anomaly Ratios ---")
    print(f"Test Point-level Anomaly Ratio:   {point_anomaly_ratio:.2f}% ({np.sum(raw_test_labels)}/{len(raw_test_labels)} points)")
    print(f"Test Window-level Anomaly Ratio:  {window_anomaly_ratio:.2f}% ({np.sum(test_win_labels)}/{len(test_win_labels)} windows)")

    print("\n--- Interpretation Labels ---")
    print(f"Total anomaly segments with interpretation labels: {len(interp_labels)}")
    for idx, seg in enumerate(interp_labels, 1):
        print(f"  Segment {idx}: Timesteps {seg['start']}-{seg['end']} (Length: {seg['end'] - seg['start'] + 1}) -> Dimensions: {seg['dimensions']}")

    print("\n--- Sensor Graph Edge Information ---")
    if hasattr(meta["edge_index"], "shape"):
        num_edges = meta["edge_index"].shape[1]
    else:
        num_edges = len(meta["edge_index"][0]) if len(meta["edge_index"]) > 0 else 0
    print(f"Number of sensor correlation edges (|r| >= {meta['corr_threshold']}): {num_edges}")

    print("\n--- Converting to PyTorch Geometric Data Objects ---")
    train_pyg_dataset = build_pyg_dataset(
        windows=train_wins,
        labels=None,
        edge_index=meta["edge_index"],
        edge_attr=meta["edge_attr"]
    )
    test_pyg_dataset = build_pyg_dataset(
        windows=test_wins,
        labels=test_win_labels,
        edge_index=meta["edge_index"],
        edge_attr=meta["edge_attr"]
    )

    print(f"Train PyG Dataset size: {len(train_pyg_dataset)} graphs")
    print(f"Test PyG Dataset size:  {len(test_pyg_dataset)} graphs")

    # Sanity check single Data object
    sample_data = test_pyg_dataset[0]
    print("\n--- Sanity Check: Single PyG Data Object (Sample 0 from Test Split) ---")
    print(f"Data object:       {sample_data}")
    print(f"Node features (x): shape={sample_data.x.shape} (sensors={sample_data.x.shape[0]}, features_per_sensor={sample_data.x.shape[1]} [mean, std])")
    print(f"Edge index:        shape={sample_data.edge_index.shape}")
    print(f"Edge attr:         shape={sample_data.edge_attr.shape}")
    print(f"Window Label (y):  {sample_data.y}")

    print("\n[SUCCESS] SMD data loader and PyG dataset construction completed successfully.")
