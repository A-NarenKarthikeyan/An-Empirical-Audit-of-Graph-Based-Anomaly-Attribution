import sys, os, yaml, torch, glob
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.data.loader import load_dataset
from src.gnn.model import GATAnomalyDetector
from src.gnn.attribution import compute_node_features_6d, compute_integrated_gradients_attribution
from torch_geometric.data import Data as PyGData
from src.explanation.prompt_builder import build_decoupled_explanation_prompt

config_path = "configs/config.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

device = torch.device(cfg.get("device", "cpu"))
gnn = GATAnomalyDetector(in_dim=6, hidden_dim=cfg.get("gnn_hidden_dim", 64), num_heads=cfg.get("gnn_num_heads", 4), edge_dim=1, dropout=0.2).to(device)
if os.path.exists("results/gnn_checkpoint.pt"):
    ckpt = torch.load("results/gnn_checkpoint.pt", map_location=device)
    gnn.load_state_dict(ckpt["model_state_dict"])
gnn.eval()

machine_paths = glob.glob(cfg.get("dataset_path", "data/raw") + "/SMD/train/*.txt")
machines = [os.path.basename(p).replace(".txt", "") for p in machine_paths]

valid_segments = []
for machine in machines:
    if len(valid_segments) >= 20: break
    _, _, interp_labels, meta = load_dataset(dataset_name="SMD", raw_dir=cfg.get("dataset_path", "data/raw"), machine_id=machine, window_size=100, stride=1, corr_threshold=0.6)
    for idx, seg in enumerate(interp_labels):
        if len(valid_segments) >= 20: break
        start, end = seg["start"], seg["end"]
        fine_margin = 15
        start_idx = max(0, int(start - fine_margin))
        end_idx = min(len(meta["norm_test"]), int(end + fine_margin) + 1)
        fine_norm_win = meta["norm_test"][start_idx:end_idx]
        fine_raw_win = meta["raw_test"][start_idx:end_idx]
        node_feats = compute_node_features_6d(fine_norm_win)
        data_sample = PyGData(x=torch.tensor(node_feats, dtype=torch.float32, device=device), edge_index=torch.tensor(meta["edge_index"], dtype=torch.long, device=device))
        ig_scores = compute_integrated_gradients_attribution(gnn, data_sample, steps=20)
        pb_out = build_decoupled_explanation_prompt(idx+1, (start, end), fine_raw_window=fine_raw_win, ig_scores=ig_scores, norm_stats=meta["norm_stats"])
        if not pb_out["abstained"]:
            valid_segments.append((machine, idx+1, pb_out))
            print(f"Found non-abstained: {machine} Seg {idx+1}")
print(f"Total found: {len(valid_segments)}")
