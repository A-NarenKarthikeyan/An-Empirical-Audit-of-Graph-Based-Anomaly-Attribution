"""Step 1: Scan all SMD segments, compute raw top-5 IG mass for every segment
(bypassing the abstention gate), then draw a stratified sample of ≤6 segments
from three mass bins: [0, 0.40), [0.40, 0.70), [0.70, 1.00].

Output: threshold_blind_sample.json  {machine, segment_id, topk_mass}
"""
import glob
import json
import os
import random
import sys
import warnings
warnings.filterwarnings("ignore")

import torch
from torch_geometric.data import Data as PyGData

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_dataset, compute_node_features_6d
from src.gnn_attribution.model import GATAnomalyDetector
from src.gnn_attribution.validate_attribution import compute_integrated_gradients_attribution
from src.explanation.prompt_builder import extract_topk_sensor_evidence

# ---------------------------------------------------------------------------
# Load GNN
# ---------------------------------------------------------------------------
device = torch.device("cpu")
gnn = GATAnomalyDetector(in_dim=6, hidden_dim=64, num_heads=4, edge_dim=1).to(device)
ckpt = torch.load("results/gnn_checkpoint.pt", map_location=device)
gnn.load_state_dict(ckpt["model_state_dict"])
gnn.eval()
print(f"Loaded GNN epoch {ckpt.get('epoch')}.\n")

machine_paths = glob.glob("data/raw/SMD/train/*.txt")
machines = sorted(os.path.basename(p).replace(".txt", "") for p in machine_paths)
print(f"Found {len(machines)} machines.\n")

segment_records = []

for machine in machines:
    try:
        _, _, interp_labels, meta = load_dataset(
            dataset_name="SMD", raw_dir="data/raw", machine_id=machine, window_size=100
        )
    except Exception as e:
        print(f"  [SKIP] {machine}: {e}")
        continue

    norm_test  = meta["norm_test"]
    raw_test   = meta["raw_test"]
    norm_stats = meta["norm_stats"]
    edge_index = meta["edge_index"]
    edge_attr  = meta.get("edge_attr")

    for idx_0, seg in enumerate(interp_labels):
        idx   = idx_0 + 1
        start, end = seg["start"], seg["end"]
        s_idx = max(0, int(start - 15))
        e_idx = min(len(norm_test), int(end + 15) + 1)
        fine_norm_win = norm_test[s_idx:e_idx]
        fine_raw_win  = raw_test[s_idx:e_idx]

        # Build PyG data object
        node_feats = compute_node_features_6d(fine_norm_win)
        ea_ten = torch.tensor(edge_attr, dtype=torch.float32) if edge_attr is not None else None
        data = PyGData(
            x          = torch.tensor(node_feats, dtype=torch.float32),
            edge_index = torch.tensor(edge_index,  dtype=torch.long),
            edge_attr  = ea_ten
        )

        # Compute IG attribution (returns 1D Tensor [num_sensors])
        ig_scores = compute_integrated_gradients_attribution(gnn, data, steps=50)

        # Use extract_topk_sensor_evidence (same as prompt_builder) to get topk_mass
        # Pass min_sensor_std=0.0 so variance floor is not applied
        _, topk_mass, _ = extract_topk_sensor_evidence(
            ig_scores       = ig_scores,
            fine_raw_window = fine_raw_win,
            norm_stats      = norm_stats,
            top_k           = 5,
            one_indexed     = True,
            min_sensor_std  = None   # no variance floor
        )

        segment_records.append({
            "machine":    machine,
            "segment_id": idx,
            "topk_mass":  float(topk_mass)
        })

total = len(segment_records)
print(f"Computed mass for {total} segments across {len(machines)} machines.\n")

# ---- Bin statistics -------------------------------------------------------
bins_raw = {
    "low":  [s for s in segment_records if s["topk_mass"] <  0.40],
    "med":  [s for s in segment_records if 0.40 <= s["topk_mass"] < 0.70],
    "high": [s for s in segment_records if s["topk_mass"] >= 0.70],
}
for name, pool in bins_raw.items():
    masses = [s["topk_mass"] for s in pool]
    mn = min(masses) if masses else 0
    mx = max(masses) if masses else 0
    avg = sum(masses)/len(masses) if masses else 0
    print(f"Bin [{name}] count={len(pool):3d}  mass range=[{mn:.3f},{mx:.3f}]  mean={avg:.3f}")

# ---- Stratified sampling --------------------------------------------------
random.seed(42)
sampled = []
for name, pool in bins_raw.items():
    n = min(6, len(pool))
    chosen = random.sample(pool, n) if pool else []
    sampled.extend(chosen)
    masses = [f"{x['topk_mass']:.3f}" for x in chosen]
    print(f"Bin [{name}]: sampled {n}  ->  masses: {masses}")

# ---- Exclude segments already in the existing 28-segment pool ------------
# (i.e. segments that were already scored in Stage 6 broad sample)
# We will skip de-duplication here; the run_blind_eval.py script will skip
# already-evaluated segments via the blind cache. Keeping all for completeness.
print(f"\nTotal threshold-blind sample: {len(sampled)} segments")
with open("threshold_blind_sample.json", "w") as f:
    json.dump(sampled, f, indent=2)
print("Saved → threshold_blind_sample.json")
