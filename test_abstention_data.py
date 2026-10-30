import sys, os, yaml, torch, json, glob
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_dataset, compute_node_features_6d
from src.gnn_attribution.model import GATAnomalyDetector
from src.gnn_attribution.validate_attribution import compute_integrated_gradients_attribution
from torch_geometric.data import Data as PyGData
from src.explanation.prompt_builder import build_decoupled_explanation_prompt
from src.evaluation.faithfulness import run_faithfulness_evaluation

config_path = "configs/config.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

device = torch.device(cfg.get("device", "cpu"))
gnn = GATAnomalyDetector(in_dim=6, hidden_dim=cfg.get("gnn_hidden_dim", 64), num_heads=cfg.get("gnn_num_heads", 4), edge_dim=1, dropout=0.2).to(device)
ckpt = torch.load("results/gnn_checkpoint.pt", map_location=device)
gnn.load_state_dict(ckpt["model_state_dict"])
gnn.eval()

# Run faithfulness eval to get scored non-abstained segments
eval_out = run_faithfulness_evaluation(verbose=False)
eval_results = eval_out["results"]

scored_lookup = {}
for r in eval_results:
    key = (r["machine"], r["segment_id"])
    scored_lookup[key] = {
        "c_score": r["constrained_swap"]["faithfulness_score"],
        "u_score": r["unconstrained_swap"]["faithfulness_score"],
        "v_score": r["value_perturbation"]["faithfulness_score"],
        "v_category": r["value_perturbation"]["category"]
    }

print(f"Loaded {len(scored_lookup)} scored segments from Part 8.")

# Collect all segments from the evaluated machines
target_machines = ["machine-1-1", "machine-1-3", "machine-1-4", "machine-1-5", "machine-1-6"]

all_segment_data = []

for machine in target_machines:
    _, _, interp_labels, meta = load_dataset("SMD", "data/raw", machine, window_size=100, stride=1, corr_threshold=0.6)
    for idx_0, seg in enumerate(interp_labels):
        idx = idx_0 + 1
        start, end = seg["start"], seg["end"]
        start_idx = max(0, int(start - 15))
        end_idx = min(len(meta["norm_test"]), int(end + 15) + 1)
        fine_norm_win = meta["norm_test"][start_idx:end_idx]
        fine_raw_win = meta["raw_test"][start_idx:end_idx]
        
        node_feats = compute_node_features_6d(fine_norm_win)
        data_sample = PyGData(x=torch.tensor(node_feats, dtype=torch.float32, device=device), edge_index=torch.tensor(meta["edge_index"], dtype=torch.long, device=device))
        ig_scores = compute_integrated_gradients_attribution(gnn, data_sample, steps=50)
        
        pb_out = build_decoupled_explanation_prompt(
            idx, (start, end), fine_raw_window=fine_raw_win, ig_scores=ig_scores,
            norm_stats=meta["norm_stats"], top_k=5, min_topk_mass=0.70, min_sensor_std=1e-3
        )
        
        key = (machine, idx)
        scored_info = scored_lookup.get(key)
        
        entry = {
            "machine": machine,
            "segment_id": idx,
            "timesteps": f"{start}-{end}",
            "topk_mass": pb_out["cumulative_mass"],
            "unfiltered_mass": pb_out.get("unfiltered_mass"),
            "abstained_at_070": pb_out["abstained"],
            "abstain_reason_code": pb_out.get("abstain_reason_code"),
            "is_scored": scored_info is not None,
            "c_score": scored_info["c_score"] if scored_info else None,
            "u_score": scored_info["u_score"] if scored_info else None,
            "v_score": scored_info["v_score"] if scored_info else None,
            "v_category": scored_info["v_category"] if scored_info else None
        }
        all_segment_data.append(entry)

print(f"\nTotal segments across targeted machines: {len(all_segment_data)}")
print(f"Scored: {sum(1 for e in all_segment_data if e['is_scored'])}")
print(f"Abstained at 0.70: {sum(1 for e in all_segment_data if e['abstained_at_070'])}")

print("\n--- SAMPLE SEGMENTS TABLE ---")
print(f"{'Machine':<12} | {'Seg':<3} | {'TopK Mass':<10} | {'Abstained@0.70':<14} | {'Abstain Reason':<35} | {'Scored'}")
print("-" * 88)
for e in all_segment_data:
    ar = str(e["abstain_reason_code"])
    print(f"{e['machine']:<12} | {e['segment_id']:<3} | {e['topk_mass']:<10.4f} | {str(e['abstained_at_070']):<14} | {ar:<35} | {e['is_scored']}")
