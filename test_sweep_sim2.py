import sys, os, yaml, torch, json, glob
import numpy as np

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

target_machines = ["machine-1-1", "machine-1-3", "machine-1-4", "machine-1-5", "machine-1-6"]
all_segments = []

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
        all_segments.append(entry)

# Combined evaluation set: 25 broad sample segments + machine-1-1's 3 abstained segments = 28 segments
comb_28 = [s for s in all_segments if s["is_scored"] or s["machine"] == "machine-1-1"]

def run_sweep(dataset_set, name):
    print("\n" + "="*95)
    print(f"THRESHOLD SWEEP FOR: {name} (N = {len(dataset_set)})")
    print("="*95)
    print(f"{'Thresh (tau)':<12} | {'Abstain Rate':<13} | {'Survivors':<10} | {'Struct. (Const)':<18} | {'Evid. (Value)':<18} | {'LowVar / Insuff'}")
    print("-" * 95)
    
    thresholds = np.arange(0.0, 1.00, 0.05)
    
    for tau in thresholds:
        abstained_count = 0
        low_var_count = 0
        insuff_count = 0
        survivors = []
        
        for seg in dataset_set:
            mass = seg["topk_mass"]
            code = seg["abstain_reason_code"]
            
            is_low_var = (code == "low_variance_filter_removed_candidate")
            
            if is_low_var or (mass < tau):
                abstained_count += 1
                if is_low_var:
                    low_var_count += 1
                else:
                    insuff_count += 1
            else:
                survivors.append(seg)
                
        abst_rate = (abstained_count / len(dataset_set)) * 100.0
        
        scored_survivors = [s for s in survivors if s["is_scored"]]
        n_surv = len(scored_survivors)
        
        if n_surv > 0:
            c_faith = (sum(s["c_score"] for s in scored_survivors) / n_surv) * 100.0
            v_faith = (sum(s["v_score"] for s in scored_survivors) / n_surv) * 100.0
            c_str = f"{c_faith:.1f}% ({sum(s['c_score'] for s in scored_survivors)}/{n_surv})"
            v_str = f"{v_faith:.1f}% ({sum(s['v_score'] for s in scored_survivors)}/{n_surv})"
        else:
            c_faith = 0.0
            v_faith = 0.0
            c_str = "N/A"
            v_str = "N/A"
            
        print(f"{tau:<12.2f} | {abst_rate:>5.1f}% ({abstained_count}/{len(dataset_set)}) | {len(survivors):<10} | {c_str:<18} | {v_str:<18} | {low_var_count} / {insuff_count}")

run_sweep(comb_28, "Combined Evaluation Set (25 Scored Broad Sample + 3 m1-1 Abstained)")
run_sweep(all_segments, "All 68 Segments Across 5 Evaluated Machines")
