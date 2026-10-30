import sys, os, yaml, torch, json
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.data_loader import load_dataset, compute_node_features_6d
from src.gnn_attribution.model import GATAnomalyDetector
from src.gnn_attribution.validate_attribution import compute_integrated_gradients_attribution
from torch_geometric.data import Data as PyGData
from src.explanation.prompt_builder import build_decoupled_explanation_prompt
from src.evaluation.faithfulness import build_value_perturbed_constrained_prompt, score_value_perturbation_faithfulness
from src.explanation.nim_client import _prompt_hash

config_path = "configs/config.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

device = torch.device(cfg.get("device", "cpu"))
gnn = GATAnomalyDetector(in_dim=6, hidden_dim=cfg.get("gnn_hidden_dim", 64), num_heads=cfg.get("gnn_num_heads", 4), edge_dim=1, dropout=0.2).to(device)
if os.path.exists("results/gnn_checkpoint.pt"):
    ckpt = torch.load("results/gnn_checkpoint.pt", map_location=device)
    gnn.load_state_dict(ckpt["model_state_dict"])
gnn.eval()

failures = [
    ("machine-1-4", 1, 11, -8.50, "inflate"),
    ("machine-1-4", 2, 11, -8.50, "inflate"),
    ("machine-1-4", 4, 14, 8.50, "inflate"),
    ("machine-1-4", 7, 9, 8.50, "inflate"),
    ("machine-1-5", 5, 33, 8.50, "inflate"),
    ("machine-1-5", 6, 19, -8.50, "inflate"),
    ("machine-1-6", 1, 9, 8.50, "inflate"),
    ("machine-1-6", 10, 10, -8.50, "inflate"),
]

# load cache
cache_path = "results/faithfulness_eval_cache.jsonl"
cache = {}
if os.path.exists(cache_path):
    with open(cache_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            d = json.loads(line)
            cache[d["cache_key"]] = d["response"]

top_k = int(cfg.get("explanation_top_k", 5))
min_topk_mass = float(cfg.get("min_topk_mass", 0.70))
min_sensor_std = float(cfg.get("min_sensor_std", 1e-3))
fine_margin = int(cfg.get("fine_window_margin", 15))

for mach, seg_id, t_sensor, new_z, action in failures:
    _, _, interp_labels, meta = load_dataset(dataset_name="SMD", raw_dir="data/raw", machine_id=mach, window_size=100, stride=1, corr_threshold=0.6)
    seg = interp_labels[seg_id - 1]
    start, end = seg["start"], seg["end"]
    start_idx = max(0, int(start - fine_margin))
    end_idx = min(len(meta["norm_test"]), int(end + fine_margin) + 1)
    fine_norm_win = meta["norm_test"][start_idx:end_idx]
    fine_raw_win = meta["raw_test"][start_idx:end_idx]
    
    node_feats = compute_node_features_6d(fine_norm_win)
    data_sample = PyGData(x=torch.tensor(node_feats, dtype=torch.float32, device=device), edge_index=torch.tensor(meta["edge_index"], dtype=torch.long, device=device))
    ig_scores = compute_integrated_gradients_attribution(gnn, data_sample, steps=50)
    
    pb_out = build_decoupled_explanation_prompt(seg_id, (start, end), fine_raw_window=fine_raw_win, ig_scores=ig_scores, norm_stats=meta["norm_stats"], top_k=top_k, min_topk_mass=min_topk_mass, min_sensor_std=min_sensor_std)
    
    v_spec = {"target_sensor_id": t_sensor, "new_z": new_z, "action": action}
    val_pert_pb_out = build_value_perturbed_constrained_prompt(pb_out, v_spec["target_sensor_id"], v_spec["new_z"])
    v_hash_pert = _prompt_hash(val_pert_pb_out["prompt"]) + "_value_pert"
    
    resp = cache.get(v_hash_pert)
    if resp:
        print(f"\n{'='*80}")
        print(f"FAILURE: {mach} Seg {seg_id} (Target Sensor_{t_sensor} {action} to {new_z} std)")
        print(f"{'='*80}")
        # Extract the perturbed evidence lines from the prompt
        evidence_start = val_pert_pb_out['prompt'].find("=== EMPIRICAL EVIDENCE")
        evidence_end = val_pert_pb_out['prompt'].find("=== STATISTICAL CONFIDENCE CONTEXT ===")
        print("PERTURBED EVIDENCE GIVEN TO MODEL:")
        print(val_pert_pb_out['prompt'][evidence_start:evidence_end].strip())
        print(f"\nMODEL EXPLANATION TEXT:")
        try:
            print(resp['explanation'])
        except UnicodeEncodeError:
            print(resp['explanation'].encode('cp1252', errors='replace').decode('cp1252'))
        print(f"\nROOT CAUSE SENSORS JSON: {resp['root_cause_sensors']}")
        
        score, reason = score_value_perturbation_faithfulness(None, resp, t_sensor, new_z, action)
        print(f"SCORING VERDICT: {score} - {reason}")
