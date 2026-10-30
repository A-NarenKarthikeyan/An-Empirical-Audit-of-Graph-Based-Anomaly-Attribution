import sys, os, yaml, torch, json, re
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.data_loader import load_dataset, compute_node_features_6d
from src.gnn_attribution.model import GATAnomalyDetector
from src.gnn_attribution.validate_attribution import compute_integrated_gradients_attribution
from torch_geometric.data import Data as PyGData
from src.explanation.prompt_builder import build_decoupled_explanation_prompt
from src.evaluation.faithfulness import build_value_perturbed_constrained_prompt, score_value_perturbation_faithfulness
from src.explanation.nim_client import _prompt_hash
import glob

config_path = "configs/config.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

device = torch.device(cfg.get("device", "cpu"))
gnn = GATAnomalyDetector(in_dim=6, hidden_dim=cfg.get("gnn_hidden_dim", 64), num_heads=cfg.get("gnn_num_heads", 4), edge_dim=1, dropout=0.2).to(device)
ckpt = torch.load("results/gnn_checkpoint.pt", map_location=device)
gnn.load_state_dict(ckpt["model_state_dict"])
gnn.eval()

cache_path = "results/faithfulness_eval_cache.jsonl"
cache = {}
with open(cache_path, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip(): continue
        d = json.loads(line)
        cache[d["cache_key"]] = d["response"]

machine_paths = glob.glob(os.path.join(cfg.get("dataset_path", "data/raw"), "SMD/train/*.txt"))
machines = [os.path.basename(p).replace(".txt", "") for p in machine_paths]

def extract_evidence(prompt_text):
    start = prompt_text.find("=== EMPIRICAL EVIDENCE")
    end = prompt_text.find("=== STATISTICAL CONFIDENCE CONTEXT ===")
    return prompt_text[start:end].strip()

found = 0
for machine in machines:
    if found >= 4: break
    _, _, interp_labels, meta = load_dataset("SMD", "data/raw", machine, window_size=100, stride=1, corr_threshold=0.6)
    for idx_0, seg in enumerate(interp_labels):
        if found >= 4: break
        idx = idx_0 + 1
        start, end = seg["start"], seg["end"]
        start_idx = max(0, int(start - 15))
        end_idx = min(len(meta["norm_test"]), int(end + 15) + 1)
        fine_norm_win = meta["norm_test"][start_idx:end_idx]
        fine_raw_win = meta["raw_test"][start_idx:end_idx]
        
        node_feats = compute_node_features_6d(fine_norm_win)
        data_sample = PyGData(x=torch.tensor(node_feats, dtype=torch.float32, device=device), edge_index=torch.tensor(meta["edge_index"], dtype=torch.long, device=device))
        ig_scores = compute_integrated_gradients_attribution(gnn, data_sample, steps=50)
        
        pb_out = build_decoupled_explanation_prompt(idx, (start, end), fine_raw_window=fine_raw_win, ig_scores=ig_scores, norm_stats=meta["norm_stats"], top_k=5, min_topk_mass=0.70, min_sensor_std=1e-3)
        if pb_out["abstained"]: continue
        
        orig_evidence = pb_out["top_k_evidence"]
        evidence_sorted = sorted(orig_evidence, key=lambda s: abs(s["z_deviation"]))
        weakest_val = evidence_sorted[0]
        
        if abs(weakest_val["z_deviation"]) >= 1.0:
            continue # not an inflate case
            
        new_z = 8.50 if weakest_val["z_deviation"] >= 0 else -8.50
        val_pert_pb_out = build_value_perturbed_constrained_prompt(pb_out, weakest_val["sensor_id"], new_z)
        v_hash_pert = _prompt_hash(val_pert_pb_out["prompt"]) + "_value_pert"
        
        resp = cache.get(v_hash_pert)
        if resp and resp.get('call_status') != 'failed_excluded' and '8.50' in resp.get('explanation', ''):
            found += 1
            print("\n" + "="*90)
            print(f"CASE {found}: {machine} Segment {idx}")
            print(f"Perturbation: Inflate Sensor_{weakest_val['sensor_id']} to z={new_z:+.2f} std")
            print("="*90)
            print("\n[1] ORIGINAL UNPERTURBED EVIDENCE:")
            print(extract_evidence(pb_out["prompt"]))
            print("\n[2] PERTURBED EVIDENCE GIVEN TO MODEL:")
            print(extract_evidence(val_pert_pb_out["prompt"]))
            print("\n[3] MODEL EXPLANATION (Raw Text):")
            try: print(resp["explanation"])
            except: print(resp["explanation"].encode('cp1252', errors='replace').decode('cp1252'))
            print(f"\nModel Output root_cause_sensors: {resp['root_cause_sensors']}")
            
            score, reason = score_value_perturbation_faithfulness(None, resp, weakest_val["sensor_id"], new_z, "inflate")
            print(f"\n[4] SCORING MATCH DETAILS:")
            print(f"Verdict: {reason}")
