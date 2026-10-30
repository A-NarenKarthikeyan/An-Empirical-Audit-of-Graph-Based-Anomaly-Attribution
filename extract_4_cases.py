import sys, os, yaml, torch, json, re
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.data_loader import load_dataset, compute_node_features_6d
from src.gnn_attribution.model import GATAnomalyDetector
from src.gnn_attribution.validate_attribution import compute_integrated_gradients_attribution
from torch_geometric.data import Data as PyGData
from src.explanation.prompt_builder import build_decoupled_explanation_prompt
from src.evaluation.faithfulness import build_value_perturbed_constrained_prompt
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

# 4 distinct inflate cases from different machines
cases = [
    ("machine-1-1", 5, 9, 8.50),
    ("machine-1-3", 4, 11, -8.50),
    ("machine-1-4", 7, 9, 8.50),
    ("machine-1-6", 10, 10, -8.50)
]

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

def extract_evidence_block(prompt_text):
    start = prompt_text.find("=== EMPIRICAL EVIDENCE")
    end = prompt_text.find("=== STATISTICAL CONFIDENCE CONTEXT ===")
    return prompt_text[start:end].strip()

for mach, seg_id, t_sensor, new_z in cases:
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
    
    val_pert_pb_out = build_value_perturbed_constrained_prompt(pb_out, t_sensor, new_z)
    v_hash_pert = _prompt_hash(val_pert_pb_out["prompt"]) + "_value_pert"
    
    resp = cache.get(v_hash_pert)
    
    print("\n" + "="*90)
    print(f"CASE: {mach} Segment {seg_id}")
    print(f"Perturbation: Inflate Sensor_{t_sensor} to z={new_z:+.2f} std")
    print("="*90)
    
    print("\n[1] ORIGINAL UNPERTURBED EVIDENCE:")
    print(extract_evidence_block(pb_out["prompt"]))
    
    print("\n[2] PERTURBED EVIDENCE GIVEN TO MODEL:")
    print(extract_evidence_block(val_pert_pb_out["prompt"]))
    
    if not resp:
        print("\n[!] CACHE ENTRY NOT FOUND.")
        continue
        
    print("\n[3] MODEL EXPLANATION (Raw Text):")
    try:
        print(resp["explanation"])
    except:
        print(resp["explanation"].encode("cp1252", errors="replace").decode("cp1252"))
        
    print(f"\nModel Output root_cause_sensors: {resp['root_cause_sensors']}")
    
    # 4. Keyword / Scoring match logic
    pert_expl = str(resp.get("explanation", "")).lower()
    pert_sensors = set(resp.get("root_cause_sensors", []))
    pattern_target = rf"\bsensor_{t_sensor}\b"
    
    in_list = t_sensor in pert_sensors
    in_prose = re.search(pattern_target, pert_expl) is not None
    
    elevate_keywords = ["dramatic", "highest", "critical", "extreme", "large deviation", "marked", "significant", f"{new_z:+.2f}", f"{abs(new_z):.2f}", "highest z-score", "surge", "spike", "drop"]
    matched_kws = [kw for kw in elevate_keywords if kw in pert_expl]
    
    print("\n[4] SCORING MATCH DETAILS:")
    print(f"- Target Sensor_{t_sensor} in root_cause_sensors list? {in_list}")
    print(f"- Target Sensor_{t_sensor} mentioned in prose? {in_prose}")
    print(f"- Elevating keywords found in prose: {matched_kws}")
    
