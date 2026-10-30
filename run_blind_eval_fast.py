import json, os, sys, time, yaml, threading, glob
import numpy as np
import torch
from torch_geometric.data import Data as PyGData
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

# Imports
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
from src.data_loader import load_dataset, compute_node_features_6d
from src.gnn_attribution.model import GATAnomalyDetector
from src.gnn_attribution.validate_attribution import compute_integrated_gradients_attribution
from src.explanation.prompt_builder import extract_topk_sensor_evidence
from src.evaluation.faithfulness import (
    _call_llm_api, find_counterfactual_swap_sensor, build_perturbed_constrained_prompt,
    build_unconstrained_prompts, build_value_perturbed_constrained_prompt,
    build_decoupled_explanation_prompt, score_faithfulness, score_value_perturbation_faithfulness,
    NIM_BASE_URL, _prompt_hash
)
from src.explanation.nim_client import _get_checkpoint_hash
from dotenv import load_dotenv
load_dotenv()

with open("configs/config.yaml") as f: cfg = yaml.safe_load(f)
device = torch.device("cpu")
top_k = int(cfg.get("explanation_top_k", 5))
fine_margin = int(cfg.get("fine_window_margin", 15))
model_name = cfg.get("nim_model_name", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
temperature = float(cfg.get("nim_temperature", 0.0))
client = OpenAI(base_url=NIM_BASE_URL, api_key=os.getenv("NVIDIA_API_KEY", ""))
OUTPUT_CACHE = "results/threshold_blind_eval_cache.jsonl"
ckpt_hash = _get_checkpoint_hash()

gnn = GATAnomalyDetector(in_dim=6, hidden_dim=64, num_heads=4, edge_dim=1).to(device)
ckpt = torch.load("results/gnn_checkpoint.pt", map_location=device)
gnn.load_state_dict(ckpt["model_state_dict"])
gnn.eval()

existing = set()
if os.path.exists(OUTPUT_CACHE):
    with open(OUTPUT_CACHE) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                existing.add((r["machine"], r["segment_id"], r.get("checkpoint_hash", "legacy")))

with open("threshold_blind_sample.json") as fh: sampled = json.load(fh)

def call_api(sys_msg, prompt, key):
    for attempt in range(5):
        resp = _call_llm_api(client=client, model_name=model_name, system_message=sys_msg, user_prompt=prompt, temperature=temperature, cache_key=key)
        if resp.get("call_status") != "failed_excluded": return resp
        time.sleep(3.0 ** (attempt + 1))
    return resp

def is_failed(resp): return resp.get("call_status") == "failed_excluded" or resp.get("api_error", False)

file_lock = threading.Lock()

def evaluate_segment(seg_info):
    machine = seg_info["machine"]
    seg_id = seg_info["segment_id"]
    if (machine, seg_id, ckpt_hash) in existing: return
    try:
        _, _, interp_labels, meta = load_dataset(dataset_name="SMD", raw_dir="data/raw", machine_id=machine, window_size=100)
    except: return
    seg = interp_labels[seg_id - 1]
    start, end = seg["start"], seg["end"]
    s_idx = max(0, int(start - fine_margin))
    e_idx = min(len(meta["norm_test"]), int(end + fine_margin) + 1)
    fine_norm_win = meta["norm_test"][s_idx:e_idx]
    fine_raw_win = meta["raw_test"][s_idx:e_idx]
    norm_stats = meta["norm_stats"]

    node_feats = compute_node_features_6d(fine_norm_win)
    ea_ten = torch.tensor(meta["edge_attr"], dtype=torch.float32) if meta.get("edge_attr") is not None else None
    data = PyGData(x=torch.tensor(node_feats, dtype=torch.float32), edge_index=torch.tensor(meta["edge_index"], dtype=torch.long), edge_attr=ea_ten)
    ig_scores = compute_integrated_gradients_attribution(gnn, data, steps=50)

    pb_out = build_decoupled_explanation_prompt(
        anomaly_id=seg_id, timesteps=(start, end), fine_raw_window=fine_raw_win, ig_scores=ig_scores,
        norm_stats=norm_stats, top_k=top_k, min_topk_mass=0.0, min_sensor_std=None
    )
    if pb_out["abstained"]: return
    
    topk_mass = pb_out["topk_mass"]
    removed_id, added_id, added_z = find_counterfactual_swap_sensor(
        top_sensor_ids=pb_out["topk_sensor_ids"], fine_raw_window=fine_raw_win, norm_stats=norm_stats, min_std_threshold=0.00
    )
    if removed_id is None: return
    
    pert_pb = build_perturbed_constrained_prompt(pb_output=pb_out, removed_id=removed_id, added_id=added_id, added_z=added_z)
    u_orig, u_pert = build_unconstrained_prompts(
        pb_output=pb_out, fine_raw_window=fine_raw_win, norm_stats=norm_stats, removed_id=removed_id, added_id=added_id, added_z=added_z
    )

    evid_vals = [{"sensor_id": s_id, "z": float(np.mean((fine_raw_win[:, s_id] - norm_stats[s_id]["mean"]) / (norm_stats[s_id]["std"] + 1e-8)))} for s_id in pb_out.get("topk_sensor_ids", [])]
    if not evid_vals: return
    evid_vals.sort(key=lambda x: abs(x["z"]))
    v_target_id = evid_vals[0]["sensor_id"]
    if abs(evid_vals[0]["z"]) < 2.0:
        v_new_z, v_action = (8.50 if evid_vals[0]["z"] >= 0 else -8.50), "inflate"
    else:
        v_new_z, v_action = 0.05, "flatten"

    c_r_orig = call_api(pb_out["system_message"], pb_out["prompt"], _prompt_hash(pb_out["prompt"]) + "_constrained_orig")
    c_r_pert = call_api(pert_pb["system_message"], pert_pb["prompt"], _prompt_hash(pert_pb["prompt"]) + "_constrained_pert")
    u_r_orig = call_api(u_orig["system_message"], u_orig["prompt"], _prompt_hash(u_orig["prompt"]) + "_unconstrained_orig")
    u_r_pert = call_api(u_pert["system_message"], u_pert["prompt"], _prompt_hash(u_pert["prompt"]) + "_unconstrained_pert")
    
    c_failed = is_failed(c_r_orig) or is_failed(c_r_pert)
    u_failed = is_failed(u_r_orig) or is_failed(u_r_pert)
    
    c_score, c_rat = (None, "API failure") if c_failed else score_faithfulness(c_r_orig, c_r_pert, removed_id, added_id, added_z, mode="constrained")
    u_score, u_rat = (None, "API failure") if u_failed else score_faithfulness(u_r_orig, u_r_pert, removed_id, added_id, added_z, mode="unconstrained")

    val_pb = build_value_perturbed_constrained_prompt(pb_output=pb_out, target_sensor_id=v_target_id, new_z=v_new_z)
    v_r_pert = call_api(val_pb["system_message"], val_pb["prompt"], _prompt_hash(val_pb["prompt"]) + "_value_pert")
    
    if is_failed(v_r_pert):
        v_score, v_rat, v_cat = None, "API failure", "api_failure"
    else:
        v_score, v_rat, v_cat = score_value_perturbation_faithfulness(
            orig_response=c_r_orig if not c_failed else {}, pert_response=v_r_pert, target_sensor_id=v_target_id, new_z=v_new_z, action=v_action
        )

    result = {
        "machine": machine, "segment_id": seg_id, "checkpoint_hash": ckpt_hash, "timesteps": f"{start}-{end}",
        "topk_mass": topk_mass, "removed_sensor": f"Sensor_{removed_id}", "added_sensor": f"Sensor_{added_id}", "added_sensor_z": float(added_z),
        "constrained_swap": {"faithfulness_score": c_score, "rationale": c_rat, "orig_root_causes": c_r_orig.get("root_cause_sensors", []) if not c_failed else [], "pert_root_causes": c_r_pert.get("root_cause_sensors", []) if not c_failed else []},
        "unconstrained_swap": {"faithfulness_score": u_score, "rationale": u_rat},
        "value_perturbation": {"faithfulness_score": v_score, "rationale": v_rat, "category": v_cat, "target_sensor": f"Sensor_{v_target_id}", "new_z": v_new_z, "action": v_action, "pert_root_causes": v_r_pert.get("root_cause_sensors", []) if not is_failed(v_r_pert) else []}
    }

    with file_lock:
        with open(OUTPUT_CACHE, "a") as fh: fh.write(json.dumps(result) + "\n")
    print(f"[{machine}-{seg_id}] c_score={c_score} | v_score={v_score}")

print(f"Evaluating {len(sampled)} segments...")
with ThreadPoolExecutor(max_workers=8) as executor:
    list(executor.map(evaluate_segment, sampled))
print("Done!")
