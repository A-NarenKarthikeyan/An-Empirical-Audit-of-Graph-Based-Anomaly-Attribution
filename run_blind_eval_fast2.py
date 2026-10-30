"""Step 2: Blind evaluation of stratified sample.

Runs sensor-swap + value-perturbation faithfulness tests on the threshold-blind
sample WITHOUT any min_topk_mass or variance-floor gate.

_call_llm_api returns the parsed response dict DIRECTLY (not wrapped in
{"response": ...}), so we use the returned value as the response object.

Results are incrementally appended to results/threshold_blind_eval_cache.jsonl.
Already-cached segments are skipped.
"""
import glob
import json
import os
import sys
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch_geometric.data import Data as PyGData
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

import yaml

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.data_loader import load_dataset, compute_node_features_6d
from src.gnn_attribution.model import GATAnomalyDetector
from src.gnn_attribution.validate_attribution import compute_integrated_gradients_attribution
from src.explanation.prompt_builder import extract_topk_sensor_evidence

# All helpers live in faithfulness.py
from src.evaluation.faithfulness import (
    _call_llm_api,
    find_counterfactual_swap_sensor,
    build_perturbed_constrained_prompt,
    build_unconstrained_prompts,
    build_value_perturbed_constrained_prompt,
    build_decoupled_explanation_prompt,
    score_faithfulness,
    score_value_perturbation_faithfulness,
    NIM_BASE_URL,
    _prompt_hash,
)

# ── Config ──────────────────────────────────────────────────────────────────
with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

device      = torch.device("cpu")
top_k       = int(cfg.get("explanation_top_k", 5))
fine_margin = int(cfg.get("fine_window_margin", 15))
model_name  = cfg.get("nim_model_name", "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning")
temperature = float(cfg.get("nim_temperature", 0.0))

api_key = os.getenv("NVIDIA_API_KEY", "")
if not api_key:
    raise EnvironmentError("NVIDIA_API_KEY not set")
client = OpenAI(base_url=NIM_BASE_URL, api_key=api_key)

OUTPUT_CACHE = "results/threshold_blind_eval_cache.jsonl"

# ── GNN ──────────────────────────────────────────────────────────────────────
gnn = GATAnomalyDetector(in_dim=6, hidden_dim=64, num_heads=4, edge_dim=1).to(device)
ckpt = torch.load("results/gnn_checkpoint.pt", map_location=device)
gnn.load_state_dict(ckpt["model_state_dict"])
gnn.eval()
print(f"Loaded GNN epoch {ckpt.get('epoch')}")

# ── Existing blind cache ──────────────────────────────────────────────────────
existing = {}
from src.explanation.nim_client import _get_checkpoint_hash
ckpt_hash = _get_checkpoint_hash()

if os.path.exists(OUTPUT_CACHE):
        with open(OUTPUT_CACHE) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rec = json.loads(line)
                # Key now includes the checkpoint hash
                rec_ckpt = rec.get("checkpoint_hash", "legacy")
                existing[(rec["machine"], rec["segment_id"], rec_ckpt)] = rec
print(f"Already-evaluated blind entries: {len(existing)}")

# ── Blind sample ──────────────────────────────────────────────────────────────
with open("threshold_blind_sample.json") as fh:
    sampled = json.load(fh)
print(f"Segments to evaluate: {len(sampled)}\n")


# ── Helper: call with retry ───────────────────────────────────────────────────
def call_api(sys_msg, prompt, key, retries=5):
    """Call LLM and retry on failure. Returns the parsed response dict directly."""
    for attempt in range(retries):
        resp = _call_llm_api(
            client=client, model_name=model_name,
            system_message=sys_msg, user_prompt=prompt,
            temperature=temperature, cache_key=key
        )
        if resp.get("call_status") != "failed_excluded":
            return resp
        wait = min(3.0 ** (attempt + 1), 60.0)
        print(f"    [Retry {attempt+1}/{retries}] failed. Waiting {wait:.0f}s...")
        time.sleep(wait)
    return resp  # return last (failed) response

def is_failed(resp):
    """True if the API call failed (no scoreable response)."""
    return resp.get("call_status") == "failed_excluded" or resp.get("api_error", False)


# ── Main loop ─────────────────────────────────────────────────────────────────

import threading
    file_lock = threading.Lock()

def evaluate_segment(seg_info):
    
        machine = seg_info["machine"]
        seg_id  = seg_info["segment_id"]
    
        if (machine, seg_id, ckpt_hash) in existing:
            print(f"[SKIP] {machine} Seg {seg_id} (already evaluated for this ckpt)")
            return
    
        print(f"\n{'='*72}")
        print(f"[EVAL] {machine}  Seg {seg_id}  (expected mass ~= {seg_info['topk_mass']:.3f})")
    
        try:
            _, _, interp_labels, meta = load_dataset(
                dataset_name="SMD", raw_dir="data/raw", machine_id=machine, window_size=100
            )
        except Exception as exc:
            print(f"  [ERROR] load_dataset: {exc}")
            return
    
        seg        = interp_labels[seg_id - 1]
        start, end = seg["start"], seg["end"]
        s_idx = max(0, int(start - fine_margin))
        e_idx = min(len(meta["norm_test"]), int(end + fine_margin) + 1)
        fine_norm_win = meta["norm_test"][s_idx:e_idx]
        fine_raw_win  = meta["raw_test"][s_idx:e_idx]
        norm_stats    = meta["norm_stats"]
    
        # ── IG attribution ─────────────────────────────────────────────────────────
        node_feats = compute_node_features_6d(fine_norm_win)
        ea_ten = torch.tensor(meta["edge_attr"], dtype=torch.float32) if meta.get("edge_attr") is not None else None
        data = PyGData(
            x          = torch.tensor(node_feats, dtype=torch.float32),
            edge_index = torch.tensor(meta["edge_index"], dtype=torch.long),
            edge_attr  = ea_ten
        )
        ig_scores = compute_integrated_gradients_attribution(gnn, data, steps=50)
    
        # ── Constrained prompt — bypass abstention ────────────────────────────────
        pb_out = build_decoupled_explanation_prompt(
            anomaly_id      = seg_id,
            timesteps       = (start, end),
            fine_raw_window = fine_raw_win,
            ig_scores       = ig_scores,
            norm_stats      = norm_stats,
            top_k           = top_k,
            min_topk_mass   = 0.0,   # blind: no abstention gate
            min_sensor_std  = None,  # blind: no variance floor
        )
    
        topk_mass     = pb_out["cumulative_mass"]
        orig_evidence = pb_out["top_k_evidence"]
        topk_dims     = pb_out["top_k_dims"]
        print(f"  Top-K mass = {topk_mass:.4f} | dims = {topk_dims}")
    
        # ── Sensor-swap counterfactual ─────────────────────────────────────────────
        weakest, replacement = find_counterfactual_swap_sensor(
            orig_topk_evidence = orig_evidence,
            fine_raw_window    = fine_raw_win,
            norm_stats         = norm_stats,
            topk_dims          = topk_dims,
            min_sensor_std     = 0.0
        )
        removed_id = weakest["sensor_id"]
        added_id   = replacement["sensor_id"]
        added_z    = replacement["z_deviation"]
    
        pert_pb  = build_perturbed_constrained_prompt(pb_output=pb_out, weakest=weakest, replacement=replacement)
        u_orig, u_pert = build_unconstrained_prompts(
            anomaly_id            = seg_id,
            timesteps             = (start, end),
            fine_raw_window       = fine_raw_win,
            norm_stats            = norm_stats,
            weakest_sensor_id     = removed_id,
            replacement_sensor_id = added_id
        )
    
        c_orig_key = _prompt_hash(pb_out["prompt"])   + "_constrained_orig"
        c_pert_key = _prompt_hash(pert_pb["prompt"])  + "_constrained_pert"
        u_orig_key = _prompt_hash(u_orig["prompt"])   + "_unconstrained_orig"
        u_pert_key = _prompt_hash(u_pert["prompt"])   + "_unconstrained_pert"
    
        print(f"  Swap: remove Sensor_{removed_id} -> add Sensor_{added_id} (z={added_z:+.2f})")
    
        # _call_llm_api returns the parsed response dict directly (NOT {"response": ...})
        c_r_orig = call_api(pb_out["system_message"],  pb_out["prompt"],  c_orig_key)
        c_r_pert = call_api(pert_pb["system_message"],  pert_pb["prompt"],  c_pert_key)
        u_r_orig = call_api(u_orig["system_message"],  u_orig["prompt"],  u_orig_key)
        u_r_pert = call_api(u_pert["system_message"],  u_pert["prompt"],  u_pert_key)
    
        c_failed = is_failed(c_r_orig) or is_failed(c_r_pert)
        u_failed = is_failed(u_r_orig) or is_failed(u_r_pert)
    
        if c_failed:
            c_score, c_rat = None, "API failure"
        else:
            # c_r_orig and c_r_pert ARE the response dicts (root_cause_sensors, explanation, ...)
            c_score, c_rat = score_faithfulness(
                c_r_orig, c_r_pert,
                removed_id, added_id, added_z, mode="constrained"
            )
    
        if u_failed:
            u_score, u_rat = None, "API failure"
        else:
            u_score, u_rat = score_faithfulness(
                u_r_orig, u_r_pert,
                removed_id, added_id, added_z, mode="unconstrained"
            )
    
        # ── Value-perturbation ─────────────────────────────────────────────────────
        evidence_sorted = sorted(orig_evidence, key=lambda s: abs(s["z_deviation"]))
        weakest_val = evidence_sorted[0]
        if abs(weakest_val["z_deviation"]) < 1.0:
            v_target_id = weakest_val["sensor_id"]
            v_new_z     = 8.50 if weakest_val["z_deviation"] >= 0 else -8.50
            v_action    = "inflate"
        else:
            v_target_id = weakest_val["sensor_id"]
            v_new_z     = 0.05
            v_action    = "flatten"
    
        val_pb     = build_value_perturbed_constrained_prompt(pb_output=pb_out, target_sensor_id=v_target_id, new_z=v_new_z)
        v_pert_key = _prompt_hash(val_pb["prompt"]) + "_value_pert"
        v_r_pert   = call_api(val_pb["system_message"], val_pb["prompt"], v_pert_key)
    
        if is_failed(v_r_pert):
            v_score, v_rat, v_cat = None, "API failure", "api_failure"
        else:
            # orig response is c_r_orig (the response dict directly)
            v_score, v_rat, v_cat = score_value_perturbation_faithfulness(
                orig_response    = c_r_orig if not c_failed else {},
                pert_response    = v_r_pert,
                target_sensor_id = v_target_id,
                new_z            = v_new_z,
                action           = v_action
            )
    
        # ── Write result ───────────────────────────────────────────────────────────
        result = {
            "machine":    machine,
            "segment_id": seg_id,
            "checkpoint_hash": ckpt_hash,
            "timesteps":  f"{start}-{end}",
            "topk_mass":  topk_mass,
            "removed_sensor": f"Sensor_{removed_id}",
            "added_sensor":   f"Sensor_{added_id}",
            "added_sensor_z": float(added_z),
            "constrained_swap": {
                "faithfulness_score": c_score,
                "rationale":          c_rat,
                "orig_root_causes":   c_r_orig.get("root_cause_sensors", []) if not c_failed else [],
                "pert_root_causes":   c_r_pert.get("root_cause_sensors", []) if not c_failed else [],
            },
            "unconstrained_swap": {
                "faithfulness_score": u_score,
                "rationale":          u_rat,
            },
            "value_perturbation": {
                "faithfulness_score": v_score,
                "rationale":          v_rat,
                "category":           v_cat,
                "target_sensor":      f"Sensor_{v_target_id}",
                "new_z":              v_new_z,
                "action":             v_action,
                "pert_root_causes":   v_r_pert.get("root_cause_sensors", []) if not is_failed(v_r_pert) else [],
            }
        }
    
    import threading
        global file_lock
        if 'file_lock' not in globals():
            file_lock = threading.Lock()
        with file_lock:
            with open(OUTPUT_CACHE, "a") as fh:
                fh.write(json.dumps(result) + "\n")
    
        c_str = str(c_score) if c_score is not None else "FAIL"
        v_str = str(v_score) if v_score is not None else "FAIL"
        print(f"  -> Constrained swap: {c_str} | Value pert: {v_str}")
    
    print(f"\nDone. All results in {OUTPUT_CACHE}")
    

from concurrent.futures import ThreadPoolExecutor, as_completed
print(f"Running blind evaluation on {len(sampled)} segments with 4 workers...")
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(evaluate_segment, seg) for seg in sampled]
    for fut in as_completed(futures):
        fut.result()

print(f"\nDone. All results in {OUTPUT_CACHE}")
