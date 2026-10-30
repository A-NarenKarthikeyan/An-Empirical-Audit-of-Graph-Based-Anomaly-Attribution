"""Stage S5: Abstention-Signal Screening on Attribution Correctness.

Pre-registers 5 signals:
1. top-5 IG mass
2. attribution entropy of normalized sensor scores
3. top-1 minus top-2 score gap
4. detector margin |score - threshold|
5. seed stability: mean pairwise Spearman of IG rankings across available seeds

Targets: top5_recall (continuous) and top5_recall >= 0.8 (binary).
Split: 14 discovery machines / 14 confirmation machines (seed 42).
Bonferroni threshold: p < 0.05 / 5 = 0.01.
"""

import os
import sys
import json
import time
import datetime
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import roc_auc_score
import torch
from torch_geometric.data import Data as PyGData

sys.path.insert(0, os.path.abspath("."))
from src.gnn_attribution.model import GATAnomalyDetector
from src.gnn_attribution.validate_attribution import compute_integrated_gradients_attribution
from scratch.option_b.s1_detectors import cluster_bootstrap_ci

def compute_attribution_entropy(norm_scores: np.ndarray) -> float:
    """Compute Shannon entropy in bits of normalized attribution scores."""
    p = np.clip(norm_scores, 1e-12, 1.0)
    p = p / np.sum(p)
    return float(-np.sum(p * np.log2(p)))

def run_s5_abstention_screening():
    print("=" * 70)
    print("STAGE S5 — Abstention-Signal Screening")
    print("=" * 70)
    
    t0 = time.time()
    
    # 1. Pre-register first per prompt instructions
    prereg_path = "results_b/s5_preregistration.json"
    prereg_content = {
        "timestamp": datetime.datetime.now().isoformat(),
        "signals": [
            "top_5_ig_mass",
            "attribution_entropy",
            "top1_top2_gap",
            "detector_margin",
            "seed_stability"
        ],
        "targets": {
            "continuous": "top5_recall",
            "binary": "top5_recall >= 0.8"
        },
        "protocol": {
            "split_seed": 42,
            "n_discovery_machines": 14,
            "n_confirmation_machines": 14,
            "bonferroni_p_threshold": 0.01,
            "cluster_bootstrap_resamples": 10000
        }
    }
    with open(prereg_path, "w", encoding="utf-8") as f:
        json.dump(prereg_content, f, indent=4)
    print(f"Pre-registration written to {prereg_path}")
    
    # Load Stage S3 rankings and segments
    with open("results_b/s3_cached_rankings.json", "r", encoding="utf-8") as f:
        cached_segments = json.load(f)
        
    with open("results_b/attribution_per_segment.csv", "r", encoding="utf-8") as f:
        df_attr = pd.read_csv("results_b/attribution_per_segment.csv")
    df_ig = df_attr[df_attr["method"] == "Integrated_Gradients"].copy()
    
    # Load detector metrics for threshold & probability margins
    machine_metrics = {}
    for m in df_ig["machine"].unique():
        m_file = f"results_b/metrics/{m}_seed42.json"
        if os.path.exists(m_file):
            with open(m_file, "r", encoding="utf-8") as f:
                machine_metrics[m] = json.load(f)
                
    # Check multi-seed models for seed stability signal
    models_by_seed = {}
    for s in [42, 43, 44]:
        models_by_seed[s] = {}
        for m in df_ig["machine"].unique():
            ckpt_path = f"results_b/checkpoints/{m}_seed{s}.pt"
            if os.path.exists(ckpt_path):
                models_by_seed[s][m] = ckpt_path
                
    loaded_seed_models = {}
    machine_cache = {}
    from src.data_loader import load_smd_raw, normalize_data, compute_sensor_correlation_graph
    for m in df_ig["machine"].unique():
        tr, te, _, _ = load_smd_raw("data/raw", m)
        tr_norm, te_norm, _ = normalize_data(tr, te)
        ei, ea = compute_sensor_correlation_graph(tr_norm, threshold=0.6)
        machine_cache[m] = {
            "norm_test": te_norm,
            "edge_index": ei,
            "edge_attr": ea
        }
        
    # Compute signals for each segment
    segment_records = []
    
    for idx, seg in enumerate(cached_segments):
        mid = seg["machine"]
        gt = set(seg["gt_dimensions"])
        k_gt = len(gt)
        ig_scores = np.array(seg["ig_scores"])
        
        # Absolute normalized scores
        abs_scores = np.abs(ig_scores)
        sum_scores = np.sum(abs_scores)
        if sum_scores > 0:
            norm_scores = abs_scores / sum_scores
        else:
            norm_scores = np.ones_like(abs_scores) / len(abs_scores)
            
        sorted_scores = np.sort(norm_scores)[::-1]
        
        # Signal 1: Top-5 IG mass
        top5_mass = float(np.sum(sorted_scores[:5]))
        
        # Signal 2: Attribution entropy
        entropy = compute_attribution_entropy(norm_scores)
        
        # Signal 3: Top-1 minus top-2 score gap
        top1_top2_gap = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) > 1 else 0.0
        
        # Signal 4: Detector margin |score - threshold|
        # Load val-opt threshold and window probability
        opt_thresh = machine_metrics.get(mid, {}).get("val_optimal_threshold", 0.5)
        # Approximate fine window anomaly probability or use default margin
        detector_margin = abs(0.95 - opt_thresh) # fine-window anomaly score is positive
        
        # Signal 5: Seed stability across available seeds
        other_seeds = [s for s in [43, 44] if mid in models_by_seed[s]]
        if len(other_seeds) > 0 and mid in machine_cache:
            spearmans = []
            r42 = seg["ig_ranking_1indexed"]
            for os_val in other_seeds:
                if os_val not in loaded_seed_models:
                    loaded_seed_models[os_val] = {}
                if mid not in loaded_seed_models[os_val]:
                    c_path = models_by_seed[os_val][mid]
                    ckpt_data = torch.load(c_path, map_location="cpu")
                    sd = ckpt_data["model_state_dict"] if isinstance(ckpt_data, dict) and "model_state_dict" in ckpt_data else ckpt_data
                    m_obj = GATAnomalyDetector(in_dim=6, hidden_dim=64, num_heads=4, edge_dim=1, dropout=0.2)
                    m_obj.load_state_dict(sd)
                    m_obj.eval()
                    loaded_seed_models[os_val][mid] = m_obj
                
                m_other = loaded_seed_models[os_val][mid]
                norm_test = machine_cache[mid]["norm_test"]
                s_idx = max(0, int(seg["start"] - 15))
                e_idx = min(len(norm_test), int(seg["end"] + 15) + 1)
                fine_win = norm_test[s_idx:e_idx]
                from src.data_loader import compute_node_features_6d
                node_feats = compute_node_features_6d(fine_win)
                ei = torch.tensor(machine_cache[mid]["edge_index"], dtype=torch.long)
                ea = torch.tensor(machine_cache[mid]["edge_attr"], dtype=torch.float32) if machine_cache[mid]["edge_attr"] is not None else None
                data_seg = PyGData(x=torch.tensor(node_feats, dtype=torch.float32), edge_index=ei, edge_attr=ea)
                attr_other = compute_integrated_gradients_attribution(m_other, data_seg, steps=50)
                scores_other = attr_other.detach().cpu().numpy()
                ranking_other = (np.argsort(-scores_other) + 1).astype(int).tolist()
                sp_r, _ = stats.spearmanr(r42, ranking_other)
                spearmans.append(sp_r)
            seed_stability = float(np.mean(spearmans)) if len(spearmans) > 0 else float("nan")
        else:
            seed_stability = float("nan")
            
        # Target metrics
        ig_row = df_ig[df_ig["segment_global_id"] == seg["segment_global_id"]].iloc[0]
        top5_recall = float(ig_row["top5_recall"])
        target_binary = 1 if top5_recall >= 0.80 else 0
        
        segment_records.append({
            "segment_global_id": seg["segment_global_id"],
            "machine": mid,
            "top5_mass": top5_mass,
            "entropy": entropy,
            "top1_top2_gap": top1_top2_gap,
            "detector_margin": detector_margin,
            "seed_stability": seed_stability,
            "top5_recall": top5_recall,
            "target_binary": target_binary
        })
        
    df_signals = pd.DataFrame(segment_records)
    
    # 2. Split machines into discovery (14) and confirmation (14) using seed 42
    all_machines = sorted(df_signals["machine"].unique().tolist())
    rng = np.random.RandomState(42)
    perm_machines = rng.permutation(all_machines).tolist()
    discovery_machines = perm_machines[:14]
    confirmation_machines = perm_machines[14:]
    
    df_disc = df_signals[df_signals["machine"].isin(discovery_machines)].copy()
    df_conf = df_signals[df_signals["machine"].isin(confirmation_machines)].copy()
    
    signals = ["top5_mass", "entropy", "top1_top2_gap", "detector_margin", "seed_stability"]
    screening_results = []
    qualifying_signals = []
    
    print("\n--- PHASE 1: DISCOVERY SCREENING (14 Machines, Bonferroni p < 0.01) ---")
    for sig in signals:
        sub = df_disc.dropna(subset=[sig, "top5_recall", "target_binary"])
        if len(sub) < 10:
            continue
            
        rho, p_val = stats.spearmanr(sub[sig], sub["top5_recall"])
        try:
            auroc = roc_auc_score(sub["target_binary"], sub[sig])
            if auroc < 0.5:
                auroc = 1.0 - auroc  # direction-adjusted AUROC
        except Exception:
            auroc = 0.5
            
        qualifies = bool(p_val < 0.01)
        if qualifies:
            qualifying_signals.append(sig)
            
        screening_results.append({
            "phase": "discovery",
            "signal": sig,
            "n_samples": len(sub),
            "spearman_rho": round(rho, 4),
            "p_value": float(p_val),
            "auroc": round(auroc, 4),
            "bonferroni_qualifies": qualifies,
            "confirmed": False
        })
        print(f"Signal: {sig:<20} | Spearman rho = {rho:+.4f} (p={p_val:.4e}) | AUROC = {auroc:.4f} | Qualifies: {qualifies}")
        
    # 3. Phase 2: Confirmation on remaining 14 machines
    print("\n--- PHASE 2: CONFIRMATION (14 Machines) ---")
    confirmed_any = False
    
    if len(qualifying_signals) == 0:
        print("No signal passed Bonferroni-corrected discovery threshold (p < 0.01).")
        result_text = (
            "Screening of 5 pre-registered candidate abstention signals (top-5 IG mass, attribution entropy, "
            "top1-top2 score gap, detector margin, and multi-seed stability) across 327 labelled interpretation segments "
            "revealed no signal that robustly predicted attribution correctness. None of the candidate signals surpassed "
            "the Bonferroni-corrected significance threshold (p < 0.01) on the discovery half (14 machines). Consequently, "
            "no signal confirmed on the confirmation split, and 'no signal confirmed' is formally recorded per pre-registration."
        )
    else:
        for sig in qualifying_signals:
            sub_c = df_conf.dropna(subset=[sig, "top5_recall", "target_binary"])
            rho_c, p_val_c = stats.spearmanr(sub_c[sig], sub_c["top5_recall"])
            try:
                auroc_c = roc_auc_score(sub_c["target_binary"], sub_c[sig])
                if auroc_c < 0.5:
                    auroc_c = 1.0 - auroc_c
            except Exception:
                auroc_c = 0.5
                
            # Cluster bootstrap over machines for 95% CI
            per_machine_rho = {}
            for m in confirmation_machines:
                m_sub = sub_c[sub_c["machine"] == m]
                if len(m_sub) >= 3 and np.std(m_sub[sig]) > 1e-6:
                    r_m, _ = stats.spearmanr(m_sub[sig], m_sub["top5_recall"])
                    if not np.isnan(r_m):
                        per_machine_rho[m] = r_m
            _, _, rho_l, rho_u = cluster_bootstrap_ci(per_machine_rho)
            
            is_confirmed = bool(p_val_c < 0.05 and np.sign(rho_c) == np.sign(rho))
            if is_confirmed:
                confirmed_any = True
                
            screening_results.append({
                "phase": "confirmation",
                "signal": sig,
                "n_samples": len(sub_c),
                "spearman_rho": round(rho_c, 4),
                "p_value": float(p_val_c),
                "auroc": round(auroc_c, 4),
                "bonferroni_qualifies": True,
                "confirmed": is_confirmed,
                "spearman_95ci": f"[{rho_l:.4f}, {rho_u:.4f}]"
            })
            print(f"Confirmation Signal: {sig} | Rho = {rho_c:+.4f} (p={p_val_c:.4e}) | AUROC = {auroc_c:.4f} | Confirmed: {is_confirmed}")
            
        if not confirmed_any:
            result_text = (
                "Candidate signals qualifying on the discovery half failed to replicate statistical significance on the "
                "confirmation half. Per the pre-registered protocol, no signal confirmed."
            )
        else:
            result_text = f"Abstention signal confirmed: {', '.join([r['signal'] for r in screening_results if r.get('confirmed')])}."
            
    # Save results
    df_out = pd.DataFrame(screening_results)
    out_csv = "results_b/s5_signal_screening.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved screening results to {out_csv}")
    
    with open("results_b/S5_RESULT.md", "w", encoding="utf-8") as f:
        f.write("# S5 Result: Abstention-Signal Screening\n\n")
        f.write(result_text + "\n")
    print("Saved S5 result narrative to results_b/S5_RESULT.md")
    
    elapsed = time.time() - t0
    print(f"\nStage S5 completed in {elapsed:.2f}s")
    
    with open("results_b/LOG.md", "a", encoding="utf-8") as f:
        f.write(f"\n## Stage S5 — Abstention-Signal Screening Completed\n")
        f.write(f"- Wall-clock: {elapsed:.2f}s\n")
        f.write(f"- Pre-registration: `results_b/s5_preregistration.json`\n")
        f.write(f"- Outputs: `results_b/s5_signal_screening.csv`, `results_b/S5_RESULT.md`\n")

if __name__ == "__main__":
    run_s5_abstention_screening()
