"""Stage S7: Blind Human-Audit Sheet & Verification Key.

Exports 30 explanations from evaluation pool:
- Segments marked unfaithful, filled up to 30 with random segments (seed 42),
  including cases where the tau = 0.70 gate would abstain.
Outputs:
- results_b/audit_blind.csv: audit_id, evidence packet, explanation text,
  and EMPTY columns: human_individual_honest, human_aggregate_consistent, human_evidence_faithful.
- results_b/audit_key.csv: audit_id, machine, segment_index, scorer verdicts, tau_070_abstains.
"""

import os
import sys
import json
import time
from typing import Dict, Any, List
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.abspath("."))

def run_s7_human_audit():
    print("=" * 70)
    print("STAGE S7 — Blind Human-Audit Sheet Generation")
    print("=" * 70)
    
    t0 = time.time()
    
    # Source candidate explanation records from existing evaluation cache
    cache_sources = [
        "results/threshold_blind_eval_cache.jsonl",
        "results/faithfulness_eval_cache.jsonl"
    ]
    
    records = []
    seen_keys = set()
    
    for cpath in cache_sources:
        if os.path.exists(cpath):
            with open(cpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                        k = (d.get("machine"), d.get("segment_index"))
                        if k not in seen_keys:
                            seen_keys.add(k)
                            records.append(d)
                    except Exception:
                        pass
                        
    print(f"Loaded {len(records)} candidate evaluation records with explanations.")
    
    # Identify unfaithful cases
    unfaithful_cases = []
    faithful_cases = []
    
    for r in records:
        # Check if any scorer marked unfaithful
        sc = r.get("scores", {})
        ind_honest = sc.get("individual_honest", True)
        agg_consistent = sc.get("aggregate_consistent", True)
        ev_faithful = sc.get("evidence_faithful", True)
        
        is_unfaithful = not (ind_honest and agg_consistent and ev_faithful)
        if is_unfaithful:
            unfaithful_cases.append(r)
        else:
            faithful_cases.append(r)
            
    print(f"Found {len(unfaithful_cases)} unfaithful cases and {len(faithful_cases)} faithful cases.")
    
    # Select up to 30: all unfaithful + fill with random faithful (seed 42)
    rng = np.random.RandomState(42)
    n_needed = max(0, 30 - len(unfaithful_cases))
    
    if len(faithful_cases) >= n_needed:
        selected_faithful_idx = rng.choice(len(faithful_cases), size=n_needed, replace=False).tolist()
        fill_cases = [faithful_cases[i] for i in selected_faithful_idx]
    else:
        fill_cases = faithful_cases
        
    final_30 = unfaithful_cases + fill_cases
    # Shuffle order so blind auditor cannot guess position
    final_30 = [final_30[i] for i in rng.permutation(len(final_30))]
    
    blind_rows = []
    key_rows = []
    
    for audit_id, item in enumerate(final_30, 1):
        m = item.get("machine", "unknown")
        s_idx = item.get("segment_index", 0)
        ev_packet = item.get("evidence_packet", item.get("prompt", "Evidence Packet"))
        exp_text = item.get("explanation", item.get("response", "Explanation text"))
        
        sc = item.get("scores", {})
        top5_mass = item.get("top5_mass", 0.0)
        tau_070_abstains = bool(top5_mass < 0.70)
        
        blind_rows.append({
            "audit_id": audit_id,
            "evidence_packet": ev_packet,
            "explanation_text": exp_text,
            "human_individual_honest": "",
            "human_aggregate_consistent": "",
            "human_evidence_faithful": ""
        })
        
        key_rows.append({
            "audit_id": audit_id,
            "machine": m,
            "segment_index": s_idx,
            "top5_mass": top5_mass,
            "tau_070_abstains": tau_070_abstains,
            "scorer_individual_honest": sc.get("individual_honest"),
            "scorer_aggregate_consistent": sc.get("aggregate_consistent"),
            "scorer_evidence_faithful": sc.get("evidence_faithful"),
            "overall_scorer_faithful": bool(sc.get("individual_honest", True) and sc.get("aggregate_consistent", True) and sc.get("evidence_faithful", True))
        })
        
    df_blind = pd.DataFrame(blind_rows)
    df_key = pd.DataFrame(key_rows)
    
    blind_csv = "results_b/audit_blind.csv"
    key_csv = "results_b/audit_key.csv"
    
    df_blind.to_csv(blind_csv, index=False)
    df_key.to_csv(key_csv, index=False)
    
    print(f"Saved {len(df_blind)} blind audit cases to {blind_csv}")
    print(f"Saved audit key to {key_csv}")
    
    elapsed = time.time() - t0
    print(f"Stage S7 completed in {elapsed:.2f}s")
    
    with open("results_b/LOG.md", "a", encoding="utf-8") as f:
        f.write(f"\n## Stage S7 — Blind Human-Audit Sheet Generated\n")
        f.write(f"- Wall-clock: {elapsed:.2f}s\n")
        f.write(f"- Total audit cases: {len(df_blind)}\n")
        f.write(f"- Outputs: `results_b/audit_blind.csv`, `results_b/audit_key.csv`\n")

if __name__ == "__main__":
    run_s7_human_audit()
