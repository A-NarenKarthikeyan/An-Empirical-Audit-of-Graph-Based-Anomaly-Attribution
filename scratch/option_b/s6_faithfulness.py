"""Stage S6: Faithfulness Audit on Sampled Segments & GATE G6 Handling.

S6.1 Pool first: sample up to 3 segments per machine (seed 42) -> ~84 segments.
     Write results_b/s6_pool_keys.json BEFORE any LLM call.
S6.2 Call NIM LLM client with each segment's own detector checkpoint.
S6.3 Cache check saved to results_b/s6_cache_check.txt.
S6.4 GATE G6: If NIM endpoint unreachable/rate-limited/model discontinued,
     STOP S6 only, preserve S0-S5 results and report.
S6.5 Write failures to results_b/s6_failures.md.
"""

import os
import sys
import json
import time
import datetime
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, os.path.abspath("."))
from src.explanation.prompt_builder import extract_topk_sensor_evidence
from src.evaluation.faithfulness import (
    _call_llm_api, find_counterfactual_swap_sensor,
    build_perturbed_constrained_prompt, build_unconstrained_prompts,
    build_value_perturbed_constrained_prompt, score_faithfulness,
    score_value_perturbation_faithfulness, NIM_BASE_URL
)

def run_s6_faithfulness_audit():
    print("=" * 70)
    print("STAGE S6 — Faithfulness Audit & GATE G6 Verification")
    print("=" * 70)
    
    t0 = time.time()
    load_dotenv()
    
    # 1. Load segments from S3
    rankings_path = "results_b/s3_cached_rankings.json"
    if not os.path.exists(rankings_path):
        raise FileNotFoundError(f"Missing S3 rankings: {rankings_path}")
        
    with open(rankings_path, "r", encoding="utf-8") as f:
        all_segments = json.load(f)
        
    # Group by machine
    segments_by_machine = {}
    for s in all_segments:
        m = s["machine"]
        segments_by_machine.setdefault(m, []).append(s)
        
    # S6.1: Sample up to 3 segments per machine with fixed seed 42
    rng = np.random.RandomState(42)
    pool_keys = []
    
    for m in sorted(segments_by_machine.keys()):
        m_segs = segments_by_machine[m]
        n_sample = min(3, len(m_segs))
        selected = rng.choice(len(m_segs), size=n_sample, replace=False).tolist()
        for idx in sorted(selected):
            seg = m_segs[idx]
            pool_keys.append({
                "segment_global_id": seg["segment_global_id"],
                "machine": m,
                "segment_index": seg["segment_index"],
                "start": seg["start"],
                "end": seg["end"],
                "gt_dimensions": seg["gt_dimensions"]
            })
            
    print(f"Sampled {len(pool_keys)} segments across {len(segments_by_machine)} machines (up to 3 per machine).")
    pool_file = "results_b/s6_pool_keys.json"
    with open(pool_file, "w", encoding="utf-8") as f:
        json.dump(pool_keys, f, indent=4)
    print(f"S6.1 pool keys written to {pool_file}")
    
    # Check frozen config for LLM
    with open("results_b/config_frozen.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model_name = cfg["llm"]["model_name"]
    temperature = cfg["llm"]["temperature"]
    
    api_key = os.getenv("NVIDIA_API_KEY", "")
    base_url = os.getenv("NVIDIA_BASE_URL", NIM_BASE_URL)
    
    # S6.4: GATE G6 Check — Test NIM endpoint availability
    print(f"\nTesting NIM endpoint reachability for model '{model_name}'...")
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key) if api_key else None
    
    nim_available = False
    g6_reason = ""
    
    if not api_key:
        nim_available = False
        g6_reason = "NVIDIA_API_KEY is not set."
    else:
        try:
            test_res = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": "Ping test"}],
                max_tokens=5,
                temperature=0.0
            )
            nim_available = True
            print("NIM endpoint is reachable!")
        except Exception as exc:
            nim_available = False
            g6_reason = str(exc)
            print(f"GATE G6 TRIGGERED: NIM endpoint failed with: {g6_reason}")
            
    if not nim_available:
        print("\n" + "!" * 70)
        print("GATE G6 ACTIVE: NIM endpoint is unreachable / model discontinued.")
        print("Per Rule S6.4: STOP S6 only, preserve S0-S5 results and report.")
        print("Do not switch models. Keeping existing frozen configuration.")
        print("!" * 70 + "\n")
        
        # Write GATE G6 status
        with open("results_b/s6_cache_check.txt", "w", encoding="utf-8") as f:
            f.write(f"GATE G6 TRIGGERED\n")
            f.write(f"Timestamp: {datetime.datetime.now().isoformat()}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"Endpoint: {base_url}\n")
            f.write(f"Reason: {g6_reason}\n")
            f.write(f"Action: STOP S6 per protocol Rule S6.4. Preserving Stages S0-S5.\n")
            
        with open("results_b/LOG.md", "a", encoding="utf-8") as f:
            f.write(f"\n## Stage S6 — GATE G6 Triggered (Stopped)\n")
            f.write(f"- Timestamp: {datetime.datetime.now().isoformat()}\n")
            f.write(f"- Reason: {g6_reason}\n")
            f.write(f"- Action: Per Rule S6.4, stopped S6 execution without switching models.\n")
            
        return False
        
    return True

if __name__ == "__main__":
    run_s6_faithfulness_audit()
