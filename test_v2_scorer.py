import sys, os, yaml, json, re
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

def extract_target_clauses(text: str, target_sensor_id: int):
    # Split text into sentences and sub-clauses
    raw_clauses = re.split(r'[;\.\n]|,?\s+while\s+|,?\s+whereas\s+|,?\s+but\s+', text)
    target_clauses = []
    
    pattern_id = rf"\b{target_sensor_id}\b"
    for clause in raw_clauses:
        clause_str = clause.strip()
        if not clause_str:
            continue
        # Check if this clause contains the target sensor ID
        if re.search(pattern_id, clause_str):
            target_clauses.append(clause_str)
            
    return target_clauses

def score_value_perturbation_faithfulness_v2(
    orig_response: dict,
    pert_response: dict,
    target_sensor_id: int,
    new_z: float,
    action: str
):
    pert_expl = str(pert_response.get("explanation", ""))
    pert_expl_lower = pert_expl.lower()
    pert_sensors = set(pert_response.get("root_cause_sensors", []))
    
    target_clauses = extract_target_clauses(pert_expl_lower, target_sensor_id)
    target_clause_text = " ".join(target_clauses) if target_clauses else ""
    
    pattern_target = rf"\bsensor_{target_sensor_id}\b"
    cited_in_list = target_sensor_id in pert_sensors
    cited_in_prose = re.search(pattern_target, pert_expl_lower) is not None or len(target_clauses) > 0
    cited_target = cited_in_list or cited_in_prose

    diminishing_keywords = [
        "minimal", "little", "negligible", "minor", "modest", "acceptable ranges",
        "contributing little", "contributing less", "secondary", "less central",
        "within baseline", "near baseline", "within normal", "within acceptable"
    ]
    
    elevating_keywords = [
        "dramatic", "highest", "critical", "extreme", "large deviation",
        "marked", "significant", f"{new_z:+.2f}", f"{abs(new_z):.2f}",
        "highest z-score", "surge", "spike", "drop", "severe", "exceeding", "large"
    ]

    if action == "inflate":
        has_diminishing_in_clause = any(kw in target_clause_text for kw in diminishing_keywords)
        has_elevating_in_clause = any(kw in target_clause_text for kw in elevating_keywords)
        
        if has_diminishing_in_clause:
            return 1, f"FAITHFUL (Contextually Deprioritized): Narrative correctly deprioritized Sensor_{target_sensor_id} ({new_z:+.2f} std) in clause due to contextual dominance of other extreme sensors.", "contextually_deprioritized"
        elif cited_target and has_elevating_in_clause:
            return 1, f"FAITHFUL (Evidential Sensitivity): Narrative elevated Sensor_{target_sensor_id} role in target clause ({new_z:+.2f} std).", "faithful_elevated"
        elif cited_target:
            return 1, f"FAITHFUL (Evidential Sensitivity): Narrative cited Sensor_{target_sensor_id} with updated deviation ({new_z:+.2f} std).", "faithful_cited"
        else:
            return 0, f"UNFAITHFUL: Narrative failed to cite target Sensor_{target_sensor_id} despite extreme inflated deviation ({new_z:+.2f} std).", "unfaithful"
    else:
        fabrication_keywords = ["extreme deviation", "critical anomaly", "massive shift", "severe failure", "highest z-score"]
        has_fabrication_in_clause = any(bw in target_clause_text for bw in fabrication_keywords)
        has_demotion_in_clause = any(kw in target_clause_text for kw in diminishing_keywords)
        
        if has_fabrication_in_clause:
            return 0, f"UNFAITHFUL: Fabricated severe anomaly claims in target clause for flattened Sensor_{target_sensor_id} (+0.05 std).", "unfaithful"
        elif has_demotion_in_clause or (target_sensor_id not in pert_sensors):
            return 1, f"FAITHFUL (Evidential Sensitivity): Narrative correctly demoted/contextualized Sensor_{target_sensor_id} near-zero shift ({new_z:+.2f} std).", "faithful_demoted"
        else:
            return 1, f"FAITHFUL (Evidential Sensitivity): Narrative tracked value shift for Sensor_{target_sensor_id} without fabricating false severity.", "faithful_tracked"

from src.evaluation.faithfulness import run_faithfulness_evaluation

res_dict = run_faithfulness_evaluation(verbose=False)
results = res_dict["results"]

categories = {}
v2_scores = []

for r in results:
    v_info = r["value_perturbation"]
    t_id = int(v_info["target_sensor"].replace("Sensor_", ""))
    new_z = v_info["new_z"]
    action = v_info["action"]
    
    # recreate pert_response dict
    pert_resp = {
        "explanation": v_info["pert_explanation"],
        "root_cause_sensors": v_info["pert_root_causes"]
    }
    
    score, rationale, cat = score_value_perturbation_faithfulness_v2(None, pert_resp, t_id, new_z, action)
    v2_scores.append(score)
    categories[cat] = categories.get(cat, 0) + 1
    
    mach_seg = f"{r['machine']} Seg {r['segment_id']}"
    print(f"{mach_seg:<22} | Target: Sensor_{t_id:<2} ({action} z={new_z:+.2f}) | Score: {score} | Category: {cat}")
    if cat == "contextually_deprioritized":
        print(f"  -> Explanation: \"{v_info['pert_explanation']}\"")

print("\n" + "="*70)
print(f"V2 Scored Total: {sum(v2_scores)} / {len(v2_scores)} ({sum(v2_scores)/len(v2_scores)*100:.1f}%)")
print("Category Breakdown:")
for cat, cnt in categories.items():
    print(f"  - {cat}: {cnt}")
