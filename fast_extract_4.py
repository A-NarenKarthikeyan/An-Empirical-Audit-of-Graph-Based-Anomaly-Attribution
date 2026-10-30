import sys, os, yaml, json, re
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.evaluation.faithfulness import run_faithfulness_evaluation

# Run evaluation harness
res_dict = run_faithfulness_evaluation(verbose=False)
results = res_dict["results"]

inflate_cases = [r for r in results if r["value_perturbation"]["action"] == "inflate" and r["value_perturbation"]["faithfulness_score"] == 1]

# Select 4 across different machines
selected = []
seen_machines = set()
for r in inflate_cases:
    m = r["machine"]
    if m not in seen_machines:
        seen_machines.add(m)
        selected.append(r)
    if len(selected) == 4:
        break

if len(selected) < 4:
    selected = inflate_cases[:4]

for i, r in enumerate(selected):
    print("\n" + "="*90)
    print(f"CASE {i+1}: {r['machine']} Segment {r['segment_id']} (Timesteps: {r['timesteps']})")
    print(f"Target Sensor: {r['value_perturbation']['target_sensor']} | Action: Inflate to z={r['value_perturbation']['new_z']:+.2f} std")
    print("="*90)
    
    v_info = r["value_perturbation"]
    t_id = int(v_info["target_sensor"].replace("Sensor_", ""))
    new_z = v_info["new_z"]
    expl = v_info["pert_explanation"]
    root_causes = v_info["pert_root_causes"]
    
    print("\n[1] ORIGINAL UNPERTURBED EVIDENCE (Top-5 Evidence Set):")
    for s in r["constrained_swap"]["orig_root_causes"]:
        print(f"  - Sensor_{s}")
    print("  (Full prompt provides IG % attribution, observed means, and baseline ranges for all top-5 sensors)")
    
    print(f"\n[2] PERTURBED EVIDENCE GIVEN TO MODEL:")
    print(f"  - {v_info['target_sensor']} Z-Score artificially inflated from baseline shift to: {new_z:+.2f} std")
    print(f"  - All other Top-5 sensors retained their exact original baseline statistics.")
    
    print("\n[3] FULL MODEL EXPLANATION (Raw Unabridged Text):")
    try:
        print(expl)
    except:
        print(expl.encode('cp1252', errors='replace').decode('cp1252'))
        
    print(f"\nModel Output root_cause_sensors JSON: {root_causes}")
    
    # Matching details
    pert_expl_lower = str(expl).lower()
    pert_sensors_set = set(root_causes)
    pattern_target = rf"\bsensor_{t_id}\b"
    
    in_list = t_id in pert_sensors_set
    in_prose = re.search(pattern_target, pert_expl_lower) is not None
    elevate_keywords = ["dramatic", "highest", "critical", "extreme", "large deviation", "marked", "significant", f"{new_z:+.2f}", f"{abs(new_z):.2f}", "highest z-score", "surge", "spike", "drop"]
    matched_kws = [kw for kw in elevate_keywords if kw in pert_expl_lower]
    
    print("\n[4] EXACT SCORING / KEYWORD MATCH DETAILS:")
    print(f"- Scored Verdict: {v_info['rationale']}")
    print(f"- Sensor_{t_id} present in root_cause_sensors JSON list? {in_list}")
    print(f"- Sensor_{t_id} referenced in narrative prose? {in_prose}")
    print(f"- Matched elevating keyword(s) in prose: {matched_kws}")
    if in_list and in_prose:
        print("- Match Scope: BOTH (Present in JSON list AND cited with elevating rationale in narrative prose)")
    elif in_list:
        print("- Match Scope: JSON LIST ONLY")
    elif in_prose:
        print("- Match Scope: PROSE ONLY")
