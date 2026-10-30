import os
import sys
import json
import shutil
import pandas as pd
import numpy as np

# Ensure UTF-8 output encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
os.chdir(PROJECT_DIR)
sys.path.insert(0, PROJECT_DIR)

from src.data_loader import load_smd_raw

print("Creating human rater sheets and practice cases...")

# -----------------------------------------------------------------------------
# 1. Load audit_blind.csv and audit_key.csv
# -----------------------------------------------------------------------------
blind_path = os.path.join(PROJECT_DIR, "results_b/final/audit_blind.csv")
key_path = os.path.join(PROJECT_DIR, "results_b/final/audit_key.csv")

df_blind = pd.read_csv(blind_path)
df_key = pd.read_csv(key_path)

print(f"Loaded {len(df_blind)} records from {blind_path}")
print(f"Loaded {len(df_key)} records from {key_path}")

audit_gids = set(df_key["segment_global_id"].tolist())

# -----------------------------------------------------------------------------
# 2. Build 30 audit rows with the requested columns
# Columns: audit_id, prompt_text, llm_response, magnitudes_correct,
#          only_packet_sensors, no_exaggeration, overall_faithful, comment
# -----------------------------------------------------------------------------
columns_requested = [
    "audit_id",
    "prompt_text",
    "llm_response",
    "magnitudes_correct",
    "only_packet_sensors",
    "no_exaggeration",
    "overall_faithful",
    "comment"
]

audit_rows = []
for _, r in df_blind.iterrows():
    audit_rows.append({
        "audit_id": int(r["audit_id"]),
        "prompt_text": r["packet_text"],
        "llm_response": r["explanation_text"],
        "magnitudes_correct": "",
        "only_packet_sensors": "",
        "no_exaggeration": "",
        "overall_faithful": "",
        "comment": ""
    })

df_audit = pd.DataFrame(audit_rows, columns=columns_requested)

# -----------------------------------------------------------------------------
# 3. Shuffle row order once with seed 42
# -----------------------------------------------------------------------------
df_shuffled = df_audit.sample(frac=1.0, random_state=42).reset_index(drop=True)
print(f"Shuffled 30 audit cases with seed 42. Order of audit_ids:")
print(df_shuffled["audit_id"].tolist())

# -----------------------------------------------------------------------------
# 4. Generate practice_cases.csv with 3 extra cases NOT in the 30 audit cases
# -----------------------------------------------------------------------------
practice_gids = [2, 6, 14]
for gid in practice_gids:
    assert gid not in audit_gids, f"Segment {gid} is in audit cases!"

# Load conformal calibration stats for dynamic prompt template
conf_df = pd.read_csv(os.path.join(PROJECT_DIR, "results_b/conformal_summary.csv"))
row_target = conf_df[(conf_df["containment_rho"] == 0.8) & (np.isclose(conf_df["alpha"], 0.10))].iloc[0]
held_out_mean = float(row_target["held_out_coverage_mean"])
held_out_sd = float(row_target["held_out_coverage_sd"])
q_hat = int(row_target["full_pool_q_hat"])

dynamic_confidence_template = (
    f"Attribution Confidence Context:\n"
    f"  - Instance Concentration: These top-5 sensors account for {{cumulative_mass_pct:.2f}}% "
    f"of the model's total Integrated Gradients attribution signal.\n"
    f"  - Statistical Calibration: At 90% nominal coverage level (criterion rho = 0.80, alpha = 0.10), "
    f"the calibrated prediction set size is {q_hat} of 38 sensors, with empirical held-out marginal coverage "
    f"of {held_out_mean * 100:.2f}% (SD: {held_out_sd * 100:.2f}%)."
)

with open(os.path.join(PROJECT_DIR, "results_b/s3_cached_rankings.json"), "r", encoding="utf-8") as f:
    all_segments = json.load(f)

v3_cache = {}
v3_cache_path = os.path.join(PROJECT_DIR, "results_b/final/faithfulness_v3_cache.jsonl")
with open(v3_cache_path, "r", encoding="utf-8") as f:
    for line in f:
        entry = json.loads(line)
        v3_cache[entry["cache_key"]] = entry["response"]

machine_data = {}
practice_rows = []

for p_idx, gid in enumerate(practice_gids, start=1):
    seg = next(s for s in all_segments if s["segment_global_id"] == gid)
    m = seg["machine"]
    
    if m not in machine_data:
        tr, te, _, _ = load_smd_raw(os.path.join(PROJECT_DIR, "data/raw"), m)
        h_mean = np.mean(tr, axis=0)
        h_std = np.std(tr, axis=0)
        machine_data[m] = {"tr": tr, "te": te, "h_mean": h_mean, "h_std": h_std}
    
    m_data = machine_data[m]
    te = m_data["te"]
    h_mean = m_data["h_mean"]
    h_std = m_data["h_std"]
    
    # Identify top-5 sensors excluding h_std < 1e-3
    valid_sensor_indices = [i for i in range(38) if h_std[i] >= 1e-3]
    ig_scores = np.array(seg["ig_scores"])
    valid_ig = [(i + 1, abs(ig_scores[i])) for i in valid_sensor_indices]
    valid_ig.sort(key=lambda x: x[1], reverse=True)
    top5_sensors = [x[0] for x in valid_ig[:5]]
    
    abs_ig = np.abs(ig_scores)
    sum_ig = np.sum(abs_ig) if np.sum(abs_ig) > 0 else 1.0
    top5_mass = float(sum(abs_ig[s - 1] for s in top5_sensors) / sum_ig)
    
    s_idx = max(0, int(seg["start"] - 15))
    e_idx = min(len(te), int(seg["end"] + 15) + 1)
    fine_raw = te[s_idx:e_idx]
    fine_mean = np.mean(fine_raw, axis=0)
    fine_min = np.min(fine_raw, axis=0)
    fine_max = np.max(fine_raw, axis=0)
    
    h_std_clipped = np.clip(h_std, 1e-3, None)
    z_devs = (fine_mean - h_mean) / h_std_clipped
    
    conf_stmt = dynamic_confidence_template.format(cumulative_mass_pct=top5_mass * 100.0)
    
    sensor_evidence_records = []
    for rk, s_id in enumerate(top5_sensors, start=1):
        s_0 = s_id - 1
        z_s = z_devs[s_0]
        pct = (abs_ig[s_0] / sum_ig) * 100.0
        sensor_evidence_records.append({
            "sensor_id": s_id,
            "rank": rk,
            "ig_pct": pct,
            "observed_mean": fine_mean[s_0],
            "observed_range": [fine_min[s_0], fine_max[s_0]],
            "historical_mean": h_mean[s_0],
            "historical_std": h_std[s_0],
            "z_deviation": z_s
        })
    
    struct_records = [dict(r) for r in sensor_evidence_records]
    for field in ["observed_mean", "observed_range", "historical_mean", "historical_std", "z_deviation"]:
        struct_records[0][field], struct_records[1][field] = struct_records[1][field], struct_records[0][field]
    
    ev_lines_struct = []
    for r in struct_records:
        z_sign = "+" if r["z_deviation"] >= 0 else ""
        ev_lines_struct.append(
            f"  - Sensor_{r['sensor_id']} (Rank #{r['rank']}): IG Score = {r['ig_pct']:.2f}% | "
            f"Observed Window Mean = {r['observed_mean']:.4f} (range: [{r['observed_range'][0]:.4f}, {r['observed_range'][1]:.4f}]) | "
            f"Normal Historical Baseline = {r['historical_mean']:.4f} +/- {r['historical_std']:.4f} | "
            f"Z-Score Shift = {z_sign}{r['z_deviation']:.2f} std"
        )
    ev_text_struct = "\n".join(ev_lines_struct)
    
    prompt_struct = (
        f"=== ANOMALY DIAGNOSTIC REQUEST ===\n"
        f"Anomaly Segment ID: {gid}\n"
        f"Active Timesteps: {seg['start']} to {seg['end']} (Duration: {seg['end'] - seg['start'] + 1} steps)\n\n"
        f"=== EMPIRICAL EVIDENCE (Top-5 Attributed Sensors) ===\n"
        f"{ev_text_struct}\n\n"
        f"=== STATISTICAL CONFIDENCE CONTEXT ===\n"
        f"{conf_stmt}\n\n"
        f"=== CRITICAL INSTRUCTIONS ===\n"
        f"1. From the 5 sensors listed in EMPIRICAL EVIDENCE, identify and output ONLY the sensor IDs that you judge to be the true root causes of the anomaly based on their empirical deviations.\n"
        f"2. DO NOT include sensors that show normal or near-baseline behavior. If only 1 or 2 sensors are driving the anomaly, list ONLY those.\n"
        f"3. DO NOT mention, infer, or hallucinate any sensor IDs outside this set of 5.\n"
        f"4. State the confidence level exactly as given in the STATISTICAL CONFIDENCE CONTEXT.\n"
        f"5. Return your diagnostic strictly as a valid JSON object matching the schema below.\n\n"
        f"=== OUTPUT FORMAT (JSON ONLY) ===\n"
        f"{{\n"
        f'  "root_cause_sensors": [list of integer sensor IDs that you judge to be actual root causes],\n'
        f'  "explanation": "concise technical rationale explaining why the cited sensors are root causes based strictly on the observed z-score shifts",\n'
        f'  "confidence_statement": "exact replication of statistical confidence statement from above"\n'
        f"}}\n"
    )
    
    metrics_p = os.path.join(PROJECT_DIR, f"results_b/metrics/{m}_seed42.json")
    with open(metrics_p, "r", encoding="utf-8") as fp:
        m_meta = json.load(fp)
    ckpt_h = m_meta["checkpoint_sha256"][:8]
    cache_k = f"{ckpt_h}_{m}_{gid}_struct_v3"
    assert cache_k in v3_cache, f"Missing cache key {cache_k}"
    
    raw_resp = v3_cache[cache_k]
    parsed = json.loads(raw_resp)
    exp_text = parsed.get("explanation", "")
    assert len(exp_text) > 0, f"Empty explanation for {cache_k}"
    
    practice_rows.append({
        "audit_id": p_idx,
        "prompt_text": prompt_struct,
        "llm_response": exp_text,
        "magnitudes_correct": "",
        "only_packet_sensors": "",
        "no_exaggeration": "",
        "overall_faithful": "",
        "comment": ""
    })

df_practice = pd.DataFrame(practice_rows, columns=columns_requested)

# -----------------------------------------------------------------------------
# 5. Save outputs
# -----------------------------------------------------------------------------
os.makedirs(os.path.join(PROJECT_DIR, "results_c"), exist_ok=True)
path_sheet_A = os.path.join(PROJECT_DIR, "results_c/rater_sheet_A.csv")
path_sheet_B = os.path.join(PROJECT_DIR, "results_c/rater_sheet_B.csv")
path_practice = os.path.join(PROJECT_DIR, "results_c/practice_cases.csv")

df_shuffled.to_csv(path_sheet_A, index=False)
df_shuffled.to_csv(path_sheet_B, index=False)
df_practice.to_csv(path_practice, index=False)

# Also write / mirror to root results_c/ if different
root_results_c = os.path.abspath(os.path.join(PROJECT_DIR, "..", "results_c"))
if os.path.exists(root_results_c) and os.path.abspath(root_results_c) != os.path.abspath(os.path.join(PROJECT_DIR, "results_c")):
    shutil.copy(path_sheet_A, os.path.join(root_results_c, "rater_sheet_A.csv"))
    shutil.copy(path_sheet_B, os.path.join(root_results_c, "rater_sheet_B.csv"))
    shutil.copy(path_practice, os.path.join(root_results_c, "practice_cases.csv"))

print(f"Successfully generated files:")
print(f"  - rater_sheet_A.csv: {path_sheet_A} ({len(df_shuffled)} rows)")
print(f"  - rater_sheet_B.csv: {path_sheet_B} ({len(df_shuffled)} rows)")
print(f"  - practice_cases.csv: {path_practice} ({len(df_practice)} rows)")

# -----------------------------------------------------------------------------
# 6. Append entry to LOG.md
# -----------------------------------------------------------------------------
log_path = os.path.join(PROJECT_DIR, "results_c/LOG.md")
with open(log_path, "a", encoding="utf-8") as f_log:
    f_log.write("\n## Human Rater Sheets Generation 2026-10-29\n")
    f_log.write(f"- Source: results_b/final/audit_blind.csv (30 cases) shuffled once with seed 42\n")
    f_log.write(f"- Shuffled audit_ids: {df_shuffled['audit_id'].tolist()}\n")
    f_log.write(f"- Columns: {columns_requested}\n")
    f_log.write(f"- Output Sheet A: results_c/rater_sheet_A.csv ({len(df_shuffled)} rows)\n")
    f_log.write(f"- Output Sheet B: results_c/rater_sheet_B.csv ({len(df_shuffled)} rows)\n")
    f_log.write(f"- Practice Cases: results_c/practice_cases.csv ({len(df_practice)} rows from non-audit segments {practice_gids})\n")

print("Updated LOG.md successfully.")
