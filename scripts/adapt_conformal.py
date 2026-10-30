import json
import csv
import os
from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RANKINGS_PATH = str(PROJECT_ROOT / "results_b" / "s3_cached_rankings.json")
RESULTS_V2_PATH = str(PROJECT_ROOT / "results_b" / "final" / "faithfulness_results_v2.csv")
OUTPUT_DIR = str(PROJECT_ROOT / "results_c")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "adapt_conformal_results.csv")
LOG_PATH = os.path.join(OUTPUT_DIR, "LOG.md")

# Load rankings (list of dicts)
with open(RANKINGS_PATH, "r", encoding="utf-8") as f:
    rankings = json.load(f)

# Load target sensors per segment
segment_to_target = {}
with open(RESULTS_V2_PATH, newline="", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        try:
            seg_id = int(row["segment_global_id"].strip())
            target = row["target_sensor"].strip()
            if target == "":
                continue
            # target_sensor may be a float string, convert to int sensor id
            target = int(float(target))
            # Some rows may have missing target (e.g., not_testable). Skip those.
            if target == "":
                continue
            # target_sensor is a float but represents an integer sensor id.
            segment_to_target[seg_id] = int(float(target))
        except Exception:
            continue

# Compute normalized set sizes
N_SENSORS = 38
normalized_sizes = []
for entry in rankings:
    seg_id = entry.get("segment_global_id")
    if seg_id not in segment_to_target:
        # No ground‑truth target for this segment; skip.
        continue
    target = segment_to_target[seg_id]
    ranking = entry.get("ig_ranking_1indexed", [])
    # ranking is 1‑indexed list of sensor ids.
    try:
        position = ranking.index(target) + 1  # convert to 1‑based rank
    except ValueError:
        # Target not found in ranking – treat as worst case (set size > 38)
        position = N_SENSORS + 1
    normalized = position / N_SENSORS
    normalized_sizes.append(normalized)

if not normalized_sizes:
    raise RuntimeError("No segment data found to compute adaptive conformal metrics.")

total = len(normalized_sizes)
# Alpha thresholds
alphas = [0.1, 0.2, 0.3]
rows = []
for alpha in alphas:
    threshold = 1 - alpha  # coverage criterion
    covered = sum(1 for ns in normalized_sizes if ns <= threshold)
    coverage = covered / total
    mean_set = sum(normalized_sizes) / total
    rows.append({"alpha": alpha, "coverage": coverage, "mean_set_fraction": mean_set})

# Write CSV
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["alpha", "coverage", "mean_set_fraction"])
    writer.writeheader()
    for r in rows:
        writer.writerow({"alpha": f"{r['alpha']:.2f}", "coverage": f"{r['coverage']:.4f}", "mean_set_fraction": f"{r['mean_set_fraction']:.4f}"})

# Append to LOG.md
timestamp = datetime.now().isoformat()
log_entry = f"## Adaptive Conformal Set Computation\n\n- Timestamp: {timestamp}\n- Input rankings: {os.path.basename(RANKINGS_PATH)}\n- Target sensors file: {os.path.basename(RESULTS_V2_PATH)}\n- Output CSV: {os.path.basename(OUTPUT_CSV)}\n- Alphas evaluated: {', '.join(map(str, alphas))}\n\n" 
with open(LOG_PATH, "a", encoding="utf-8") as log_f:
    log_f.write(log_entry)

print("Adaptive conformal metrics computed and saved to", OUTPUT_CSV)
