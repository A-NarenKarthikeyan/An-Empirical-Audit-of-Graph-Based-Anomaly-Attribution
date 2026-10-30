import csv, os, sys
from pathlib import Path
import numpy as np

# Paths
ATTR_PATH = str(Path(__file__).resolve().parents[1] / "results_b" / "attribution_per_segment.csv")

values = []
with open(ATTR_PATH, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row.get('method', '').strip() == 'Integrated_Gradients':
            try:
                top5 = float(row['top5_recall'])
                top1 = float(row['top1_hit'])
                legacy = float(row['legacy_overlap'])
                values.append((top5, top1, legacy))
            except Exception:
                continue
if not values:
    print('No IG rows found')
    sys.exit(1)
arr = np.array(values)
import pandas as pd

# Load the CSV with pandas
df = pd.read_csv(ATTR_PATH)
# Keep only Integrated Gradients rows
df = df[df["method"].str.strip() == "Integrated_Gradients"]
# Group by machine and compute mean of the three metrics per machine
m = df.groupby("machine")[["top5_recall", "top1_hit", "legacy_overlap"]].mean()
# Verify we have 28 machines
print(len(m))  # must print 28
# Compute the macro‑average across machines, convert to percentages, round to 2 dp
print((m.mean() * 100).round(2))

