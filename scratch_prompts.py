import sys, os, yaml, torch
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.data_loader import load_dataset
from src.evaluation.faithfulness import build_unconstrained_prompts

cfg = {"fine_window_margin": 15}
mach = "machine-1-1"
seg_id = 1
removed_id = 30
added_id = 1

_, _, interp_labels, meta = load_dataset(dataset_name="SMD", raw_dir="data/raw", machine_id=mach, window_size=100, stride=1, corr_threshold=0.6)
seg = interp_labels[seg_id - 1]
start, end = seg["start"], seg["end"]
start_idx = max(0, int(start - cfg["fine_window_margin"]))
end_idx = min(len(meta["norm_test"]), int(end + cfg["fine_window_margin"]) + 1)
fine_raw_win = meta["raw_test"][start_idx:end_idx]

u_orig, u_pert = build_unconstrained_prompts(
    anomaly_id=seg_id, timesteps=(start, end), fine_raw_window=fine_raw_win,
    norm_stats=meta["norm_stats"], weakest_sensor_id=removed_id, replacement_sensor_id=added_id
)

print("="*60)
print(f"ORIGINAL UNCONSTRAINED (Machine {mach} Seg {seg_id})")
print("="*60)
print(u_orig["prompt"][:1000])
print("...\n[Truncated]\n...")
# Find the specific sensors in the prompt to highlight
lines_orig = u_orig["prompt"].splitlines()
for l in lines_orig:
    if "Sensor_30:" in l or "Sensor_1:" in l:
        print(l)

print("\n"+"="*60)
print(f"PERTURBED UNCONSTRAINED (Machine {mach} Seg {seg_id})")
print("="*60)
print(u_pert["prompt"][:1000])
print("...\n[Truncated]\n...")
lines_pert = u_pert["prompt"].splitlines()
for l in lines_pert:
    if "Sensor_30:" in l or "Sensor_1:" in l:
        print(l)
