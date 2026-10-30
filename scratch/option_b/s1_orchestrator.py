"""Stage S1 Orchestrator: Multi-worker process pool execution of all detectors."""

import os
import sys
import json
import time
import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, os.path.abspath("."))
from scratch.option_b.s1_detectors import run_single_detector_task, build_detection_summary

# Force unbuffered standard output
sys.stdout.reconfigure(line_buffering=True)

def main():
    print("=" * 70)
    print("STAGE S1 — Per-Machine Detectors Orchestrator")
    print("=" * 70)
    
    with open("results_b/plan_seeds.json", "r", encoding="utf-8") as f:
        plan_info = json.load(f)
        
    seeds_per_machine = plan_info["seeds_per_machine"]
    env_workers = os.getenv("N_WORKERS")
    n_workers = int(env_workers) if env_workers else plan_info.get("n_workers", max(1, (os.cpu_count() or 4) - 1))
    
    print(f"Plan: {plan_info['plan']}, N_WORKERS: {n_workers}")
    
    tasks = []
    
    # 1. Purged 6D detectors for all (machine, seed) in plan
    for m, seeds in seeds_per_machine.items():
        for s in seeds:
            tasks.append({
                "machine": m,
                "seed": s,
                "split_mode": "purged",
                "ablation_2d": False
            })
            
    # 2. Legacy split on machine-1-1 (seed 42) per S1.2
    tasks.append({
        "machine": "machine-1-1",
        "seed": 42,
        "split_mode": "legacy",
        "ablation_2d": False
    })
    
    # 3. 2-D ablation for seed 42 on all 28 machines per S1.6
    all_machines = list(seeds_per_machine.keys())
    for m in all_machines:
        tasks.append({
            "machine": m,
            "seed": 42,
            "split_mode": "purged",
            "ablation_2d": True
        })
        
    print(f"Total detector tasks queued: {len(tasks)}")
    
    # Track results and failures
    results = []
    failed_tasks = []
    
    # Run tasks with ProcessPoolExecutor
    t_start = time.time()
    
    # Check already completed tasks first
    pending_tasks = []
    for t in tasks:
        m = t["machine"]
        s = t["seed"]
        s_mode = f"_{t['split_mode']}" if t['split_mode'] == "legacy" else ""
        abl = "_2d" if t.get("ablation_2d") else ""
        run_id = f"{m}_seed{s}{s_mode}{abl}"
        m_file = f"results_b/metrics/{run_id}.json"
        c_file = f"results_b/checkpoints/{run_id}.pt"
        
        if os.path.exists(m_file) and os.path.exists(c_file):
            try:
                with open(m_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                results.append(data)
                continue
            except Exception:
                pass
        pending_tasks.append(t)
        
    print(f"Already completed: {len(results)} tasks. Pending execution: {len(pending_tasks)} tasks.")
    
    if len(pending_tasks) > 0:
        print(f"\nLaunching {len(pending_tasks)} tasks on {n_workers} workers...")
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            future_to_task = {executor.submit(run_single_detector_task, t): t for t in pending_tasks}
            
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                run_id = f"{task['machine']}_seed{task['seed']}_{task['split_mode']}{'_2d' if task.get('ablation_2d') else ''}"
                try:
                    res = future.result()
                    results.append(res)
                    print(f"[{len(results)}/{len(tasks)}] Done {res['run_id']} | Val-Opt={res['val_optimal_threshold']:.2f} | Test F1={res['test_metrics']['f1']:.4f} | AUROC={res['test_metrics']['auroc']:.4f} | AUPRC={res['test_metrics']['auprc']:.4f} in {res['wall_clock_seconds']:.1f}s", flush=True)
                except Exception as exc:
                    print(f"ERROR: Task {run_id} failed with: {exc}. Retrying once identically per Rule 3...")
                    # Retry once identically
                    try:
                        res = run_single_detector_task(task)
                        results.append(res)
                        print(f"Task {run_id} succeeded on retry!")
                    except Exception as retry_exc:
                        print(f"CRITICAL: Task {run_id} failed on retry: {retry_exc}. Logging failure per Rule 3.")
                        failed_tasks.append({
                            "run_id": run_id,
                            "error": str(retry_exc),
                            "timestamp": datetime.datetime.now().isoformat()
                        })
                        
    total_time = time.time() - t_start
    print(f"\nAll detector tasks completed in {total_time:.2f}s ({total_time / 3600.0:.2f} hours)!")
    
    # Log any failures to results_b/LOG.md
    if failed_tasks:
        with open("results_b/LOG.md", "a", encoding="utf-8") as f:
            f.write("\n### S1 Failures\n")
            for ft in failed_tasks:
                f.write(f"- `{ft['run_id']}`: failed at {ft['timestamp']} with error: `{ft['error']}`\n")
                
    # Build aggregates and summaries
    build_detection_summary(results)
    
    # Append to LOG.md
    with open("results_b/LOG.md", "a", encoding="utf-8") as f:
        f.write(f"\n## Stage S1 — Per-Machine Detectors Completed\n")
        f.write(f"- Timestamp: {datetime.datetime.now().isoformat()}\n")
        f.write(f"- Total tasks evaluated: {len(results)}\n")
        f.write(f"- Wall-clock time: {total_time:.2f}s ({total_time / 3600.0:.2f} hours)\n")
        f.write(f"- Failures: {len(failed_tasks)}\n")
        f.write(f"- Outputs: `results_b/detection_per_machine.csv`, `results_b/detection_summary.csv`, `results_b/leakage_checks.csv`\n")

if __name__ == "__main__":
    main()
