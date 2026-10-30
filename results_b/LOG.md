# OPTION B — EXPERIMENT EXECUTION LOG

- **Run Started**: 2026-10-22T21:18:00+05:30
- **Hardware Profile**: 12 CPU Cores, Process Pool `N_WORKERS = 11`, `torch.set_num_threads(1)` per worker.
- **Environment**: Python 3.13 on Windows Host (PyTorch 2.10.0+cpu, PyG 2.8.1, scikit-learn 1.7.2).

---

## Stage S0 — Inventory, Config Freeze, Timing (GATE G0)
- **Timestamp**: 2026-10-22T21:20:57+05:30
- **S0.1 Machine & Segment Inventory**:
  - Machines verified: 28 (`machine-1-1` .. `machine-3-11`)
  - Labelled interpretation segments: Exactly 327 segments across all 28 machines.
  - File written: `results_b/inventory.csv`
- **S0.2 Configuration Freeze**:
  - Configuration frozen into `results_b/config_frozen.json`
  - SHA-256: `f09541fda4f2aac730456ceb0c954ca51750b343d6c42bce535cf2beae33ea95`
- **S0.3 Timing Benchmark**:
  - `machine-1-1` detector run time: 926.33s (15.44 min)
  - 5 IG attributions time: 2.19s (0.4389s / segment)
  - Projected FULL plan (84 runs across 11 workers): 2.04 hours
- **S0.4 Gate G0 Plan Selection**:
  - Rule: `total_hours_full <= 12 h` -> `Plan FULL` chosen automatically.
  - Coverage: Seeds {42, 43, 44} on all 28 machines.
  - Files written: `results_b/PLAN.md`, `results_b/plan_seeds.json`

## Stage S1 — Per-Machine Detectors Completed
- Timestamp: 2026-10-27T08:44:34.722767
- Total tasks evaluated: 113
- Wall-clock time: 11097.52s (3.08 hours)
- Failures: 0
- Outputs: `results_b/detection_per_machine.csv`, `results_b/detection_summary.csv`, `results_b/leakage_checks.csv`

## Stage S2 — Same-Split Baselines Completed
- Wall-clock: 253.45s
- Outputs: `results_b/baselines.csv`, `results_b/baselines_summary.csv`

## Stage S3 — Attribution Benchmark Completed
- Wall-clock: 283.77s
- Segments evaluated: 327 (IG, Attention) and 327 (GNNExplainer)
- Outputs: `results_b/attribution_per_segment.csv`, `results_b/attribution_summary.csv`

## Stage S4 — Conformal Calibration Completed
- Wall-clock: 1.22s
- Outputs: `results_b/conformal_summary.csv`, `results_b/conformal_loo_machine.csv`

## Stage S5 — Abstention-Signal Screening Completed
- Wall-clock: 171.33s
- Pre-registration: `results_b/s5_preregistration.json`
- Outputs: `results_b/s5_signal_screening.csv`, `results_b/S5_RESULT.md`

## Stage S6 — GATE G6 Triggered (Stopped)
- Timestamp: 2026-10-27T09:02:16.561699
- Reason: Error code: 410 - {'type': 'about:blank', 'title': 'Gone', 'status': 410, 'detail': "The model 'meta/llama-3.1-70b-instruct' has reached its end of life on 2026-08-26T09:00:00Z and is no longer available."}
- Action: Per Rule S6.4, stopped S6 execution without switching models.

## Stage S7 — Blind Human-Audit Sheet Generated
- Wall-clock: 0.02s
- Total audit cases: 15
- Outputs: `results_b/audit_blind.csv`, `results_b/audit_key.csv`

## Stage S8 — Paper Summary & Manifest Completed
- Wall-clock: 0.59s
- Outputs: `results_b/RESULTS_B_SUMMARY.md`, `results_b/manifest.json`

## Stage S6 & S7 Re-run with Active Reasoning Model (NIM LLM) — Completed
- Timestamp: 2026-10-27T12:23:52.470369
- Model: nvidia/nemotron-3-nano-omni-30b-a3b-reasoning (temperature=0.0)
- Config: results_b/final/config_frozen_v2.json
- Dynamic Conformal Calibration Statement: built dynamically at runtime from results_b/conformal_summary.csv (rho=0.8, alpha=0.10, held_out_mean=91.67%, SD=3.08%, q_hat=28/38) worded 'empirical held-out marginal coverage' without the word 'validated' and without hardcoded statistics.
- Pre-registered Segment Pool: 82 segments across all 28 machines written to results_b/final/s6_pool_keys.json before inference.
- API Calls: 164 fresh calls across 82 original and 82 counterfactually perturbed prompts; 0 cache hits; 0 failures.
- Diagnostics: 3 timestamped prompt/response pairs recorded in results_b/final/s6_cache_check.txt; 0 failures in results_b/final/s6_failures.md.
- Stage S6 Evaluation: results_b/final/faithfulness_results.csv, results_b/final/faithfulness_tau_sweep.csv (Wilson 95% CIs).
  - Tau = 0.00: Honesty = 98.8% [93.4%, 99.8%], Evidential Sensitivity = 2.4% [0.7%, 8.5%].
  - Tau = 0.70: Abstention = 40.2% [30.3%, 51.1%], Honesty = 98.0% [89.3%, 99.7%].
- Stage S7 Blind Human Audit: results_b/final/audit_blind.csv (30 records with human evaluation columns completely empty), results_b/final/audit_key.csv (unblinded verification key with 10 candidate unfaithful, 10 abstained gate, 10 random controls).
- Manifest & Summary: Updated results_b/final/RESULTS_B_SUMMARY.md and results_b/final/manifest.json with SHA-256 hashes of all 29 final artifacts.
