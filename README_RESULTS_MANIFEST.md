# Complete Results Portfolio & Audit Manifest
**Project**: Autonomous Incident Root-Cause Explanations via Twin-GNN Attribution & Conformal Filtering
**Archive Created**: 2026-10-18 12:53:04
**Target Output**: Q1 Journal Publication Portfolio

---

## 1. Executive Summary of Key Results

| Component | Metric / Finding | Value / 95% Confidence Interval |
|---|---|---|
| **SMD 6-D Test Performance** | Precision / Recall / F1 / AUROC | **P: 96.47%, R: 99.01%, F1: 97.73%, AUROC: 99.63%** |
| | Confusion Matrix (Test $N=4,257$) | $TP=301, FP=11, FN=3, TN=3,942$ at val-optimal $\tau=0.84$ |
| | Leak-Free Guarantee | Contiguous 70/15/15 temporal split (Zero train/val/test leakage) |
| **2-D Baseline Comparison** | Feature set: [mean, std] only | **P: 30.58%, R: 99.67%, F1: 46.80%, AUROC: 99.75%** |
| | Degradation vs 6-D | FP explodes from 11 -> 688; validates necessity of 6-D spatio-temporal features |
| **Attribution Accuracy** | Integrated Gradients (Top-1) | **71.40% [62.78%, 79.64%]** (10,000 bootstrap replicates) |
| | GNNExplainer | 40.37% [29.93%, 52.06%] |
| | Raw Attention Weights | 25.15% [12.85%, 38.93%] |
| **Pooled Conformal Calibration** | 100% Target Coverage | **91.44% [88.38%, 94.50%]** ($N=327$ segments, 28 machines) |
| | 80% Target Coverage | **90.83% [87.46%, 93.88%]** |
| **Optimal Abstention (\tau=0.80)**| Wilson Score 95% CIs ($N=39$) | Abstention: **17.95% [8.93%, 32.74%]** |
| | Individual Honesty | **94.87% [83.11%, 98.58%]** |
| | Evidential Sensitivity | **97.44% [86.82%, 99.55%]** |
| | Aggregate Consistency | **82.05% [67.31%, 91.02%]** |

---

## 2. Master Archive Directory Manifest

### A. Publication Manuscript & Visuals
- `results/Q1_Paper_Final_Results_and_Plots.docx`: Complete publication document containing all formatted tables, Wilson score CIs, bootstrap CIs, 2-D comparison, and embedded high-resolution figures.
- `results/abstention_tradeoff_v3.png`: High-resolution primary tradeoff figure showing Abstention Rate, Individual Honesty, Aggregate Consistency, and Evidential Sensitivity across $\tau \in [0.00, 0.95]$.
- `results/abstention_tradeoff_v2.png`: Tradeoff plot variant with dual-axis layout.
- `results/abstention_tradeoff.png`: Initial tradeoff visualization.

### B. Machine Learning Checkpoints (.pt)
- `results/gnn_checkpoint.pt`: Leak-free trained 6-D Spatio-Temporal GAT model weights (Best Val Loss: 0.0526).
- `results/gnn_checkpoint_2d.pt`: Leak-free retrained 2-D baseline GAT model weights (Best Val Loss: 0.1593).
- `results/psm_gnn_checkpoint.pt`: PSM benchmark dataset trained GAT model weights.

### C. Detection & Attribution Benchmark Tables (.csv & .json)
- `results/smd_test_metrics.json`: Detailed SMD 6-D test metrics, optimal $\tau$, confusion matrix, and split metadata.
- `results/smd_2d_test_metrics.json`: SMD 2-D baseline test metrics evaluated at val-optimal $\tau=0.61$ and standard $\tau=0.50$.
- `results/detection_baseline_table.csv`: Master baseline comparison table (SMD 6-D, 2-D, PSM, Isolation Forest, Deep SVDD, OmniAnomaly, USAD, Anomaly Transformer).
- `results/detection_table_footnote.md`: Methodological footnote clarifying threshold tuning policy and contiguous split.
- `results/tau_sweep_wilson_ci.csv`: Comprehensive $\tau$-sweep table ($	au=0.00$ to $0.95$) with exact Wilson score 95% lower/upper bounds.
- `results/tau_sweep_wilson_ci.json`: Structured JSON format of the $\tau$-sweep with Wilson CIs.
- `results/attribution_and_calibration_bootstrap.json`: Seed-42 rerun attribution Top-1 accuracies and conformal calibration coverage with 10,000 bootstrap replicates and 95% empirical CIs.
- `results/additional_audit_stats.json`: 1,000-split held-out conformal coverage (mean ± SD) and Spearman correlation between Top-5 IG mass and Top-5 GT recall with bootstrap CIs.
- `results/attribution_method_comparison.csv`: Attribution comparison table (IG vs GNNExplainer vs Attention).
- `results/calibration_summary.csv`: Machine-level and pooled conformal calibration coverage summary.
- `results/ablation_table.csv`: Component ablation table (Full Pipeline, No Conformal Filter, No GAT Attribution, etc.).
- `results/faithfulness_summary.csv`: Automated LLM explanation faithfulness scorer summary.
- `results/psm_detection_results.csv`: PSM detection benchmark test results.

### D. Qualitative & Audit Portfolios
- `results/faithfulness_20_audit_explanations.json`: Full 20-case audit portfolio containing input telemetry, GNN attribution, conformal mass, full LLM explanation, and automated scorer reasoning.
- `results/faithfulness_20_audit_explanations.md`: Formatted Markdown audit document for human review and committee auditing.
- `results/methodology_note.md`: Full derivation and composition of the $N=39$ evaluation pool (25 Stage-6 + 18 blind stratified - 4 overlap = 39 unique segments).
- `results/config2_qualitative_note.md`: Detailed qualitative analysis of telemetry configuration.
- `results/example_cases.json`: Representative normal, subtle anomaly, and severe anomaly telemetry explanation cases.
- `results/_pool_segment_keys.json`: Machine-segment IDs defining the standardized $N=39$ evaluation pool.

### E. LLM & Evaluation Caches (.jsonl)
- `results/faithfulness_eval_cache.jsonl`: Cached LLM generation and evaluation outputs on the verified leak-free model.
- `results/threshold_blind_eval_cache.jsonl`: Blind sample evaluation cache.
- `results/faithfulness_eval_cache_leaked_checkpoint_archive.jsonl`: Archived historical evaluation cache for audit traceability.
- `results/threshold_blind_eval_cache_leaked_checkpoint_archive.jsonl`: Archived historical blind cache.
- `results/llm_cache_leaked_checkpoint_archive.jsonl`: Archived LLM query cache.

### F. Verification Logs & Reproducibility Traces
- `verification_logs/smd_train_true.log`: Execution log confirming leak-free 70/15/15 contiguous split training.
- `verification_logs/psm_train.log`: PSM training log.
- `verification_logs/leakage_check.txt`: Mathematical boundary check confirming 0 overlapping timestamps.
- `verification_logs/check_split.txt`: Split confirmation details.
- `verification_logs/pooled_conformal_out.txt`: Full terminal output from the seed-42 conformal calibration and bootstrap rerun.
- `verification_logs/sweep_out.txt`: Raw terminal output from the $\tau$-sweep simulation across all 39 segments.
