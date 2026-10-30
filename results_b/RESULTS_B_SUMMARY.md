# OPTION B — FULL 28-MACHINE BENCHMARK STUDY RESULTS

**Frozen Config SHA-256**: `f09541fda4f2aac730456ceb0c954ca51750b343d6c42bce535cf2beae33ea95`  
**Machines Evaluated**: 28 (`machine-1-1` .. `machine-3-11`)  
**Labelled Segments Evaluated**: 327  

---

## Table 2: Multi-Machine Anomaly Detection Benchmark

| Method / Configuration | N (Non-Deg / Total) | Macro F1 (95% CI) | Macro Precision | Macro Recall | Macro AUROC | Macro AUPRC | Event Recall | Oracle F1 | Micro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **6D_GAT_Purged_Seed42** | 19 / 28 | 51.36% ± 30.15% [0.3793, 0.6442] | 50.47% | 64.22% | 86.09% | 76.85% | 75.00% | 76.65% | 55.64% |
| **6D_GAT_Purged_MultiSeed** | 19 / 28 | 56.12% ± 22.10% [0.4615, 0.6554] | 57.05% | 69.71% | 88.17% | 77.61% | 82.46% | 77.06% | 57.38% |
| **2D_GAT_Purged_Seed42** | 19 / 28 | 46.82% ± 33.50% [0.3228, 0.6134] | 49.00% | 59.50% | 86.49% | 68.32% | 65.79% | 67.55% | 54.51% |
| **6D_GAT_Legacy_m11** | 1 / 1 | 89.85% ± 0.00% [0.8985, 0.8985] | 82.24% | 99.01% | 99.64% | 99.27% | 100.00% | 99.50% | 89.85% |
| Baseline: Isolation_Forest | 28 / 28 | 44.77% ± 26.72% [0.3296, 0.5618] | 46.42% | 58.25% | 91.64% | 61.09% | 79.39% | 66.22% | 55.32% |
| Baseline: Max_Z_Score | 28 / 28 | 48.53% ± 31.39% [0.3452, 0.6262] | 53.54% | 58.17% | 91.44% | 74.77% | 64.91% | 77.87% | 60.49% |

*Note: Degenerate machines (zero anomalous windows in val or test) are reported individually and excluded from macro averages.*  

## Table 3: Root-Cause Sensor Attribution (N=327 Segments, Cluster-Bootstrap 95% CIs)

| Attribution Method | N Segments | Top-5 Recall (95% CI) | Top-1 Hit Rate (95% CI) | Legacy Overlap (95% CI) | Paired Diff vs IG (Top-5 Recall) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Integrated_Gradients** | 327 | 53.93% ± 17.66% [0.4759, 0.6035] | 83.89% ± 15.70% [0.7798, 0.8953] | 62.85% ± 12.12% [0.5842, 0.6740] | reference |
| **Layer2_Attention** | 327 | 12.99% ± 7.37% [0.1048, 0.1588] | 33.72% ± 22.58% [0.2609, 0.4242] | 21.12% ± 16.13% [0.1614, 0.2782] | +0.4094 [+0.3284, +0.4879] |
| **GNNExplainer** | 327 | 40.80% ± 15.86% [0.3515, 0.4659] | 57.73% ± 22.60% [0.4920, 0.6565] | 46.87% ± 13.71% [0.4214, 0.5211] | +0.1313 [+0.0978, +0.1653] |


## Table 4: Conformal Attribution Calibration (Pooled 1,000 Splits & LOO-Machine)

| Containment Target (rho) | Nominal Coverage (1 - alpha) | Full Pool q-hat (sensors) | q-hat Fraction of 38 | Mean Min Prefix | Held-Out Coverage (Mean ± SD) | Held-Out 95% Range |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 100% | 90% | 36 | 94.7% | 15.81 | 92.47% ± 3.06% | [0.8648, 0.9846] |
| 100% | 80% | 31 | 81.6% | 15.81 | 81.71% ± 4.33% | [0.7322, 0.9020] |
| 80% | 90% | 28 | 73.7% | 12.68 | 91.67% ± 3.08% | [0.8564, 0.9771] |
| 80% | 80% | 24 | 63.2% | 12.68 | 81.96% ± 4.49% | [0.7316, 0.9076] |


## Table 5: Pre-Registered Abstention-Signal Screening

# S5 Result: Abstention-Signal Screening

Candidate signals qualifying on the discovery half failed to replicate statistical significance on the confirmation half. Per the pre-registered protocol, no signal confirmed.


| Phase | Signal | N | Spearman rho | p-value | AUROC | Bonferroni Qualifies | Confirmed |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| discovery | top5_mass | 174 | +0.2032 | 0.0072 | 0.6233 | True | False |
| discovery | entropy | 174 | -0.1555 | 0.0405 | 0.5729 | False | False |
| discovery | top1_top2_gap | 174 | +0.1546 | 0.0416 | 0.5485 | False | False |
| discovery | detector_margin | 174 | +0.1396 | 0.0661 | 0.5992 | False | False |
| discovery | seed_stability | 174 | +0.0552 | 0.4695 | 0.5559 | False | False |
| confirmation | top5_mass | 153 | -0.0502 | 0.5375 | 0.5168 | True | False |


## Table 6: Faithfulness Audit Status & GATE G6 Verification

```
GATE G6 TRIGGERED
Timestamp: 2026-10-27T09:02:16.559346
Model: meta/llama-3.1-70b-instruct
Endpoint: https://integrate.api.nvidia.com/v1
Reason: Error code: 410 - {'type': 'about:blank', 'title': 'Gone', 'status': 410, 'detail': "The model 'meta/llama-3.1-70b-instruct' has reached its end of life on 2026-08-26T09:00:00Z and is no longer available."}
Action: STOP S6 per protocol Rule S6.4. Preserving Stages S0-S5.

```



## Machine Integrity & Quality Exclusions

- **Degenerate Machines** (zero validation or test anomalies): `machine-1-5, machine-1-7, machine-2-8, machine-3-1, machine-3-2, machine-3-4, machine-3-5, machine-3-10, machine-3-9`

- **Leakage Failures (`leak_fail`)**: `None (0 overlap timesteps across all machines under purged split)`

- **Failed Runs**: 0 (all tasks completed successfully or cleanly handled per Rule 3)
