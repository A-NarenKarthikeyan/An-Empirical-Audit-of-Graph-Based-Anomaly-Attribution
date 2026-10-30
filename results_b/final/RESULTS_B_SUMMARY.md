# OPTION B — FULL 28-MACHINE BENCHMARK STUDY RESULTS (FINAL CONSOLIDATED)

**Frozen Config SHA-256**: `a68dde35d9e5d86479e5a8bd8bef34c5b564d761d8e746c023238e3c07aa17ae`  
**LLM Model Evaluated**: `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning`  
**Machines Evaluated**: 28 (`machine-1-1` .. `machine-3-11`)  
**Labelled Ground-Truth Interpretation Segments**: 327  
**Total Detector Runs Audited for Leakage**: 85 (84 pass, 1 intentional legacy leakage run)  
**Total Test Anomaly Events**: 54 (across 9,263 anomalous test windows)  

---

## Table 2: Multi-Machine Anomaly Detection Benchmark (19 Non-Degenerate Machines)

| Method / Configuration | N (Non-Deg / Total) | Macro F1 (95% CI) | Macro Precision | Macro Recall | Macro AUROC | Macro AUPRC | Event Recall | Oracle F1 | Micro F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **6D_GAT_Purged_Seed42** | 19 / 28 | 51.36% ± 30.15% [0.3793, 0.6442] | 50.47% | 64.22% | 86.09% | 76.85% | 75.00% | 76.65% | 55.64% |
| **6D_GAT_Purged_MultiSeed** | 19 / 28 | 56.12% ± 22.10% [0.4615, 0.6554] | 57.05% | 69.71% | 88.17% | 77.61% | 82.46% | 77.06% | 57.38% |
| **2D_GAT_Purged_Seed42** | 19 / 28 | 46.82% ± 33.50% [0.3228, 0.6134] | 49.00% | 59.50% | 86.49% | 68.32% | 65.79% | 67.55% | 54.51% |
| **6D_GAT_Legacy_m11** | 1 / 1 | 89.85% ± 0.00% [0.8985, 0.8985] | 82.24% | 99.01% | 99.64% | 99.27% | 100.00% | 99.50% | 89.85% |
| Baseline: Isolation_Forest | 19 / 28 | 44.77% ± 26.72% [0.3296, 0.5618] | 46.42% | 58.25% | 91.64% | 61.09% | 79.39% | 66.22% | 55.32% |
| Baseline: Max_Z_Score | 19 / 28 | 48.53% ± 31.39% [0.3452, 0.6262] | 53.54% | 58.17% | 91.44% | 74.77% | 64.91% | 77.87% | 60.49% |

### Paired Cluster-Bootstrap Differences (GAT-6D Seed 42 vs Baselines, 10,000 Resamples)

| Comparison | Metric | Mean Diff | 95% Cluster-Bootstrap CI |
| :--- | :--- | :---: | :---: |
| GAT-6D minus Max-z | f1 | +0.0283 | [-0.1408, +0.2020] |
| GAT-6D minus Max-z | auroc | -0.0534 | [-0.1567, +0.0366] |
| GAT-6D minus Max-z | auprc | +0.0208 | [-0.1387, +0.1866] |
| GAT-6D minus Max-z | oracle_f1 | -0.0123 | [-0.1505, +0.1303] |
| GAT-6D minus Isolation Forest | f1 | +0.0659 | [-0.1139, +0.2653] |
| GAT-6D minus Isolation Forest | auroc | -0.0555 | [-0.1594, +0.0219] |
| GAT-6D minus Isolation Forest | auprc | +0.1576 | [-0.0054, +0.3247] |
| GAT-6D minus Isolation Forest | oracle_f1 | +0.1042 | [-0.0402, +0.2406] |
| GAT-6D minus GAT-2D | f1 | +0.0454 | [-0.1299, +0.2179] |
| GAT-6D minus GAT-2D | auroc | -0.0039 | [-0.0659, +0.0627] |
| GAT-6D minus GAT-2D | auprc | +0.0853 | [-0.0228, +0.2249] |
| GAT-6D minus GAT-2D | oracle_f1 | +0.0910 | [-0.0170, +0.2243] |

---

## Table 3: Root-Cause Sensor Attribution (N=327 Segments)

| Method | Top-1 Hit Rate (95% CI) | Top-5 Recall (95% CI) | Legacy Overlap (95% CI) | Diff vs IG Top-5 Recall | Diff vs IG Top-1 Hit |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Integrated_Gradients** | 83.89% [0.7798, 0.8953] | 53.93% [0.4759, 0.6035] | 62.85% [0.5842, 0.6740] | reference | reference |
| **Layer2_Attention** | 33.72% [0.2609, 0.4242] | 12.99% [0.1048, 0.1588] | 21.12% [0.1614, 0.2782] | +0.4094 [+0.3284, +0.4879] | +0.5017 [+0.3955, +0.6018] |
| **GNNExplainer** | 57.73% [0.4920, 0.6565] | 40.80% [0.3515, 0.4659] | 46.87% [0.4214, 0.5211] | +0.1313 [+0.0978, +0.1653] | +0.2616 [+0.1895, +0.3464] |

### Attribution Performance by Detector Block (Integrated Gradients, Cluster-Bootstrap 95% CIs)

| Detector Block | N Segments | Top-1 Hit Rate | 95% CI | Top-5 Recall | 95% CI |
| :--- | :---: | :---: | :---: | :---: | :---: |
| train | 211 | 88.15% | [0.7926, 0.9316] | 61.49% | [0.4617, 0.6216] |
| validation | 56 | 85.71% | [0.8130, 0.9739] | 54.12% | [0.4281, 0.6097] |
| test | 54 | 75.93% | [0.6875, 0.9132] | 64.09% | [0.4896, 0.7210] |
| straddles_boundary | 6 | 16.67% | [0.0000, 0.7500] | 30.16% | [0.1682, 0.6161] |
| ALL | 327 | 84.40% | [0.7804, 0.8953] | 60.08% | [0.4762, 0.6031] |

---

## Table 4: Conformal Attribution Calibration

| Target Containment (rho) | Nominal Coverage (1-alpha) | Calibrated q_hat (out of 38) | Fraction of Sensors | Pooled Held-Out Coverage Mean ± SD | Pooled 95% CI |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 1.00 | 90.0% | 36 | 94.7% | 92.47% ± 3.06% | [0.8648, 0.9846] |
| 1.00 | 80.0% | 31 | 81.6% | 81.71% ± 4.33% | [0.7322, 0.9020] |
| 0.80 | 90.0% | 28 | 73.7% | 91.67% ± 3.08% | [0.8564, 0.9771] |
| 0.80 | 80.0% | 24 | 63.2% | 81.96% ± 4.49% | [0.7316, 0.9076] |

---

## Table 5: Pre-Registered Abstention-Signal Screening (Corrected Cluster-Bootstrap CIs)

| Phase | Signal | N Samples | Spearman rho (95% CI) | p-value | AUROC (95% CI) | Qualifies / Confirmed |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| discovery | top5_mass | 174 | +0.2032 [-0.1109, +0.4010] | 7.1771e-03 | 0.6233 [0.4209, 0.7359] | Qualifies (p < 0.01) |
| discovery | entropy | 174 | -0.1555 [-0.3673, +0.1369] | 4.0470e-02 | 0.5729 [0.3792, 0.7143] | Not Confirmed |
| discovery | top1_top2_gap | 174 | +0.1546 [-0.0942, +0.3716] | 4.1627e-02 | 0.5485 [0.3654, 0.7077] | Not Confirmed |
| discovery | detector_margin | 174 | +0.1396 [-0.0785, +0.3993] | 6.6120e-02 | 0.5992 [0.4628, 0.7301] | Not Confirmed |
| discovery | seed_stability | 174 | +0.0620 [-0.0751, +0.2835] | 4.1613e-01 | 0.5664 [0.4920, 0.6873] | Not Confirmed |
| confirmation | top5_mass | 153 | -0.0502 [-0.3109, +0.2495] | 5.3746e-01 | 0.5168 [0.3660, 0.6569] | Qualifies (p < 0.01) |
| all_327 | top5_mass | 327 | +0.0945 [-0.1382, +0.2825] | 8.8033e-02 | 0.5579 [0.4239, 0.6653] | Qualifies (p < 0.01) |

> [!NOTE]
> Confirmation CI for `top5_mass` was corrected from `[-0.0875, 0.3932]` to `[-0.3109, 0.2495]`. See `results_b/final/s5_ci_explanation.md` for full derivation.

---

## Table 6: LLM Faithfulness Audit & Selective Abstention Sweep (N=82 Segments, Wilson 95% CIs)

| Tau Threshold | N Evaluated | Abstained Rate (95% CI) | Structural Honesty (95% CI) | Evidential Sensitivity (95% CI) | Overall Faithfulness (95% CI) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.00 | 82 / 82 | 0.0% [0.0%, 4.5%] | 98.8% [93.4%, 99.8%] | 2.4% [0.7%, 8.5%] | 2.4% [0.7%, 8.5%] |
| 0.05 | 82 / 82 | 0.0% [0.0%, 4.5%] | 98.8% [93.4%, 99.8%] | 2.4% [0.7%, 8.5%] | 2.4% [0.7%, 8.5%] |
| 0.10 | 82 / 82 | 0.0% [0.0%, 4.5%] | 98.8% [93.4%, 99.8%] | 2.4% [0.7%, 8.5%] | 2.4% [0.7%, 8.5%] |
| 0.15 | 82 / 82 | 0.0% [0.0%, 4.5%] | 98.8% [93.4%, 99.8%] | 2.4% [0.7%, 8.5%] | 2.4% [0.7%, 8.5%] |
| 0.20 | 82 / 82 | 0.0% [0.0%, 4.5%] | 98.8% [93.4%, 99.8%] | 2.4% [0.7%, 8.5%] | 2.4% [0.7%, 8.5%] |
| 0.25 | 82 / 82 | 0.0% [0.0%, 4.5%] | 98.8% [93.4%, 99.8%] | 2.4% [0.7%, 8.5%] | 2.4% [0.7%, 8.5%] |
| 0.30 | 82 / 82 | 0.0% [0.0%, 4.5%] | 98.8% [93.4%, 99.8%] | 2.4% [0.7%, 8.5%] | 2.4% [0.7%, 8.5%] |
| 0.35 | 82 / 82 | 0.0% [0.0%, 4.5%] | 98.8% [93.4%, 99.8%] | 2.4% [0.7%, 8.5%] | 2.4% [0.7%, 8.5%] |
| 0.40 | 80 / 82 | 2.4% [0.7%, 8.5%] | 98.8% [93.2%, 99.8%] | 2.5% [0.7%, 8.7%] | 2.5% [0.7%, 8.7%] |
| 0.45 | 76 / 82 | 7.3% [3.4%, 15.1%] | 98.7% [92.9%, 99.8%] | 2.6% [0.7%, 9.1%] | 2.6% [0.7%, 9.1%] |
| 0.50 | 75 / 82 | 8.5% [4.2%, 16.6%] | 98.7% [92.8%, 99.8%] | 2.7% [0.7%, 9.2%] | 2.7% [0.7%, 9.2%] |
| 0.55 | 72 / 82 | 12.2% [6.8%, 21.0%] | 98.6% [92.5%, 99.8%] | 2.8% [0.8%, 9.6%] | 2.8% [0.8%, 9.6%] |
| 0.60 | 62 / 82 | 24.4% [16.4%, 34.7%] | 98.4% [91.4%, 99.7%] | 3.2% [0.9%, 11.0%] | 3.2% [0.9%, 11.0%] |
| 0.65 | 54 / 82 | 34.2% [24.8%, 44.9%] | 98.2% [90.2%, 99.7%] | 3.7% [1.0%, 12.5%] | 3.7% [1.0%, 12.5%] |
| 0.70 | 49 / 82 | 40.2% [30.3%, 51.1%] | 98.0% [89.3%, 99.6%] | 2.0% [0.4%, 10.7%] | 2.0% [0.4%, 10.7%] |
| 0.75 | 42 / 82 | 48.8% [38.3%, 59.4%] | 100.0% [91.6%, 100.0%] | 0.0% [0.0%, 8.4%] | 0.0% [0.0%, 8.4%] |
| 0.80 | 31 / 82 | 62.2% [51.4%, 71.9%] | 100.0% [89.0%, 100.0%] | 0.0% [0.0%, 11.0%] | 0.0% [0.0%, 11.0%] |
| 0.85 | 19 / 82 | 76.8% [66.6%, 84.6%] | 100.0% [83.2%, 100.0%] | 0.0% [0.0%, 16.8%] | 0.0% [0.0%, 16.8%] |
| 0.90 | 17 / 82 | 79.3% [69.3%, 86.6%] | 100.0% [81.6%, 100.0%] | 0.0% [0.0%, 18.4%] | 0.0% [0.0%, 18.4%] |
| 0.95 | 10 / 82 | 87.8% [79.0%, 93.2%] | 100.0% [72.2%, 100.0%] | 0.0% [0.0%, 27.8%] | 0.0% [0.0%, 27.8%] |

---

## Table 7: Stage S7 Blind Human Domain Expert Audit Records

- **Total Audit Cases**: 30 stratified explanation cases  
  - Candidate Unfaithful Cases: 10  
  - Abstained Gate Cases (tau < 0.70): 10  
  - Random Validated Controls: 10  
- **Blind Human Rater Sheet**: `results_b/final/audit_blind.csv` (human evaluation columns empty for independent evaluation)  
- **Unblinded Verification Key**: `results_b/final/audit_key.csv`  

---

## Machine Integrity & Quality Exclusions

- **Degenerate Machines** (zero validation or test anomalies): `machine-1-5, machine-1-7, machine-2-8, machine-3-1, machine-3-2, machine-3-4, machine-3-5, machine-3-10, machine-3-9`
- **Leakage Audit**: 84 of 84 production detector runs confirmed 0 overlap timesteps under purged contiguous splits (`status = pass`). Exactly 1 intentional legacy detector run documented with 99 overlap timesteps (`status = leak_legacy`).
- **Execution Failures**: 0 failures across all detector runs, attribution evaluations, conformal calibrators, and LLM queries.