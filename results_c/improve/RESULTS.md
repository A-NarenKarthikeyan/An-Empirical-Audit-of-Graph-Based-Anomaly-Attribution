# Improved analysis results

All numerical values below are copied from the CSV outputs in this directory. No rater labels were created or filled. `results_b/` was read only.

## Common gate

```text
COMMON GATE segments=327 machines=28 IG_segments=327
COMMON GATE IG machine-macro top1=83.89 top5=53.93 legacy=62.85 (targets 83.89/53.93/62.84; tolerance 0.05)
COMMON GATE max|z| machine-macro top5=59.55 (target 59.55)
COMMON GATE PASSED
```

## Part-specific gate lines

```text
PART 1 GATE PASSED: detector test metrics reproduced for 28/28 machines.
PART 2 GATE PASSED: segments=327 machines=28 common gate passed.
PART 3 SCORE GATE: normalized IG score vectors=327, sum range=[0.999999728,1.000000215], full-ranking matches=327/327 (need >=325).
PART 3 PREFIX GATE PASSED: rho=1,alpha=0.1 92.47% (reference 92.47%), q=36/36, rho=1,alpha=0.2 81.71% (reference 81.71%), q=31/31, rho=0.8,alpha=0.1 91.67% (reference 91.67%), q=28/28, rho=0.8,alpha=0.2 81.96% (reference 81.96%), q=24/24
PART 4 GATE PASSED: all five paired comparisons complete for 28 machines; 10000 cluster resamples, seed 42.
PART 5 GATE PASSED: selected=80 evaluated responses; strict verdict joined for all 80.
```

## Part 1 — detector outcome split

Detector window predictions were generated with `scratch/option_b/s1_detectors.py::run_single_detector_task` split and feature construction, `src.data_loader.compute_node_features_6d`, saved seed-42 checkpoints, and each machine's `results_b/metrics/<machine>_seed42.json::val_optimal_threshold`. We scored all windows over the SMD test series. A segment was detected if any window whose start index lies inclusively in `[start,end]` exceeded the validation-tuned threshold. The test-block subset uses segment starts at or after that machine's `test_slice.start` (`val_end`). The inference gate reproduced saved test precision/recall/F1/AUROC for 28/28 machines.

| Subset | Outcome | n | Machines | IG top1/top5/legacy % | max|z| top1/top5/legacy % | IG−max|z| top5 pp | 95% CI pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| all_327 | detected | 295 | 26 | 87.11/54.87/63.60 | 82.52/60.86/72.97 | -5.99 | [-8.55, -3.31] |
| all_327 | missed | 32 | 17 | 55.39/61.69/45.98 | 80.88/69.25/71.11 | -7.55 | [-21.29, +2.85] |
| test_block_only | detected | 37 | 19 | 95.26/56.17/62.94 | 90.00/60.98/82.56 | -4.81 | [-11.17, +2.93] |
| test_block_only | missed | 17 | 12 | 36.11/71.38/35.93 | 66.67/70.37/62.77 | +1.01 | [-10.07, +12.54] |

## Part 2 — heterogeneity

Per-machine top-5 results (percent):

| Machine | n | IG top-5 % | max|z| top-5 % | Difference pp |
| --- | --- | --- | --- | --- |
| machine-1-1 | 8 | 56.26 | 59.38 | -3.13 |
| machine-1-2 | 10 | 59.06 | 62.98 | -3.92 |
| machine-1-3 | 11 | 72.15 | 75.27 | -3.12 |
| machine-1-4 | 12 | 58.17 | 64.32 | -6.15 |
| machine-1-5 | 7 | 33.40 | 46.93 | -13.53 |
| machine-1-6 | 30 | 50.49 | 48.27 | +2.22 |
| machine-1-7 | 13 | 47.67 | 55.52 | -7.85 |
| machine-1-8 | 20 | 73.21 | 60.37 | +12.85 |
| machine-2-1 | 13 | 44.13 | 46.90 | -2.77 |
| machine-2-2 | 11 | 61.52 | 70.07 | -8.56 |
| machine-2-3 | 10 | 71.49 | 73.17 | -1.68 |
| machine-2-4 | 20 | 62.12 | 84.27 | -22.15 |
| machine-2-5 | 21 | 83.62 | 88.23 | -4.60 |
| machine-2-6 | 8 | 46.37 | 53.60 | -7.23 |
| machine-2-7 | 20 | 82.02 | 94.52 | -12.50 |
| machine-2-8 | 1 | 14.71 | 14.71 | 0.00 |
| machine-2-9 | 10 | 52.33 | 63.50 | -11.17 |
| machine-3-1 | 4 | 27.92 | 37.50 | -9.58 |
| machine-3-10 | 13 | 57.74 | 74.19 | -16.45 |
| machine-3-11 | 3 | 30.58 | 28.91 | +1.67 |
| machine-3-2 | 9 | 66.48 | 85.00 | -18.52 |
| machine-3-3 | 28 | 72.58 | 72.24 | +0.34 |
| machine-3-4 | 8 | 81.77 | 89.06 | -7.29 |
| machine-3-5 | 11 | 43.86 | 47.25 | -3.39 |
| machine-3-6 | 11 | 40.47 | 52.17 | -11.70 |
| machine-3-7 | 5 | 39.81 | 37.51 | +2.30 |
| machine-3-8 | 6 | 41.57 | 43.08 | -1.52 |
| machine-3-9 | 4 | 38.42 | 38.42 | 0.00 |

By ground-truth size bin; the three metrics are machine-macro percentages in top1/top5/legacy order:

| |G| | n | Machines | IG top1/top5/legacy % | max|z| top1/top5/legacy % | IG−max|z| top5 pp | 95% CI pp |
| --- | --- | --- | --- | --- | --- | --- |
| 1-2 | 68 | 21 | 73.10/82.42/57.54 | 76.03/85.56/71.31 | -3.13 | [-9.48, +2.58] |
| 3-5 | 132 | 21 | 80.92/72.32/63.16 | 76.09/81.61/72.11 | -9.29 | [-14.19, -4.77] |
| 6-10 | 53 | 19 | 81.58/43.13/55.35 | 74.45/50.52/68.25 | -7.39 | [-11.76, -3.19] |
| >10 | 74 | 26 | 87.89/22.39/65.71 | 88.29/24.97/75.90 | -2.57 | [-3.80, -1.36] |

## Part 3 — score-threshold conformal sets

The score gate matched the complete stored ranking on 327/327 segments; normalized score-vector sums ranged 0.999999728–1.000000215. Prefix-method gate reproduced q̂ 36/31/28/24 and held-out coverage 92.47/81.71/91.67/81.96%.

| ρ | α | Prefix q̂ / size | Prefix held-out % | Prefix LOO % | Score threshold | Score held-out % | Score mean size | Median | IQR | Score LOO % |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.0 | 0.1 | 36 | 92.47 | 91.74 | 0.001000 | 90.06 | 33.97 | 38.00 | 31.00–38.00 | 89.91 |
| 1.0 | 0.2 | 31 | 81.71 | 81.04 | 0.003616 | 80.45 | 26.98 | 29.00 | 21.00–36.00 | 80.12 |
| 0.8 | 0.1 | 28 | 91.67 | 90.21 | 0.001686 | 90.40 | 31.90 | 37.00 | 29.00–38.00 | 89.60 |
| 0.8 | 0.2 | 24 | 81.96 | 81.04 | 0.005293 | 80.70 | 21.85 | 23.00 | 14.00–29.00 | 80.73 |

## Part 4 — Holm correction

All p-values use the requested 10,000 machine-clustered resamples, seed 42, two-sided sign-tail formula and 1/10,000 floor.

| Comparison | Difference pp | 95% CI pp | p | Holm p | Significant |
| --- | --- | --- | --- | --- | --- |
| IG minus GNNExplainer | +13.13 | [+9.81, +16.66] | 0.0001 | 0.0005 | Yes |
| IG minus attention | +40.94 | [+32.74, +48.72] | 0.0001 | 0.0005 | Yes |
| IG minus max_abs_z | -5.62 | [-8.27, -2.98] | 0.0001 | 0.0005 | Yes |
| IG minus mean_abs_z | +12.95 | [+7.30, +18.86] | 0.0001 | 0.0005 | Yes |
| IG minus mean_shift | +19.04 | [+11.97, +26.41] | 0.0001 | 0.0005 | Yes |

## Part 5 — fixed-lexicon exaggeration scan

The strict-scorer verdict was joined from `results_b/final/faithfulness_results.csv::overall_faithful`; the 80 responses were selected from `faithfulness_results_v2.csv` rows with `struct_status=tested`.

| Strict scorer verdict | n | Flagged | Share | Wilson 95% CI |
| --- | --- | --- | --- | --- |
| all | 80 | 11 | 13.75% | [7.85%, 22.97%] |
| False | 78 | 11 | 14.10% | [8.06%, 23.51%] |
| True | 2 | 0 | 0.00% | [0.00%, 65.76%] |

Examples (10 sensor-magnitude instances across flagged responses):

| Audit ID | Sensor | z std | Matched wording | Attached wording |
| --- | --- | --- | --- | --- |
| 5 | 18 | 1.49 | significant | Sensor_18 exhibits the largest positive Z-Score shift (+1.49 std) relative to its normal baseline, indicating a significant deviation, while the remaining top sensors show smaller but still positive Z-Score shifts, collectively contributing to the anomaly attribution. |
| 9 | 30 | 1.29 | significant | Sensor_30 exhibits a significant positive z-score shift (+1.29 std), indicating an outlier contribution, while the remaining top-5 sensors show negligible or non-significant shifts, implying their values are within expected variation. |
| 14 | 6 | 1.27 | large | Sensor_6 exhibits a large negative Z-Score shift (-1.27 std) indicating values well below baseline; these opposing deviations collectively drive the anomaly. |
| 17 | 9 | 1.02 | significant | Sensor_13 (+1.45 std), Sensor_10 (+7.55 std), Sensor_25 (+4.20 std), Sensor_9 (+1.02 std) and Sensor_34 (+0.11 std) show significant z-score shifts from their normal baselines, indicating anomalous telemetry behavior. |
| 17 | 13 | 1.45 | significant | Sensor_13 (+1.45 std), Sensor_10 (+7.55 std), Sensor_25 (+4.20 std), Sensor_9 (+1.02 std) and Sensor_34 (+0.11 std) show significant z-score shifts from their normal baselines, indicating anomalous telemetry behavior. |
| 17 | 34 | 0.11 | significant | Sensor_34 (+0.11 std) show significant z-score shifts from their normal baselines, indicating anomalous telemetry behavior. |
| 21 | 14 | 0.69 | significant | Sensor_14 shows a +0.69 std Z-Score shift, indicating a significant deviation from its normal baseline, while the other four sensors exhibit minimal Z-Score shifts within expected variation. |
| 86 | 16 | 1.26 | substantial | Sensor_16 shows a +1.26 std Z-Score shift, indicating a substantial deviation from its normal baseline, while |
| 166 | 13 | 1.18 | significant | Sensor_13 exhibit Z-Score shifts of +1.18 std, indicating significant elevation above baseline; |
| 186 | 1 | 1.05 | large | Sensor_1 shows a +1.05 std Z-Score shift, indicating a large deviation above baseline; |

Lexicon and executable function code: `part5_lexicon_and_function.txt`. Full analysis functions and gates: `improve_analysis.py`; common gate: `common_gate.py`.

## Files

- `results_c/improve/GATE_COMMON.txt`
- `results_c/improve/LOG.md`
- `results_c/improve/common_gate.py`
- `results_c/improve/improve_analysis.py`
- `results_c/improve/part1_detected_missed_summary.csv`
- `results_c/improve/part1_segment_outcomes.csv`
- `results_c/improve/part2_by_ground_truth_size.csv`
- `results_c/improve/part3_prefix_gate_reproduction.csv`
- `results_c/improve/part3_prefix_vs_score_comparison.csv`
- `results_c/improve/part3_score_threshold_conformal.csv`
- `results_c/improve/part4_holm_paired_top5.csv`
- `results_c/improve/part5_exaggeration_summary.csv`
- `results_c/improve/part5_examples.csv`
- `results_c/improve/part5_lexicon_and_function.txt`
- `results_c/improve/part5_response_flags.csv`
- `results_c/improve/per_machine_top5.csv`
