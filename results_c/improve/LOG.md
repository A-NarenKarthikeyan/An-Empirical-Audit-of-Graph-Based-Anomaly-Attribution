# results_c/improve analysis log

Scope: derive analysis outputs from existing repository data only. No labels are created or filled. `results_b/` is read-only for this task.

## Step 1 — Common gate setup
- Added `common_gate.py` and designated `GATE_COMMON.txt` as the printed-value capture.
- Inputs: `results_b/s3_cached_rankings.json` and `results_b/attribution_per_segment.csv`; IG scores are grouped by machine, and max|z| is recomputed from the cached raw windows and stored normalization statistics using the loader calculation in `scripts/compute_baselines.py`.
- Running the common gate before any analysis part.

## Part 1 — started
- Common gate PASSED; part-specific gate will run before this part's calculations.

## Part 1 — failed
- Part 1 FAILED: RuntimeError: PART 1 GATE FAILED: detector inference dependencies unavailable: ModuleNotFoundError: No module named 'src'
- Traceback (most recent call last):
  File "C:\Users\Naren karthikeyan\OneDrive\Desktop\paper\paper\results_c\improve\improve_analysis.py", line 120, in part1
    from src.data_loader import load_smd_raw, normalize_data, compute_sensor_correlation_graph, compute_node_features_6d
ModuleNotFoundError: No module named 'src'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "C:\Users\Naren karthikeyan\OneDrive\Desktop\paper\paper\results_c\improve\improve_analysis.py", line 575, in main
    fn()
  File "C:\Users\Naren karthikeyan\OneDrive\Desktop\paper\paper\results_c\improve\improve_analysis.py", line 566, in <lambda>
    parts = [("Part 1", lambda: part1(cache, base)),
                                ^^^^^^^^^^^^^^^^^^
  File "C:\Users\Naren karthikeyan\OneDrive\Desktop\paper\paper\results_c\improve\improve_analysis.py", line 124, in part1
    raise RuntimeError(f"PART 1 GATE FAILED: detector inference dependencies unavailable: {type(exc).__name__}: {exc}") from exc
RuntimeError: PART 1 GATE FAILED: detector inference dependencies unavailable: ModuleNotFoundError: No module named 'src'

## Part 2 — started
- Common gate PASSED; part-specific gate will run before this part's calculations.

## Part 2 — failed
- Part 2 FAILED: AttributeError: 'function' object has no attribute 'mean'
- Traceback (most recent call last):
  File "C:\Users\Naren karthikeyan\OneDrive\Desktop\paper\paper\results_c\improve\improve_analysis.py", line 575, in main
    fn()
  File "C:\Users\Naren karthikeyan\OneDrive\Desktop\paper\paper\results_c\improve\improve_analysis.py", line 567, in <lambda>
    ("Part 2", lambda: part2(cache, base)),
                       ^^^^^^^^^^^^^^^^^^
  File "C:\Users\Naren karthikeyan\OneDrive\Desktop\paper\paper\results_c\improve\improve_analysis.py", line 256, in part2
    row["paired_top5_diff_machine_macro_pp"] = g.groupby("machine").diff.mean().mean() * 100
                                               ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'function' object has no attribute 'mean'

## Part 3 — started
- Common gate PASSED; part-specific gate will run before this part's calculations.

## Part 3 — score-threshold conformal sets
- PART 3 SCORE GATE: normalized IG score vectors=327, sum range=[0.999999728,1.000000215], full-ranking matches=327/327 (need >=325).
- PART 3 PREFIX GATE PASSED: rho=1,alpha=0.1 92.47% (reference 92.47%), q=36/36, rho=1,alpha=0.2 81.71% (reference 81.71%), q=31/31, rho=0.8,alpha=0.1 91.67% (reference 91.67%), q=28/28, rho=0.8,alpha=0.2 81.96% (reference 81.96%), q=24/24
- Wrote part3_score_threshold_conformal.csv and part3_prefix_gate_reproduction.csv.
-  rho  alpha  calibration_threshold  held_out_coverage_mean_pct  held_out_coverage_sd_pct  mean_set_size  median_set_size  set_size_q1  set_size_q3  loo_pooled_coverage_pct
 1.0    0.1               0.001000                   90.060366                  3.266785      33.968616             38.0         31.0         38.0                89.908257
 1.0    0.2               0.003616                   80.454878                  4.346749      26.976982             29.0         21.0         36.0                80.122324
 0.8    0.1               0.001686                   90.089634                  3.176831      31.792585             37.0         29.0         38.0                89.602446
 0.8    0.2               0.005293                   80.395732                  4.426968      21.642780             23.0         13.0         29.0                80.733945

## Part 4 — started
- Common gate PASSED; part-specific gate will run before this part's calculations.

## Part 4 — failed
- Part 4 FAILED: AttributeError: 'function' object has no attribute 'mean'
- Traceback (most recent call last):
  File "C:\Users\Naren karthikeyan\OneDrive\Desktop\paper\paper\results_c\improve\improve_analysis.py", line 575, in main
    fn()
  File "C:\Users\Naren karthikeyan\OneDrive\Desktop\paper\paper\results_c\improve\improve_analysis.py", line 569, in <lambda>
    ("Part 4", lambda: part4(base, attr)),
                       ^^^^^^^^^^^^^^^^^
  File "C:\Users\Naren karthikeyan\OneDrive\Desktop\paper\paper\results_c\improve\improve_analysis.py", line 437, in part4
    dm = pair.groupby("machine").diff.mean().reindex(machines).to_numpy()
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'function' object has no attribute 'mean'

## Part 5 — started
- Common gate PASSED; part-specific gate will run before this part's calculations.

## Part 5 — fixed-lexicon exaggeration scan
- PART 5 GATE PASSED: selected=80 evaluated responses; strict verdict joined for all 80.
- Lexicon: significant, substantial, large, massive, extreme, strong, primary driver, dominant, far outside.
- Wrote part5_response_flags.csv, part5_exaggeration_summary.csv, part5_examples.csv, and part5_lexicon_and_function.txt.
- strict_scorer_verdict  n_responses  overstated_n  overstated_share_pct  wilson95_lower_pct  wilson95_upper_pct
                  all           80             0                   0.0        0.000000e+00            4.581813
                False           78             0                   0.0       -3.469447e-16            4.693781
                 True            2             0                   0.0        0.000000e+00           65.761977
- Examples:
- No flagged examples.

## Part 1 — started
- Common gate PASSED; part-specific gate will run before this part's calculations.

## Part 1 — detector outcome split
- PART 1 GATE: common gate passed; preparing seed-42 window predictions from the saved detector checkpoints.
- PART 1 GATE PASSED: detector test metrics reproduced for 28/28 machines.
- Window predictions: `scratch/option_b/s1_detectors.py::run_single_detector_task` split/feature recipe; same `compute_node_features_6d` and saved checkpoint; window probabilities thresholded at `results_b/metrics/<machine>_seed42.json::val_optimal_threshold`. Segment hit = any window whose start index is in inclusive [start,end].
- Wrote part1_segment_outcomes.csv and part1_detected_missed_summary.csv.
-          subset  outcome  n_segments  n_machines  IG_top1_machine_macro_pct  IG_top5_machine_macro_pct  IG_legacy_machine_macro_pct  max_abs_z_top1_machine_macro_pct  max_abs_z_top5_machine_macro_pct  max_abs_z_legacy_machine_macro_pct  paired_top5_diff_machine_macro_pp  paired_diff_ci95_lower_pp  paired_diff_ci95_upper_pp                                   ci_note
        all_327 detected         295          26                  87.109170                  54.870025                    63.599756                         82.517573                         60.859049                           72.965594                          -5.989024                  -8.549951                  -3.309057 10,000 machine-cluster bootstrap, seed 42
        all_327   missed          32          17                  55.392157                  61.692803                    45.977873                         80.882353                         69.246491                           71.109416                          -7.553688                 -21.291492                   2.850315 10,000 machine-cluster bootstrap, seed 42
test_block_only detected          37          19                  95.263158                  56.167388                    62.939024                         90.000000                         60.982269                           82.561970                          -4.814881                 -11.168624                   2.925931 10,000 machine-cluster bootstrap, seed 42
test_block_only   missed          17          12                  36.111111                  71.378968                    35.932540                         66.666667                         70.367063                           62.767857                           1.011905                 -10.069444                  12.540427 10,000 machine-cluster bootstrap, seed 42

## Part 2 — started
- Common gate PASSED; part-specific gate will run before this part's calculations.

## Part 2 — heterogeneity
- PART 2 GATE PASSED: segments=327 machines=28 common gate passed.
- Wrote per_machine_top5.csv and part2_by_ground_truth_size.csv.
- Per-machine table:
-      machine  n_segments  ig_top5_pct  max_abs_z_top5_pct  difference_ig_minus_maxz_pp
 machine-1-1           8    56.257200           59.382200                -3.125000e+00
 machine-1-2          10    59.059524           62.976190                -3.916667e+00
 machine-1-3          11    72.148587           75.265470                -3.116883e+00
 machine-1-4          12    58.169935           64.320728                -6.150794e+00
 machine-1-5           7    33.395692           46.927438                -1.353175e+01
 machine-1-6          30    50.488610           48.266740                 2.221870e+00
 machine-1-7          13    47.674603           55.521978                -7.847375e+00
 machine-1-8          20    73.210317           60.365079                 1.284524e+01
 machine-2-1          13    44.134199           46.903097                -2.768898e+00
 machine-2-2          11    61.515152           70.070346                -8.555195e+00
 machine-2-3          10    71.491064           73.169386                -1.678322e+00
 machine-2-4          20    62.119048           84.273810                -2.215476e+01
 machine-2-5          21    83.621934           88.225108                -4.603175e+00
 machine-2-6           8    46.369048           53.601190                -7.232143e+00
 machine-2-7          20    82.023810           94.523810                -1.250000e+01
 machine-2-8           1    14.705882           14.705882                -8.326673e-15
 machine-2-9          10    52.333333           63.500000                -1.116667e+01
 machine-3-1           4    27.916667           37.500000                -9.583333e+00
machine-3-10          13    57.738577           74.185464                -1.644689e+01
machine-3-11           3    30.579710           28.913043                 1.666667e+00
 machine-3-2           9    66.481481           85.000000                -1.851852e+01
 machine-3-3          28    72.577799           72.241019                 3.367794e-01
 machine-3-4           8    81.770833           89.062500                -7.291667e+00
 machine-3-5          11    43.856344           47.247205                -3.390860e+00
 machine-3-6          11    40.469917           52.174697                -1.170478e+01
 machine-3-7           5    39.813853           37.510823                 2.303030e+00
 machine-3-8           6    41.569264           43.084416                -1.515152e+00
 machine-3-9           4    38.421474           38.421474                 0.000000e+00
- By-|G| table:
- G_bin  n_segments  n_machines  IG_top1_machine_macro_pct  IG_top5_machine_macro_pct  IG_legacy_machine_macro_pct  max_abs_z_top1_machine_macro_pct  max_abs_z_top5_machine_macro_pct  max_abs_z_legacy_machine_macro_pct  paired_top5_diff_machine_macro_pp  diff_ci95_lower_pp  diff_ci95_upper_pp                                   ci_note
  1-2          68          21                  73.095238                  82.420635                    57.539683                         76.031746                         85.555556                           71.309524                          -3.134921           -9.484127            2.579365 10,000 machine-cluster bootstrap, seed 42
  3-5         132          21                  80.920610                  72.320856                    63.161585                         76.093960                         81.612829                           72.108568                          -9.291973          -14.192974           -4.768280 10,000 machine-cluster bootstrap, seed 42
 6-10          53          19                  81.578947                  43.132156                    55.348744                         74.448622                         50.522696                           68.247812                          -7.390540          -11.760634           -3.189704 10,000 machine-cluster bootstrap, seed 42
  >10          74          26                  87.894477                  22.391647                    65.710612                         88.288955                         24.966063                           75.903911                          -2.574416           -3.796474           -1.357674 10,000 machine-cluster bootstrap, seed 42

## Part 3 — started
- Common gate PASSED; part-specific gate will run before this part's calculations.

## Part 3 — score-threshold conformal sets
- PART 3 SCORE GATE: normalized IG score vectors=327, sum range=[0.999999728,1.000000215], full-ranking matches=327/327 (need >=325).
- PART 3 PREFIX GATE PASSED: rho=1,alpha=0.1 92.47% (reference 92.47%), q=36/36, rho=1,alpha=0.2 81.71% (reference 81.71%), q=31/31, rho=0.8,alpha=0.1 91.67% (reference 91.67%), q=28/28, rho=0.8,alpha=0.2 81.96% (reference 81.96%), q=24/24
- Wrote part3_score_threshold_conformal.csv and part3_prefix_gate_reproduction.csv.
-  rho  alpha  calibration_threshold  held_out_coverage_mean_pct  held_out_coverage_sd_pct  mean_set_size  median_set_size  set_size_q1  set_size_q3  loo_pooled_coverage_pct
 1.0    0.1               0.001000                   90.060366                  3.266785      33.968616             38.0         31.0         38.0                89.908257
 1.0    0.2               0.003616                   80.454878                  4.346749      26.976982             29.0         21.0         36.0                80.122324
 0.8    0.1               0.001686                   90.089634                  3.176831      31.792585             37.0         29.0         38.0                89.602446
 0.8    0.2               0.005293                   80.395732                  4.426968      21.642780             23.0         13.0         29.0                80.733945

## Part 4 — started
- Common gate PASSED; part-specific gate will run before this part's calculations.

## Part 4 — failed
- Part 4 FAILED: AttributeError: 'function' object has no attribute 'mean'
- Traceback (most recent call last):
  File "C:\Users\Naren karthikeyan\OneDrive\Desktop\paper\paper\results_c\improve\improve_analysis.py", line 576, in main
    fn()
  File "C:\Users\Naren karthikeyan\OneDrive\Desktop\paper\paper\results_c\improve\improve_analysis.py", line 570, in <lambda>
    ("Part 4", lambda: part4(base, attr)),
                       ^^^^^^^^^^^^^^^^^
  File "C:\Users\Naren karthikeyan\OneDrive\Desktop\paper\paper\results_c\improve\improve_analysis.py", line 438, in part4
    dm = pair.groupby("machine").diff.mean().reindex(machines).to_numpy()
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AttributeError: 'function' object has no attribute 'mean'

## Part 5 — started
- Common gate PASSED; part-specific gate will run before this part's calculations.

## Part 5 — fixed-lexicon exaggeration scan
- PART 5 GATE PASSED: selected=80 evaluated responses; strict verdict joined for all 80.
- Lexicon: significant, substantial, large, massive, extreme, strong, primary driver, dominant, far outside.
- Wrote part5_response_flags.csv, part5_exaggeration_summary.csv, part5_examples.csv, and part5_lexicon_and_function.txt.
- strict_scorer_verdict  n_responses  overstated_n  overstated_share_pct  wilson95_lower_pct  wilson95_upper_pct
                  all           80             0                   0.0        0.000000e+00            4.581813
                False           78             0                   0.0       -3.469447e-16            4.693781
                 True            2             0                   0.0        0.000000e+00           65.761977
- Examples:
- No flagged examples.

## Part 1 — started
- Common gate PASSED; part-specific gate will run before this part's calculations.

## Part 4 — started
- Common gate PASSED; part-specific gate will run before this part's calculations.

## Part 4 — Holm correction
- PART 4 GATE PASSED: 28 machines and all five comparison rows present.
- Bootstrap p = 2*min(P(d<=0),P(d>=0)); floor=1/10,000; Holm family size=5.
- Wrote part4_holm_paired_top5.csv.
-            comparison  IG_top5_machine_macro_pct  other_top5_machine_macro_pct  difference_pp  bootstrap_ci95_lower_pp  bootstrap_ci95_upper_pp  p_raw  p_holm  significant_holm_0_05
IG minus GNNExplainer                  53.925495                     40.798674      13.126821                 9.812744                16.660780 0.0001  0.0005                   True
   IG minus attention                  53.925495                     12.985671      40.939824                32.740869                48.717390 0.0001  0.0005                   True
   IG minus max_abs_z                  53.925495                     59.547825      -5.622330                -8.271322                -2.980947 0.0001  0.0005                   True
  IG minus mean_abs_z                  53.925495                     40.974563      12.950932                 7.298239                18.859453 0.0001  0.0005                   True
  IG minus mean_shift                  53.925495                     34.884341      19.041154                11.968589                26.407267 0.0001  0.0005                   True

## Part 5 — started
- Common gate PASSED; part-specific gate will run before this part's calculations.

## Part 5 — failed
- Part 5 FAILED: UnicodeEncodeError: 'charmap' codec can't encode character '\u2011' in position 685: character maps to <undefined>
- Traceback (most recent call last):
  File "C:\Users\Naren karthikeyan\OneDrive\Desktop\paper\paper\results_c\improve\improve_analysis.py", line 589, in main
    fn()
  File "C:\Users\Naren karthikeyan\OneDrive\Desktop\paper\paper\results_c\improve\improve_analysis.py", line 562, in part5
    print("PART 5 EXAMPLES\n" + (exdf.to_string(index=False) if len(exdf) else "No flagged examples."), flush=True)
  File "C:\Users\Naren karthikeyan\AppData\Local\Programs\Python\Python311\Lib\encodings\cp1252.py", line 19, in encode
    return codecs.charmap_encode(input,self.errors,encoding_table)[0]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
UnicodeEncodeError: 'charmap' codec can't encode character '\u2011' in position 685: character maps to <undefined>

## Part 4 — started
- Common gate PASSED; part-specific gate will run before this part's calculations.

## Part 4 — Holm correction
- PART 4 GATE PASSED: 28 machines and all five comparison rows present.
- Bootstrap p = 2*min(P(d<=0),P(d>=0)); floor=1/10,000; Holm family size=5.
- Wrote part4_holm_paired_top5.csv.
-            comparison  IG_top5_machine_macro_pct  other_top5_machine_macro_pct  difference_pp  bootstrap_ci95_lower_pp  bootstrap_ci95_upper_pp  p_raw  p_holm  significant_holm_0_05
IG minus GNNExplainer                  53.925495                     40.798674      13.126821                 9.812744                16.660780 0.0001  0.0005                   True
   IG minus attention                  53.925495                     12.985671      40.939824                32.740869                48.717390 0.0001  0.0005                   True
   IG minus max_abs_z                  53.925495                     59.547825      -5.622330                -8.271322                -2.980947 0.0001  0.0005                   True
  IG minus mean_abs_z                  53.925495                     40.974563      12.950932                 7.298239                18.859453 0.0001  0.0005                   True
  IG minus mean_shift                  53.925495                     34.884341      19.041154                11.968589                26.407267 0.0001  0.0005                   True

## Part 5 — started
- Common gate PASSED; part-specific gate will run before this part's calculations.

## Part 5 — fixed-lexicon exaggeration scan
- PART 5 GATE PASSED: selected=80 evaluated responses; strict verdict joined for all 80.
- Lexicon: significant, substantial, large, massive, extreme, strong, primary driver, dominant, far outside.
- Wrote part5_response_flags.csv, part5_exaggeration_summary.csv, part5_examples.csv, and part5_lexicon_and_function.txt.
- strict_scorer_verdict  n_responses  overstated_n  overstated_share_pct  wilson95_lower_pct  wilson95_upper_pct
                  all           80             2              2.500000            0.688289            8.664433
                False           78             2              2.564103            0.706009            8.875270
                 True            2             0              0.000000            0.000000           65.761977
- Examples:
-  audit_id  segment_global_id      machine strict_scorer_verdict  sensor    z                                                                                                                                                                                                                   wording lexicon_hits
       17                  1  machine-1-1                 False       9 1.02 Sensor_13 (+1.45 std), Sensor_10 (+7.55 std), Sensor_25 (+4.20 std), Sensor_9 (+1.02 std) and Sensor_34 (+0.11 std) show significant z-score shifts from their normal baselines, indicating anomalous telemetry behavior.  significant
       17                  1  machine-1-1                 False      13 1.45 Sensor_13 (+1.45 std), Sensor_10 (+7.55 std), Sensor_25 (+4.20 std), Sensor_9 (+1.02 std) and Sensor_34 (+0.11 std) show significant z-score shifts from their normal baselines, indicating anomalous telemetry behavior.  significant
       17                  1  machine-1-1                 False      34 0.11 Sensor_13 (+1.45 std), Sensor_10 (+7.55 std), Sensor_25 (+4.20 std), Sensor_9 (+1.02 std) and Sensor_34 (+0.11 std) show significant z-score shifts from their normal baselines, indicating anomalous telemetry behavior.  significant
      237                237 machine-3-10                 False      13 0.81                                                                                  Sensor_14 (+1.33 std) and Sensor_13 (+0.81 std) exhibit significant positive Z-Score shifts, indicating notable deviation from baseline,  significant
      237                237 machine-3-10                 False      14 1.33                                                                                  Sensor_14 (+1.33 std) and Sensor_13 (+0.81 std) exhibit significant positive Z-Score shifts, indicating notable deviation from baseline,  significant

## Part 5 — started
- Common gate PASSED; part-specific gate will run before this part's calculations.

## Part 5 — fixed-lexicon exaggeration scan
- PART 5 GATE PASSED: selected=80 evaluated responses; strict verdict joined for all 80.
- Lexicon: significant, substantial, large, massive, extreme, strong, primary driver, dominant, far outside.
- Wrote part5_response_flags.csv, part5_exaggeration_summary.csv, part5_examples.csv, and part5_lexicon_and_function.txt.
- strict_scorer_verdict  n_responses  overstated_n  overstated_share_pct  wilson95_lower_pct  wilson95_upper_pct
                  all           80            16             20.000000           12.702844           30.046244
                False           78            16             20.512821           13.039795           30.753973
                 True            2             0              0.000000            0.000000           65.761977
- Examples:
-  audit_id  segment_global_id     machine strict_scorer_verdict  sensor    z                                                                                                                                                                                                                                        wording       lexicon_hits
        5                 10 machine-1-2                 False      18 1.49                                                                                                         Sensor_18 exhibits the largest positive Z-Score shift (+1.49 std) relative to its normal baseline, indicating a significant deviation,        significant
        9                313 machine-3-7                 False      30 1.29                                                                                                                                       Sensor_30 exhibits a significant positive z-score shift (+1.29 std), indicating an outlier contribution,        significant
       12                324 machine-3-9                 False      24 1.82 Sensors_35 and 36 exhibit large negative shifts (-2.90 std) suggesting significant drops, Sensor_25 has a moderate positive shift (+1.61 std), and Sensor_24 shows a negative shift (-1.82 std), collectively driving the model's attribution. large; significant
       12                324 machine-3-9                 False      25 1.61 Sensors_35 and 36 exhibit large negative shifts (-2.90 std) suggesting significant drops, Sensor_25 has a moderate positive shift (+1.61 std), and Sensor_24 shows a negative shift (-1.82 std), collectively driving the model's attribution. large; significant
       14                246 machine-3-2                 False       6 1.27                                                                                                                                             Sensor_6 exhibits a large negative Z-Score shift (-1.27 std) indicating values well below baseline              large
       17                  1 machine-1-1                 False       9 1.02                      Sensor_13 (+1.45 std), Sensor_10 (+7.55 std), Sensor_25 (+4.20 std), Sensor_9 (+1.02 std) and Sensor_34 (+0.11 std) show significant z-score shifts from their normal baselines, indicating anomalous telemetry behavior.        significant
       17                  1 machine-1-1                 False      13 1.45                      Sensor_13 (+1.45 std), Sensor_10 (+7.55 std), Sensor_25 (+4.20 std), Sensor_9 (+1.02 std) and Sensor_34 (+0.11 std) show significant z-score shifts from their normal baselines, indicating anomalous telemetry behavior.        significant
       17                  1 machine-1-1                 False      34 0.11                      Sensor_13 (+1.45 std), Sensor_10 (+7.55 std), Sensor_25 (+4.20 std), Sensor_9 (+1.02 std) and Sensor_34 (+0.11 std) show significant z-score shifts from their normal baselines, indicating anomalous telemetry behavior.        significant
       18                 94 machine-1-8                 False      10 1.17                                                                                                                                        Sensor_10 shows a +1.17 std Z-Score shift, indicating a significant deviation from its normal baseline,        significant
       21                147 machine-2-4                 False      14 0.69                                                                                                                                        Sensor_14 shows a +0.69 std Z-Score shift, indicating a significant deviation from its normal baseline,        significant

## Part 3 — started
- Common gate PASSED; part-specific gate will run before this part's calculations.

## Part 3 — score-threshold conformal sets
- PART 3 SCORE GATE: normalized IG score vectors=327, sum range=[0.999999728,1.000000215], full-ranking matches=327/327 (need >=325).
- PART 3 PREFIX GATE PASSED: rho=1,alpha=0.1 92.47% (reference 92.47%), q=36/36, rho=1,alpha=0.2 81.71% (reference 81.71%), q=31/31, rho=0.8,alpha=0.1 91.67% (reference 91.67%), q=28/28, rho=0.8,alpha=0.2 81.96% (reference 81.96%), q=24/24
- Wrote part3_score_threshold_conformal.csv, part3_prefix_gate_reproduction.csv, and part3_prefix_vs_score_comparison.csv.
-  rho  alpha  calibration_threshold  held_out_coverage_mean_pct  held_out_coverage_sd_pct  mean_set_size  median_set_size  set_size_q1  set_size_q3  loo_pooled_coverage_pct
 1.0    0.1               0.001000                   90.060366                  3.266785      33.968616             38.0         31.0         38.0                89.908257
 1.0    0.2               0.003616                   80.454878                  4.346749      26.976982             29.0         21.0         36.0                80.122324
 0.8    0.1               0.001686                   90.398171                  3.284203      31.904037             37.0         29.0         38.0                89.602446
 0.8    0.2               0.005293                   80.698171                  4.365242      21.846756             23.0         14.0         29.0                80.733945
- Prefix comparison:
-  rho  alpha  prefix_q_hat  prefix_set_size_sensors  prefix_held_out_coverage_pct  prefix_loo_pooled_coverage_pct  calibration_threshold  held_out_coverage_mean_pct  mean_set_size  median_set_size  set_size_q1  set_size_q3  loo_pooled_coverage_pct
 1.0    0.1            36                       36                     92.470122                       91.743119               0.001000                   90.060366      33.968616             38.0         31.0         38.0                89.908257
 1.0    0.2            31                       31                     81.706707                       81.039755               0.003616                   80.454878      26.976982             29.0         21.0         36.0                80.122324
 0.8    0.1            28                       28                     91.672561                       90.214067               0.001686                   90.398171      31.904037             37.0         29.0         38.0                89.602446
 0.8    0.2            24                       24                     81.956098                       81.039755               0.005293                   80.698171      21.846756             23.0         14.0         29.0                80.733945

## Part 5 — started
- Common gate PASSED; part-specific gate will run before this part's calculations.

## Part 5 — fixed-lexicon exaggeration scan
- PART 5 GATE PASSED: selected=80 evaluated responses; strict verdict joined for all 80.
- Lexicon: significant, substantial, large, massive, extreme, strong, primary driver, dominant, far outside.
- Wrote part5_response_flags.csv, part5_exaggeration_summary.csv, part5_examples.csv, and part5_lexicon_and_function.txt.
- strict_scorer_verdict  n_responses  overstated_n  overstated_share_pct  wilson95_lower_pct  wilson95_upper_pct
                  all           80            16             20.000000           12.702844           30.046244
                False           78            16             20.512821           13.039795           30.753973
                 True            2             0              0.000000            0.000000           65.761977
- Examples:
-  audit_id  segment_global_id     machine strict_scorer_verdict  sensor    z                                                                                                                                                                                                                                        wording       lexicon_hits
        5                 10 machine-1-2                 False      18 1.49                                                                                                         Sensor_18 exhibits the largest positive Z-Score shift (+1.49 std) relative to its normal baseline, indicating a significant deviation,        significant
        9                313 machine-3-7                 False      30 1.29                                                                                                                                       Sensor_30 exhibits a significant positive z-score shift (+1.29 std), indicating an outlier contribution,        significant
       12                324 machine-3-9                 False      24 1.82 Sensors_35 and 36 exhibit large negative shifts (-2.90 std) suggesting significant drops, Sensor_25 has a moderate positive shift (+1.61 std), and Sensor_24 shows a negative shift (-1.82 std), collectively driving the model's attribution. large; significant
       12                324 machine-3-9                 False      25 1.61 Sensors_35 and 36 exhibit large negative shifts (-2.90 std) suggesting significant drops, Sensor_25 has a moderate positive shift (+1.61 std), and Sensor_24 shows a negative shift (-1.82 std), collectively driving the model's attribution. large; significant
       14                246 machine-3-2                 False       6 1.27                                                                                                                                             Sensor_6 exhibits a large negative Z-Score shift (-1.27 std) indicating values well below baseline              large
       17                  1 machine-1-1                 False       9 1.02                      Sensor_13 (+1.45 std), Sensor_10 (+7.55 std), Sensor_25 (+4.20 std), Sensor_9 (+1.02 std) and Sensor_34 (+0.11 std) show significant z-score shifts from their normal baselines, indicating anomalous telemetry behavior.        significant
       17                  1 machine-1-1                 False      13 1.45                      Sensor_13 (+1.45 std), Sensor_10 (+7.55 std), Sensor_25 (+4.20 std), Sensor_9 (+1.02 std) and Sensor_34 (+0.11 std) show significant z-score shifts from their normal baselines, indicating anomalous telemetry behavior.        significant
       17                  1 machine-1-1                 False      34 0.11                      Sensor_13 (+1.45 std), Sensor_10 (+7.55 std), Sensor_25 (+4.20 std), Sensor_9 (+1.02 std) and Sensor_34 (+0.11 std) show significant z-score shifts from their normal baselines, indicating anomalous telemetry behavior.        significant
       18                 94 machine-1-8                 False      10 1.17                                                                                                                                        Sensor_10 shows a +1.17 std Z-Score shift, indicating a significant deviation from its normal baseline,        significant
       21                147 machine-2-4                 False      14 0.69                                                                                                                                        Sensor_14 shows a +0.69 std Z-Score shift, indicating a significant deviation from its normal baseline,        significant

## Part 2 — started
- Common gate PASSED; part-specific gate will run before this part's calculations.

## Part 2 — heterogeneity
- PART 2 GATE PASSED: segments=327 machines=28 common gate passed.
- Wrote per_machine_top5.csv and part2_by_ground_truth_size.csv.
- Per-machine table:
-      machine  n_segments  ig_top5_pct  max_abs_z_top5_pct  difference_ig_minus_maxz_pp
 machine-1-1           8    56.257200           59.382200                    -3.125000
 machine-1-2          10    59.059524           62.976190                    -3.916667
 machine-1-3          11    72.148587           75.265470                    -3.116883
 machine-1-4          12    58.169935           64.320728                    -6.150794
 machine-1-5           7    33.395692           46.927438                   -13.531746
 machine-1-6          30    50.488610           48.266740                     2.221870
 machine-1-7          13    47.674603           55.521978                    -7.847375
 machine-1-8          20    73.210317           60.365079                    12.845238
 machine-2-1          13    44.134199           46.903097                    -2.768898
 machine-2-2          11    61.515152           70.070346                    -8.555195
 machine-2-3          10    71.491064           73.169386                    -1.678322
 machine-2-4          20    62.119048           84.273810                   -22.154762
 machine-2-5          21    83.621934           88.225108                    -4.603175
 machine-2-6           8    46.369048           53.601190                    -7.232143
 machine-2-7          20    82.023810           94.523810                   -12.500000
 machine-2-8           1    14.705882           14.705882                     0.000000
 machine-2-9          10    52.333333           63.500000                   -11.166667
 machine-3-1           4    27.916667           37.500000                    -9.583333
machine-3-10          13    57.738577           74.185464                   -16.446886
machine-3-11           3    30.579710           28.913043                     1.666667
 machine-3-2           9    66.481481           85.000000                   -18.518519
 machine-3-3          28    72.577799           72.241019                     0.336779
 machine-3-4           8    81.770833           89.062500                    -7.291667
 machine-3-5          11    43.856344           47.247205                    -3.390860
 machine-3-6          11    40.469917           52.174697                   -11.704781
 machine-3-7           5    39.813853           37.510823                     2.303030
 machine-3-8           6    41.569264           43.084416                    -1.515152
 machine-3-9           4    38.421474           38.421474                     0.000000
- By-|G| table:
- G_bin  n_segments  n_machines  IG_top1_machine_macro_pct  IG_top5_machine_macro_pct  IG_legacy_machine_macro_pct  max_abs_z_top1_machine_macro_pct  max_abs_z_top5_machine_macro_pct  max_abs_z_legacy_machine_macro_pct  paired_top5_diff_machine_macro_pp  diff_ci95_lower_pp  diff_ci95_upper_pp                                   ci_note
  1-2          68          21                  73.095238                  82.420635                    57.539683                         76.031746                         85.555556                           71.309524                          -3.134921           -9.484127            2.579365 10,000 machine-cluster bootstrap, seed 42
  3-5         132          21                  80.920610                  72.320856                    63.161585                         76.093960                         81.612829                           72.108568                          -9.291973          -14.192974           -4.768280 10,000 machine-cluster bootstrap, seed 42
 6-10          53          19                  81.578947                  43.132156                    55.348744                         74.448622                         50.522696                           68.247812                          -7.390540          -11.760634           -3.189704 10,000 machine-cluster bootstrap, seed 42
  >10          74          26                  87.894477                  22.391647                    65.710612                         88.288955                         24.966063                           75.903911                          -2.574416           -3.796474           -1.357674 10,000 machine-cluster bootstrap, seed 42

## Final packaging
- Generated RESULTS.md from the saved output CSV files; no source results or labels modified.
- Final checks: common gate passed; Parts 1–5 completed successfully after corrected reruns.

## Part 5 — started
- Common gate PASSED; part-specific gate will run before this part's calculations.

## Part 5 — fixed-lexicon exaggeration scan
- PART 5 GATE PASSED: selected=80 evaluated responses; strict verdict joined for all 80.
- Lexicon: significant, substantial, large, massive, extreme, strong, primary driver, dominant, far outside.
- Wrote part5_response_flags.csv, part5_exaggeration_summary.csv, part5_examples.csv, and part5_lexicon_and_function.txt.
- strict_scorer_verdict  n_responses  overstated_n  overstated_share_pct  wilson95_lower_pct  wilson95_upper_pct
                  all           80            18             22.500000           14.733207           32.786790
                False           78            18             23.076923           15.125509           33.555758
                 True            2             0              0.000000            0.000000           65.761977
- Examples:
-  audit_id  segment_global_id     machine strict_scorer_verdict  sensor    z                                                                                                                                                                                                                                                                         wording lexicon_hits
        5                 10 machine-1-2                 False      18 1.49    Sensor_18 exhibits the largest positive Z-Score shift (+1.49 std) relative to its normal baseline, indicating a significant deviation, while the remaining top sensors show smaller but still positive Z-Score shifts, collectively contributing to the anomaly attribution.  significant
        9                313 machine-3-7                 False      30 1.29                                      Sensor_30 exhibits a significant positive z-score shift (+1.29 std), indicating an outlier contribution, while the remaining top-5 sensors show negligible or non-significant shifts, implying their values are within expected variation.  significant
       13                135 machine-2-2                 False      10 1.06                                                                                                                    Sensors 10, 11, 13, and 23 show positive Z-Score shifts (+1.00, +0.73, +0.57, +1.06 std) indicating significant deviation from their normal baselines, while  significant
       14                246 machine-3-2                 False       6 1.27                                                                                                                   Sensor_6 exhibits a large negative Z-Score shift (-1.27 std) indicating values well below baseline; these opposing deviations collectively drive the anomaly.        large
       14                246 machine-3-2                 False      12 0.20 Sensor_13 and Sensor_12 show positive Z-Score shifts (+0.22 and +0.20 std) indicating values above baseline, while Sensor_6 exhibits a large negative Z-Score shift (-1.27 std) indicating values well below baseline; these opposing deviations collectively drive the anomaly        large
       14                 14 machine-1-2                 False      30 0.14                                                                                                                          Sensors_30, 20, 32, and 34 also show elevated Z-scores (+1.94, +1.01, +0.20, +0.14 std) suggesting significant anomalies across multiple measurements.  significant
       17                  1 machine-1-1                 False       9 1.02                                                        Sensor_13 (+1.45 std), Sensor_10 (+7.55 std), Sensor_25 (+4.20 std), Sensor_9 (+1.02 std) and Sensor_34 (+0.11 std) show significant z-score shifts from their normal baselines, indicating anomalous telemetry behavior  significant
       17                  1 machine-1-1                 False      13 1.45                                                        Sensor_13 (+1.45 std), Sensor_10 (+7.55 std), Sensor_25 (+4.20 std), Sensor_9 (+1.02 std) and Sensor_34 (+0.11 std) show significant z-score shifts from their normal baselines, indicating anomalous telemetry behavior  significant
       17                  1 machine-1-1                 False      34 0.11                                                                                                                                                     Sensor_34 (+0.11 std) show significant z-score shifts from their normal baselines, indicating anomalous telemetry behavior.  significant
       18                 94 machine-1-8                 False      10 1.17                                                      Sensor_10 shows a +1.17 std Z-Score shift, indicating a significant deviation from its normal baseline, while the other sensors exhibit smaller shifts (+0.09 to +0.46 std), suggesting the anomaly is primarily driven by  significant

## Part 5 — started
- Common gate PASSED; part-specific gate will run before this part's calculations.

## Part 5 — fixed-lexicon exaggeration scan
- PART 5 GATE PASSED: selected=80 evaluated responses; strict verdict joined for all 80.
- Lexicon: significant, substantial, large, massive, extreme, strong, primary driver, dominant, far outside.
- Wrote part5_response_flags.csv, part5_exaggeration_summary.csv, part5_examples.csv, and part5_lexicon_and_function.txt.
- strict_scorer_verdict  n_responses  overstated_n  overstated_share_pct  wilson95_lower_pct  wilson95_upper_pct
                  all           80            17             21.250000           13.712387           31.422156
                False           78            17             21.794872           14.076812           32.160705
                 True            2             0              0.000000            0.000000           65.761977
- Examples:
-  audit_id  segment_global_id     machine strict_scorer_verdict  sensor    z                                                                                                                                                                                                                                                                      wording       lexicon_hits
        5                 10 machine-1-2                 False      18 1.49 Sensor_18 exhibits the largest positive Z-Score shift (+1.49 std) relative to its normal baseline, indicating a significant deviation, while the remaining top sensors show smaller but still positive Z-Score shifts, collectively contributing to the anomaly attribution.        significant
        9                313 machine-3-7                 False      30 1.29                                   Sensor_30 exhibits a significant positive z-score shift (+1.29 std), indicating an outlier contribution, while the remaining top-5 sensors show negligible or non-significant shifts, implying their values are within expected variation.        significant
       13                135 machine-2-2                 False      10 1.06                                                                                                                 Sensors 10, 11, 13, and 23 show positive Z-Score shifts (+1.00, +0.73, +0.57, +1.06 std) indicating significant deviation from their normal baselines, while        significant
       14                246 machine-3-2                 False       6 1.27                                                                                                                Sensor_6 exhibits a large negative Z-Score shift (-1.27 std) indicating values well below baseline; these opposing deviations collectively drive the anomaly.              large
       14                 14 machine-1-2                 False      30 0.14                                                                                                                       Sensors_30, 20, 32, and 34 also show elevated Z-scores (+1.94, +1.01, +0.20, +0.14 std) suggesting significant anomalies across multiple measurements.        significant
       17                  1 machine-1-1                 False       9 1.02                                                    Sensor_13 (+1.45 std), Sensor_10 (+7.55 std), Sensor_25 (+4.20 std), Sensor_9 (+1.02 std) and Sensor_34 (+0.11 std) show significant z-score shifts from their normal baselines, indicating anomalous telemetry behavior.        significant
       17                  1 machine-1-1                 False      13 1.45                                                    Sensor_13 (+1.45 std), Sensor_10 (+7.55 std), Sensor_25 (+4.20 std), Sensor_9 (+1.02 std) and Sensor_34 (+0.11 std) show significant z-score shifts from their normal baselines, indicating anomalous telemetry behavior.        significant
       17                  1 machine-1-1                 False      34 0.11                                                                                                                                                  Sensor_34 (+0.11 std) show significant z-score shifts from their normal baselines, indicating anomalous telemetry behavior.        significant
       21                147 machine-2-4                 False      14 0.69                                                                               Sensor_14 shows a +0.69 std Z-Score shift, indicating a significant deviation from its normal baseline, while the other four sensors exhibit minimal Z-Score shifts within expected variation.        significant
       21                 21 machine-1-3                 False      15 0.44                                                                                                                                                    Sensors 15 and 12 show large positive z-score shifts (+0.44 std), indicating significant deviation from baseline, whereas large; significant

## Part 5 — started
- Common gate PASSED; part-specific gate will run before this part's calculations.

## Part 5 — fixed-lexicon exaggeration scan
- PART 5 GATE PASSED: selected=80 evaluated responses; strict verdict joined for all 80.
- Lexicon: significant, substantial, large, massive, extreme, strong, primary driver, dominant, far outside.
- Wrote part5_response_flags.csv, part5_exaggeration_summary.csv, part5_examples.csv, and part5_lexicon_and_function.txt.
- strict_scorer_verdict  n_responses  overstated_n  overstated_share_pct  wilson95_lower_pct  wilson95_upper_pct
                  all           80            12             15.000000            8.794041           24.413229
                False           78            12             15.384615            9.025505           24.993267
                 True            2             0              0.000000            0.000000           65.761977
- Examples:
-  audit_id  segment_global_id     machine strict_scorer_verdict  sensor    z                                                                                                                                                                                                                                                                      wording       lexicon_hits
        5                 10 machine-1-2                 False      18 1.49 Sensor_18 exhibits the largest positive Z-Score shift (+1.49 std) relative to its normal baseline, indicating a significant deviation, while the remaining top sensors show smaller but still positive Z-Score shifts, collectively contributing to the anomaly attribution.        significant
        9                313 machine-3-7                 False      30 1.29                                   Sensor_30 exhibits a significant positive z-score shift (+1.29 std), indicating an outlier contribution, while the remaining top-5 sensors show negligible or non-significant shifts, implying their values are within expected variation.        significant
       14                246 machine-3-2                 False       6 1.27                                                                                                                Sensor_6 exhibits a large negative Z-Score shift (-1.27 std) indicating values well below baseline; these opposing deviations collectively drive the anomaly.              large
       17                  1 machine-1-1                 False       9 1.02                                                    Sensor_13 (+1.45 std), Sensor_10 (+7.55 std), Sensor_25 (+4.20 std), Sensor_9 (+1.02 std) and Sensor_34 (+0.11 std) show significant z-score shifts from their normal baselines, indicating anomalous telemetry behavior.        significant
       17                  1 machine-1-1                 False      13 1.45                                                    Sensor_13 (+1.45 std), Sensor_10 (+7.55 std), Sensor_25 (+4.20 std), Sensor_9 (+1.02 std) and Sensor_34 (+0.11 std) show significant z-score shifts from their normal baselines, indicating anomalous telemetry behavior.        significant
       17                  1 machine-1-1                 False      34 0.11                                                                                                                                                  Sensor_34 (+0.11 std) show significant z-score shifts from their normal baselines, indicating anomalous telemetry behavior.        significant
       21                147 machine-2-4                 False      14 0.69                                                                               Sensor_14 shows a +0.69 std Z-Score shift, indicating a significant deviation from its normal baseline, while the other four sensors exhibit minimal Z-Score shifts within expected variation.        significant
       21                 21 machine-1-3                 False      15 0.44                                                                                                                                                    Sensors 15 and 12 show large positive z-score shifts (+0.44 std), indicating significant deviation from baseline, whereas large; significant
       86                 86 machine-1-7                 False      16 1.26                                                                                                                                                                Sensor_16 shows a +1.26 std Z-Score shift, indicating a substantial deviation from its normal baseline, while        substantial
       86                 86 machine-1-7                 False      33 0.16                                                                                                                                                                                              Sensor_33 has a negative shift (-0.16 std) indicating no significant deviation.        significant

## Part 5 — started
- Common gate PASSED; part-specific gate will run before this part's calculations.

## Part 5 — fixed-lexicon exaggeration scan
- PART 5 GATE PASSED: selected=80 evaluated responses; strict verdict joined for all 80.
- Lexicon: significant, substantial, large, massive, extreme, strong, primary driver, dominant, far outside.
- Wrote part5_response_flags.csv, part5_exaggeration_summary.csv, part5_examples.csv, and part5_lexicon_and_function.txt.
- strict_scorer_verdict  n_responses  overstated_n  overstated_share_pct  wilson95_lower_pct  wilson95_upper_pct
                  all           80            11             13.750000            7.854714           22.967100
                False           78            11             14.102564            8.061044           23.513978
                 True            2             0              0.000000            0.000000           65.761977
- Examples:
-  audit_id  segment_global_id     machine strict_scorer_verdict  sensor    z                                                                                                                                                                                                                                                                      wording lexicon_hits
        5                 10 machine-1-2                 False      18 1.49 Sensor_18 exhibits the largest positive Z-Score shift (+1.49 std) relative to its normal baseline, indicating a significant deviation, while the remaining top sensors show smaller but still positive Z-Score shifts, collectively contributing to the anomaly attribution.  significant
        9                313 machine-3-7                 False      30 1.29                                   Sensor_30 exhibits a significant positive z-score shift (+1.29 std), indicating an outlier contribution, while the remaining top-5 sensors show negligible or non-significant shifts, implying their values are within expected variation.  significant
       14                246 machine-3-2                 False       6 1.27                                                                                                                Sensor_6 exhibits a large negative Z-Score shift (-1.27 std) indicating values well below baseline; these opposing deviations collectively drive the anomaly.        large
       17                  1 machine-1-1                 False       9 1.02                                                    Sensor_13 (+1.45 std), Sensor_10 (+7.55 std), Sensor_25 (+4.20 std), Sensor_9 (+1.02 std) and Sensor_34 (+0.11 std) show significant z-score shifts from their normal baselines, indicating anomalous telemetry behavior.  significant
       17                  1 machine-1-1                 False      13 1.45                                                    Sensor_13 (+1.45 std), Sensor_10 (+7.55 std), Sensor_25 (+4.20 std), Sensor_9 (+1.02 std) and Sensor_34 (+0.11 std) show significant z-score shifts from their normal baselines, indicating anomalous telemetry behavior.  significant
       17                  1 machine-1-1                 False      34 0.11                                                                                                                                                  Sensor_34 (+0.11 std) show significant z-score shifts from their normal baselines, indicating anomalous telemetry behavior.  significant
       21                147 machine-2-4                 False      14 0.69                                                                               Sensor_14 shows a +0.69 std Z-Score shift, indicating a significant deviation from its normal baseline, while the other four sensors exhibit minimal Z-Score shifts within expected variation.  significant
       86                 86 machine-1-7                 False      16 1.26                                                                                                                                                                Sensor_16 shows a +1.26 std Z-Score shift, indicating a substantial deviation from its normal baseline, while  substantial
       86                 86 machine-1-7                 False      33 0.16                                                                                                                                                                                              Sensor_33 has a negative shift (-0.16 std) indicating no significant deviation.  significant
      166                166 machine-2-5                 False      13 1.18                                                                                                                                                                              Sensor_13 exhibit Z-Score shifts of +1.18 std, indicating significant elevation above baseline;  significant

## Part 5 — started
- Common gate PASSED; part-specific gate will run before this part's calculations.

## Part 5 — fixed-lexicon exaggeration scan
- PART 5 GATE PASSED: selected=80 evaluated responses; strict verdict joined for all 80.
- Lexicon: significant, substantial, large, massive, extreme, strong, primary driver, dominant, far outside.
- Wrote part5_response_flags.csv, part5_exaggeration_summary.csv, part5_examples.csv, and part5_lexicon_and_function.txt.
- strict_scorer_verdict  n_responses  overstated_n  overstated_share_pct  wilson95_lower_pct  wilson95_upper_pct
                  all           80            11             13.750000            7.854714           22.967100
                False           78            11             14.102564            8.061044           23.513978
                 True            2             0              0.000000            0.000000           65.761977
- Examples:
-  audit_id  segment_global_id     machine strict_scorer_verdict  sensor    z                                                                                                                                                                                                                                                                      wording lexicon_hits
        5                 10 machine-1-2                 False      18 1.49 Sensor_18 exhibits the largest positive Z-Score shift (+1.49 std) relative to its normal baseline, indicating a significant deviation, while the remaining top sensors show smaller but still positive Z-Score shifts, collectively contributing to the anomaly attribution.  significant
        9                313 machine-3-7                 False      30 1.29                                   Sensor_30 exhibits a significant positive z-score shift (+1.29 std), indicating an outlier contribution, while the remaining top-5 sensors show negligible or non-significant shifts, implying their values are within expected variation.  significant
       14                246 machine-3-2                 False       6 1.27                                                                                                                Sensor_6 exhibits a large negative Z-Score shift (-1.27 std) indicating values well below baseline; these opposing deviations collectively drive the anomaly.        large
       17                  1 machine-1-1                 False       9 1.02                                                    Sensor_13 (+1.45 std), Sensor_10 (+7.55 std), Sensor_25 (+4.20 std), Sensor_9 (+1.02 std) and Sensor_34 (+0.11 std) show significant z-score shifts from their normal baselines, indicating anomalous telemetry behavior.  significant
       17                  1 machine-1-1                 False      13 1.45                                                    Sensor_13 (+1.45 std), Sensor_10 (+7.55 std), Sensor_25 (+4.20 std), Sensor_9 (+1.02 std) and Sensor_34 (+0.11 std) show significant z-score shifts from their normal baselines, indicating anomalous telemetry behavior.  significant
       17                  1 machine-1-1                 False      34 0.11                                                                                                                                                  Sensor_34 (+0.11 std) show significant z-score shifts from their normal baselines, indicating anomalous telemetry behavior.  significant
       21                147 machine-2-4                 False      14 0.69                                                                               Sensor_14 shows a +0.69 std Z-Score shift, indicating a significant deviation from its normal baseline, while the other four sensors exhibit minimal Z-Score shifts within expected variation.  significant
       86                 86 machine-1-7                 False      16 1.26                                                                                                                                                                Sensor_16 shows a +1.26 std Z-Score shift, indicating a substantial deviation from its normal baseline, while  substantial
      166                166 machine-2-5                 False      13 1.18                                                                                                                                                                              Sensor_13 exhibit Z-Score shifts of +1.18 std, indicating significant elevation above baseline;  significant
      186                186 machine-2-5                 False       1 1.05                                                                                                                                                                                       Sensor_1 shows a +1.05 std Z-Score shift, indicating a large deviation above baseline;        large

## Final packaging
- Refreshed RESULTS.md from the final output CSV files after the Part 5 magnitude-attribution guard update.
- Final checks: common gate passed; Parts 1–5 completed successfully after corrected reruns.

## Gate summary packaged
- Added exact part-specific gate lines to RESULTS.md from final successful runs.

## Task A — per-machine forest-plot table
- Reproduction gate passed: IG top-5 macro=53.925495, Max |z| top-5 macro=59.547825, n_segments=327, Max |z| minus IG=+5.622330.
- Reused aggregate_attributions() from improve_analysis.py: Table 5 stored IG scores and the same fine_raw_window Max |z| ranking code; did not modify results_b.
- Wrote 28 sorted machine rows to results_c/improve/per_machine_top5.csv with requested percent columns.

## Task B — fixed reciprocal rank fusion (exploratory)
- Reproduction gate passed before fusion: IG top-5/top-1/legacy=53.925495/83.891614/62.851890; Max |z|=59.547825/80.381315/72.554739; 327 segments, 28 machines.
- Ran the single pre-specified RRF recipe (constant 60; ties by lower sensor index) for seeds 42, 43, and 44. Reconstructed IG rankings for seeds 43 and 44 and matched their saved per-segment metrics on 327/327 segments each; seed 42 saved ranking cross-check also matched 327/327 rows.
- Paired differences use machine-clustered bootstrap, 10,000 resamples, seed 42. Seed 42 baseline metrics use official Table 5 per-segment metrics to preserve its legacy-overlap tie convention.
- Wrote 9 seed-by-metric rows to results_c/improve/fusion_rrf.csv. No alternative fusion was run.

## Task C — automated-judge audit preparation
- Selected the listed `deepseek-ai/deepseek-v4.1-flash` model, which differs from both explainer model IDs. The API key is read only from the invoking PowerShell process environment variable `NVIDIA_API_KEY`; the script does not load `.env`, log the key, or write its value.
- Implemented `results_c/improve/run_judge_audit.py`; it uses the frozen S6 v2 row-swap packets and cached explanation texts, sends the fixed supported/exaggerated JSON prompt at temperature 0 and max_tokens 300, and caches each response by judge model + prompt hash + item id.
- Local pre-call gate passed: 164/164 cache items present (82 per explainer); strict-scorer/Table 9 counts match exactly: Nemotron 77/80 and Llama 82/82. Two Nemotron source responses are unparsable as JSON; their raw response text is retained as judge input and their scorer result remains excluded from paired agreement.
- No endpoint calls were made in this sandbox. The runner requires an explicit `--run` from the user's PowerShell session. Final CSV and summary will be written after that run returns.

Task C stopped on first judge request: HTTP status=None; TimeoutError: The read operation timed out

## Task C — timeout resume handling
- User-reported first DeepSeek judge request timed out after the configured 180-second read timeout; no HTTP status or response body was returned. Its exact error record remains in judge_audit_cache.jsonl for diagnostics.
- Updated run_judge_audit.py so only a completed HTTP 200 response counts as cached; transport failures do not cause an item to be skipped on resume. Timeout, 429 and 5xx requests receive one retry after 15 seconds. Model, prompt, temperature and max_tokens are unchanged.
- Rechecked script syntax and the 164-item/Table 9 gate locally; passed. No endpoint calls made in this sandbox.

Task C stopped on first judge request: HTTP status=None; TimeoutError: The read operation timed out
Judge audit progress: 10/163 newly requested; total cached=11; failures=0
Judge audit progress: 20/163 newly requested; total cached=21; failures=0
Judge audit progress: 30/163 newly requested; total cached=31; failures=0
Judge audit progress: 40/163 newly requested; total cached=41; failures=0
Judge audit progress: 50/163 newly requested; total cached=51; failures=0
Judge audit progress: 60/163 newly requested; total cached=61; failures=0
Judge audit progress: 70/163 newly requested; total cached=71; failures=0
Judge audit progress: 80/163 newly requested; total cached=81; failures=0
Judge audit progress: 90/163 newly requested; total cached=91; failures=0
Judge audit progress: 100/163 newly requested; total cached=101; failures=0
Judge audit progress: 110/163 newly requested; total cached=111; failures=0
Judge audit progress: 120/163 newly requested; total cached=121; failures=0
Judge audit progress: 130/163 newly requested; total cached=131; failures=0
Judge audit progress: 140/163 newly requested; total cached=141; failures=0
Judge audit progress: 150/163 newly requested; total cached=151; failures=0
Judge audit progress: 160/163 newly requested; total cached=161; failures=0
Judge audit progress: 163/163 newly requested; total cached=164; failures=0

## Task C — GPT-OSS judge output diagnostic and guarded candidate change
- The completed GPT-OSS attempt returned HTTP 200 for all 164 items, but only 35 replies parsed to the required JSON. Cache inspection found 123 responses with null content and 6 truncated/invalid JSON responses; 129 total were unusable. The rates shown in that run are conditional on the 35 parseable replies and are not representative of the full pool. Cohen's kappa was undefined because all paired labels in each parseable subset were positive.
- Set the next candidate to listed instruct model `google/gemma-4-31b-it`; retained all prior model-specific cache entries. Added `--judge-model` selection restricted to IDs in the saved endpoint list and excluded the two explainer IDs.
- Added a first-item JSON-validity guard: if the candidate returns a non-200 response or invalid JSON on the first real item, the run stops before launching the other 163 calls. Response payloads are retained in the cache for diagnosis; null content is no longer converted into literal `None`.
- Syntax and local reproduction-gate checks passed. No endpoint calls made in this sandbox.
- Summary diagnostics now distinguish non-JSON text from empty HTTP-200 content. The next candidate must return valid supported/exaggerated JSON for the first real item before the runner will start the remaining 163 calls.

Task C stopped on first judge request: HTTP status=None; TimeoutError: The read operation timed out

## Task C — candidate preflight command
- Added `--preflight-only` to send one non-study `Reply with the word OK.` request at temperature 0 and max_tokens 30, with a 60-second timeout. It prints the HTTP status, response text, elapsed time, or exact error; it does not log the key or create audit rows.
- Candidate remains `google/gemma-4-31b-it` by default. For a clean route check, override with another saved-list model. Run full audit only after preflight reports PASSED; full audit still requires valid JSON on its first real item.
- Local syntax and reproduction-gate checks passed. No endpoint calls made in this environment.

## Task C — preflight candidate rejected
- User ran tiny preflight for `mistralai/mistral-large-2-instruct`; endpoint returned HTTP 404 Not Found (Function not found for this account) in 0.360 s.
- No audit packets were sent for this candidate and no judge output was added. This candidate is not enabled on the account; the saved /models listing alone does not establish a callable route.

2026-10-29: Added probe_model_routes.py, a sequential tiny-request availability checker over saved chat-like model IDs. It reads NVIDIA_API_KEY only from the calling process, prints no credential, writes each model/status/latency/response/error to model_route_probes.csv immediately, and supports --resume and --limit. No endpoint requests were made from this environment.

2026-10-29: Extended probe_model_routes.py with --retry-timeouts. It selects only model IDs whose prior recorded error contains 'timeout', appends their new measurements to model_route_probes.csv, and accepts a higher --timeout. No endpoint requests were made from this environment.

2026-10-29: Added --judge-json-preflight-only to run_judge_audit.py. It sends one real frozen judge prompt using the chosen model, evaluates whether the reply parses to the required supported/exaggerated JSON schema, and immediately caches the model-specific response so a subsequent full --run reuses it. No endpoint requests were made from this environment.

Task C stopped on first judge request: HTTP status=None; TimeoutError: The read operation timed out

2026-10-29: Added --timeout-seconds to run_judge_audit.py so the user can resume a slow judge route with a longer per-request read timeout. Default remains 180 seconds; prompts, model defaults, parser, and scorer are unchanged. No endpoint requests were made from this environment.
Judge audit progress: 10/162 newly requested; total cached=12; failures=0
Judge audit progress: 20/162 newly requested; total cached=22; failures=0
Judge audit progress: 30/162 newly requested; total cached=32; failures=3
Judge audit progress: 40/162 newly requested; total cached=42; failures=5
Judge audit progress: 50/162 newly requested; total cached=52; failures=8
Judge audit progress: 60/162 newly requested; total cached=62; failures=12
Judge audit progress: 70/162 newly requested; total cached=72; failures=14
Judge audit progress: 80/162 newly requested; total cached=82; failures=14
Judge audit progress: 90/162 newly requested; total cached=92; failures=15
Judge audit progress: 100/162 newly requested; total cached=102; failures=21
Judge audit progress: 110/162 newly requested; total cached=112; failures=25
Judge audit progress: 120/162 newly requested; total cached=122; failures=25
Judge audit progress: 130/162 newly requested; total cached=132; failures=28
Judge audit progress: 140/162 newly requested; total cached=142; failures=34
Judge audit progress: 150/162 newly requested; total cached=152; failures=37
Judge audit progress: 160/162 newly requested; total cached=162; failures=38
Judge audit progress: 162/162 newly requested; total cached=164; failures=38
Judge audit progress: 10/37 newly requested; total cached=137; failures=2
Judge audit progress: 20/37 newly requested; total cached=147; failures=8
Judge audit progress: 30/37 newly requested; total cached=157; failures=8
Judge audit progress: 37/37 newly requested; total cached=164; failures=9
Judge audit progress: 8/8 newly requested; total cached=164; failures=1

2026-10-30: Added run_judge.py for the gated automated-judge audit. It uses the fixed prompt, API credentials from NVIDIA_API_KEY only, judge model from JUDGE_MODEL, per-item SHA-256 cache keys, immediate JSONL persistence, a single malformed-JSON retry, and includes raw explanations for Nemotron segments 215 and 287 with explainer_parse_ok=false. Offline mock test passed for all 164 items and verified cache replay caused no duplicate calls. Recomputed gates: Nemotron magnitude 77/80, lenient 74/75, strict 60/75; Llama magnitude 82/82, lenient 61/77, strict 58/77. Nemotron magnitude exclusions are exactly 215 and 287. No network/API calls made.
Judge audit progress: 10/164 new items; terminal cache keys=10; failures logged=10
Judge audit progress: 20/164 new items; terminal cache keys=20; failures logged=20
Judge audit progress: 30/164 new items; terminal cache keys=30; failures logged=30
Judge audit progress: 40/164 new items; terminal cache keys=41; failures logged=41
Judge audit progress: 50/164 new items; terminal cache keys=50; failures logged=50
Judge audit progress: 60/164 new items; terminal cache keys=60; failures logged=60
Judge audit progress: 70/164 new items; terminal cache keys=70; failures logged=70
Judge audit progress: 80/164 new items; terminal cache keys=80; failures logged=80
Judge audit progress: 90/164 new items; terminal cache keys=90; failures logged=90
Judge audit progress: 100/164 new items; terminal cache keys=100; failures logged=100
Judge audit progress: 110/164 new items; terminal cache keys=110; failures logged=110
Judge audit progress: 120/164 new items; terminal cache keys=120; failures logged=120
Judge audit progress: 130/164 new items; terminal cache keys=130; failures logged=130
Judge audit progress: 140/164 new items; terminal cache keys=140; failures logged=140
Judge audit progress: 150/164 new items; terminal cache keys=150; failures logged=150
Judge audit progress: 160/164 new items; terminal cache keys=160; failures logged=160
Judge audit progress: 164/164 new items; terminal cache keys=164; failures logged=164
