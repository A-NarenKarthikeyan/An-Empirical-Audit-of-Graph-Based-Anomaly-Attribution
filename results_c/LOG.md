## Adaptive Conformal Set Computation

- Timestamp: 2026-10-28T15:37:16.221002
- Input rankings: s3_cached_rankings.json
- Target sensors file: faithfulness_results_v2.csv
- Output CSV: adapt_conformal_results.csv
- Alphas evaluated: 0.1, 0.2, 0.3


## Baseline run 2026-10-29T00:01:20+00:00  config_hash=5242435b
- Gate 1 (reproduction): 326/327 exact matches
- Gate 2 (IG macro): top1=83.89 top5=53.93 legacy=62.84
- Gate 3 (random): top1=23.02 top5=13.02
- Output: results_c/baseline_metrics.csv, baseline_per_segment.csv
    method         top1 [95%CI]         top5 [95%CI]       legacy [95%CI] top5 diff vs IG [95%CI]
        IG 83.89 [78.21, 89.50] 53.93 [47.47, 60.31] 62.84 [58.51, 67.23]    +0.00 [+0.00, +0.00]
 max_abs_z 80.38 [73.32, 86.90] 59.55 [52.33, 66.67] 72.55 [68.20, 77.02]    +5.62 [+2.98, +8.27]
mean_abs_z 53.78 [44.21, 62.91] 40.97 [35.27, 46.68] 52.12 [45.25, 59.02]  -12.95 [-18.86, -7.30]
mean_shift 49.78 [40.02, 59.18] 34.88 [29.39, 40.26] 47.56 [39.85, 55.40] -19.04 [-26.41, -11.97]
    random 23.02 [18.37, 29.59] 13.02 [12.79, 13.25] 23.07 [18.58, 29.35] -40.90 [-47.27, -34.44]

## Conformal max|z| run 2026-10-29T00:06:05+00:00
- Gate passed: IG q_hat (36, 31, 28, 24) and held-out coverage (92.47, 81.71, 91.67, 81.96)
- Paired Top-1 Hit Difference (IG - max|z|): +3.51% [-2.65%, +10.83%]
ranking  rho  alpha  nominal_cov  q_hat q_hat_frac mean_min_prefix            held_out_cov loo_pooled_cov
 max|z|  1.0    0.1          0.9     30     78.95%           13.17 91.89% [86.83%, 96.95%]         91.13%
 max|z|  1.0    0.2          0.8     27     71.05%           13.17 82.27% [74.07%, 90.48%]         80.73%
 max|z|  0.8    0.1          0.9     26     68.42%           10.58 92.09% [85.83%, 98.35%]         90.83%
 max|z|  0.8    0.2          0.8     21     55.26%           10.58 81.91% [73.37%, 90.44%]         81.35%

## Multi-Seed Attribution & Conformal Evaluation (Seeds 42, 43, 44) 2026-10-29T00:23:43+00:00
- Gate verified on Seed 42: IG macro top1=83.89, top5=53.93, legacy=62.84; q_hat=36,31,28,24
- max|z| baseline top-5 recall: 59.55% (identical to seed-42)
- Outputs: results_c/seeds/
                              metric  seed_42  seed_43  seed_44  mean   sd
                    IG Top-1 Hit (%)    83.89    84.23    83.56 83.90 0.33
                 IG Top-5 Recall (%)    53.93    53.11    54.55 53.86 0.72
               IG Legacy Overlap (%)    62.84    62.64    63.32 62.93 0.35
    Top-5 Diff: IG minus max|z| (pp)    -5.62    -6.44    -5.00 -5.69 0.72
    Conformal q_hat (rho=1.0, a=0.1)    36.00    37.00    36.00 36.33 0.58
Held-Out Cov Mean % (rho=1.0, a=0.1)    92.47    92.69    91.88 92.35 0.42
   LOO Pooled Cov % (rho=1.0, a=0.1)    91.74    92.97    90.52 91.74 1.22
    Conformal q_hat (rho=1.0, a=0.2)    31.00    30.00    30.00 30.33 0.58
Held-Out Cov Mean % (rho=1.0, a=0.2)    81.71    81.79    82.19 81.90 0.26
   LOO Pooled Cov % (rho=1.0, a=0.2)    81.04    81.35    81.35 81.24 0.18
    Conformal q_hat (rho=0.8, a=0.1)    28.00    30.00    30.00 29.33 1.15
Held-Out Cov Mean % (rho=0.8, a=0.1)    91.67    91.47    91.49 91.55 0.11
   LOO Pooled Cov % (rho=0.8, a=0.1)    90.21    90.21    89.30 89.91 0.53
    Conformal q_hat (rho=0.8, a=0.2)    24.00    25.00    24.00 24.33 0.58
Held-Out Cov Mean % (rho=0.8, a=0.2)    81.96    81.66    82.21 81.94 0.28
   LOO Pooled Cov % (rho=0.8, a=0.2)    81.04    81.04    82.26 81.45 0.71

## Human Rater Sheets Generation 2026-10-29
- Source: results_b/final/audit_blind.csv (30 cases) shuffled once with seed 42
- Shuffled audit_ids: [28, 16, 24, 18, 9, 10, 29, 25, 13, 1, 5, 17, 6, 14, 12, 23, 2, 3, 26, 4, 22, 27, 19, 30, 21, 8, 11, 15, 20, 7]
- Columns: ['audit_id', 'prompt_text', 'llm_response', 'magnitudes_correct', 'only_packet_sensors', 'no_exaggeration', 'overall_faithful', 'comment']
- Output Sheet A: results_c/rater_sheet_A.csv (30 rows)
- Output Sheet B: results_c/rater_sheet_B.csv (30 rows)
- Practice Cases: results_c/practice_cases.csv (3 rows from non-audit segments [2, 6, 14])
