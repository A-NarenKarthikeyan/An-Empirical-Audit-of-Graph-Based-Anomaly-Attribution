# S6 v2 second-model results: meta/llama-3.2-11b-vision-instruct

Total wall-clock seconds: 0.000

## Responses by call type

Row-swap: 82; perturbed: 77; control: 77.

## Table 9 metrics

| Metric | Evaluated | Passed | Rate | Wilson 95% CI | Excluded | Not testable |
|---|---:|---:|---:|---|---:|---:|
| Individual honesty | 82 | 82 | 1.0000 | [0.9552, 1.0000] | 0 | 0 |
| Aggregate consistency | 82 | 32 | 0.3902 | [0.2918, 0.4985] | 0 | 0 |
| Evidential lenient | 77 | 61 | 0.7922 | [0.6888, 0.8678] | 0 | 0 |
| Evidential strict | 77 | 58 | 0.7532 | [0.6465, 0.8360] | 0 | 0 |

## Paired control counts

Valid pairs: 77; order: perturbed-only pass / both pass / neither pass / control-only pass.
Lenient: 48 / 13 / 16 / 0.
Strict: 58 / 0 / 19 / 0.

## Exclusions / unparsable responses

Count: 0. See `excluded_or_unparsable.csv` for audit IDs and reasons.
Failed after retries: 0 packet calls; audit IDs: none.

## Rule-based exaggeration

Flagged 21/82 evaluated row-swap responses (0.2561); Wilson 95% CI [0.1740, 0.3600]. See `exaggeration_flags.json` for audit IDs and wording.

## Response examples differing from Nemotron

- Segment 1 — new model: The anomaly is attributed to the significant z-score shifts observed in Sensor_13 (+7.55 std), Sensor_25 (+4.20 std), and Sensor_30 (+6.70 std), indicating substantial deviations from their normal historical baselines. Nemotron: Sensor_13 exhibits the highest Integrated Gradients attribution (76%) and a Z-Score shift of +7.55 std, far above its normal baseline (0.0002 +/- 0.0013), indicating a dominant deviation that drives the anomaly.
- Segment 2 — new model: The anomaly is primarily driven by Sensor_30 (Z-Score Shift = +5.45 std) and Sensor_10 (Z-Score Shift = +21.77 std), which exhibit significant deviations from their normal historical baselines, indicating a substantial impact on the system's behavior. Nemotron: Sensor_30, Sensor_10, and Sensor_24 each show large positive Z‑Score shifts (+5.45, +21.77, +8.86 std respectively) and observed means far exceeding their normal baselines, indicating they are the primary drivers of the anomaly.
- Segment 6 — new model: The anomaly is caused by significant z-score shifts in Sensor_9 (+1.61 std) and Sensor_13 (+4.53 std), indicating substantial deviations from their normal historical baselines. Nemotron: Sensor_13 shows the highest Z‑score shift (+4.53 σ), indicating a extreme deviation from its baseline, while Sensor_14 (+2.60 σ) and Sensor_9 (+1.61 σ) also exhibit significant upward shifts, together representing the primary drivers of the anomaly.