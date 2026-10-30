# Graph-based anomaly explanation audit

This repository contains the source, frozen configurations, derived outputs, caches, and paper tables for the SMD anomaly-detection and explanation-faithfulness study. It is a reproducibility archive; the committed outputs are the records used for the paper and should not be interpreted as a fresh rerun.

## Contents

- `src/`, `scripts/`, `scratch/option_b/`: detector, attribution, audit, and orchestration code.
- `configs/`: development configuration; frozen experiment configurations are in `results_b/` and `results_b/final/`.
- `results_b/`: detector runs, checkpoints, metrics, attribution, calibration, S6/S7 records, manifests, and paper summaries.
- `results_c/`: follow-up analyses, second-model faithfulness results and cache, model availability records, and automated-judge artifacts under `improve/`.
- `results/`: earlier result artifacts and faithfulness caches retained for provenance.
- `requirements.txt`: Python package requirements; `results_b/final/environment.txt` records the saved environment snapshot.
- `data/README.md`: location and source for the raw SMD dataset, which is not included.

## Data

Obtain the Server Machine Dataset (SMD) from the [OmniAnomaly repository](https://github.com/NetManAIOps/OmniAnomaly), following its dataset terms. Place the unmodified data under `data/raw/` in the layout expected by `configs/config.yaml` and the preparation scripts. This release intentionally excludes all raw SMD data.

## Environment and configuration provenance

The saved environment snapshot records Python 3.11.3, PyTorch 2.14.0+cpu, PyTorch Geometric 2.8.0.post1, scikit-learn 1.9.1, NumPy 2.4.6, pandas 3.0.5, and SciPy 1.17.1. The snapshot reports an 8-logical-core Windows development machine. The run log separately reports Python 3.13, PyTorch 2.10.0+cpu, PyG 2.8.1, scikit-learn 1.7.2 on a 12-core host with 11 workers and one Torch thread per worker. The archive does not establish that the detector run used the same package environment as the saved 3.11.3 snapshot.

- Frozen detector config (`results_b/config_frozen.json`): SHA-256 `f09541fda4f2aac730456ceb0c954ca51750b343d6c42bce535cf2beae33ea95`.
- S6 v2 config (`results_b/final/config_frozen_v2.json`): SHA-256 `a68dde35d9e5d86479e5a8bd8bef34c5b564d761d8e746c023238e3c07aa17ae`.
- The v2 configuration adds the language-model name to the frozen detector configuration.

## Reproduction commands

Run commands from the repository root in PowerShell after installing the environment and placing SMD data as described above. These commands can rerun analyses and overwrite generated outputs; use a copy of the archive for reruns. S6 and judge execution contact the configured model endpoint and require credentials supplied through environment variables. The offline commands do not need credentials.

```powershell
# S0: inventory/config freeze and timing plan
python .\scratch\option_b\s0_prep.py
python .\scratch\option_b\s0_timing.py

# S1: detector runs
python .\scratch\option_b\s1_orchestrator.py

# S2–S8 individually
python .\scratch\option_b\s2_baselines.py
python .\scratch\option_b\s3_attribution.py
python .\scratch\option_b\s4_conformal.py
python .\scratch\option_b\s5_abstention.py
python .\scratch\option_b\s6_faithfulness.py
python .\scratch\option_b\s7_human_audit.py
python .\scratch\option_b\s8_summary_manifest.py

# Original orchestration for S1–S8
python .\scratch\option_b\run_option_b.py

# Offline cached second-model summary (does not call the model)
python .\results_c\second_model\run_s6_v2.py summarize

# Offline follow-up analyses and fixed RRF analysis
python .\results_c\improve\improve_analysis.py
python .\results_c\improve\fusion_rrf.py

# Automated-judge script: offline mock integration test only
python .\results_c\improve\run_judge.py --mock-test
```

The saved automated-judge audit was run with `results_c/improve/run_judge_audit.py` using `openai/gpt-oss-20b`; model calls require `NVIDIA_API_KEY` set locally. Do not place credentials in files or logs. The original full-stage command is provided for reproducibility, but S6 may stop at its configured gate when its endpoint or model is unavailable. No command above changes the frozen prompts or scorer rules.

## Recorded reproduction gates

Expected values recorded in the project artifacts:

- S0 inventory: 28 machines and 327 labelled segments; the selected plan is the full seed set 42, 43, and 44.
- S1: 113 detector runs, 0 failures, 11,097.52 s wall time in the run log.
- Seed-42 attribution machine-macro results: IG top-1 83.89%, top-5 53.93%, legacy overlap 62.84%; Max |z| top-5 59.55%.
- Prefix conformal coverage gate: 92.47%, 81.71%, 91.67%, and 81.96% for the four saved settings (see the conformal summary for their order and calibration details).
- Nemotron S6 v2: honesty 77/80, aggregate consistency 52/80, evidential lenient 74/75, evidential strict 60/75; control counts lenient 59/12/1/0 and strict 57/0/15/0 over 72 valid pairs.
- Llama 3.2 11B S6: honesty 82/82, evidential lenient 61/77, evidential strict 58/77.
- Automated judge audit: 164 source items; the saved summary reports parse/invalid-schema responses rather than silently dropping them.

See `results_b/LOG.md`, `results_b/final/`, `results_c/second_model/`, and `results_c/improve/` for the detailed gates, exclusions, and outputs. Historical run logs may record different environment snapshots; consult the actual run record when interpreting a result.

## Checkpoints

Detector `.pt` checkpoints are included because the release is below the 500 MiB size limit. Their paths and sizes are listed in `CHECKPOINTS.md`. Checkpoints can be excluded from a derivative source-only checkout; the recorded metrics and paper tables remain in `results_b/`.

## License and citation

Code and accompanying project artifacts are released under the MIT License in `LICENSE`. The raw SMD data is not included and remains subject to its source terms. Please cite the associated paper when reusing the methods or outputs.
