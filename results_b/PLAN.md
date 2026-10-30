# EXECUTION PLAN (GATE G0)

- **Selected Plan**: `FULL`
- **Plan Description**: Plan FULL: seeds {42, 43, 44} on all 28 machines (84 detector runs).
- **Hardware Profile**: CPU count = 12, Process Pool Workers (`N_WORKERS`) = 11
- **Benchmark Timing on machine-1-1**:
  - Time per detector run (data prep + 20-epoch training + val search + test eval): `926.33s` (15.44 min)
  - Time per IG attribution segment: `0.4389s`
- **Projections**:
  - Projected FULL Plan Wall-Clock: `2.04 hours`
  - Projected REDUCED Plan Wall-Clock: `1.02 hours`
  - Projected Seed-42-Only Wall-Clock: `0.65 hours`
- **Decision Rule Triggered**:
  - `total_hours_full <= 12.0 h`: YES (Plan FULL selected)
- **Multi-Seed Coverage**:
  - Total detector runs planned: 84
  - Seed list per machine saved in `results_b/plan_seeds.json`
