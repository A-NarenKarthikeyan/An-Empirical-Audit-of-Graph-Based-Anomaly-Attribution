# Explanation of Stage S5 Confidence Interval Revision for top5_mass

## Root Cause of the Discrepancy in `results_b/s5_signal_screening.csv`

In the original Stage S5 output (`results_b/s5_signal_screening.csv`), row 5 reported:
- Signal: `top5_mass` (Confirmation Phase, 14 machines, $N = 153$ segments)
- Spearman $\rho = -0.0502$
- 95% Confidence Interval: `[-0.0875, 0.3932]`

This CI was visibly asymmetric and shifted positive relative to the negative point estimate ($\rho = -0.0502$).

### Mathematical Cause
In the initial `scratch/option_b/s5_abstention.py` script:
1. The pooled correlation $\rho = -0.0502$ was calculated across all 153 segments pooled across the 14 confirmation machines simultaneously.
2. However, the confidence interval was calculated by:
   - Computing separate within-machine Spearman correlations $r_m$ for each individual machine $m \in \{1, \dots, 14\}$ that had $\ge 3$ segments.
   - Across these machines, individual within-machine correlations varied widely (e.g., from $-0.759$ to $+1.000$), with a machine-level mean of $+0.1515$.
   - Passing that dictionary of per-machine scalar values to `cluster_bootstrap_ci(per_machine_rho)`, which resampled the machine scalars and computed the CI around their machine-level unweighted mean ($+0.1515$), yielding `[-0.0875, 0.3932]`.
   - Because within-machine attribution dynamics differ from cross-machine pooled correlations (a classic aggregation / Simpson's paradox effect), the machine-level average correlation was positive while the overall pooled sample correlation was slightly negative ($\rho = -0.0502$).

## Correction in `results_b/final/s5_signal_screening_v2.csv`
Per instructions, `s5_signal_screening_v2.csv` implements standard cluster-bootstrap resampling of the pooled statistic:
1. In each of the $B = 10,000$ bootstrap replicates (random seed 42), the clusters (machines) are resampled with replacement ($K=14$ for discovery/confirmation splits; $K=28$ for all 327 segments).
2. For each resample, all segment observations belonging to the sampled machines are concatenated.
3. The pooled Spearman rank correlation $\rho^{(b)}$ and the pooled AUROC$^{(b)}$ are computed directly on the concatenated sample.
4. The 95% CI is obtained from the 2.5th and 97.5th percentiles of the empirical bootstrap distributions.

### Resulting Corrected CIs
- **Confirmation `top5_mass` ($N = 153$):**
  - Spearman $\rho = -0.0502$, 95% Cluster-Bootstrap CI: `[-0.3109, 0.2495]`
  - AUROC = $0.5168$, 95% Cluster-Bootstrap CI: `[0.3660, 0.6569]`
  - Both intervals properly cover and center around the respective sample estimates ($\rho = -0.0502$ and AUROC = $0.5168$), firmly confirming that the signal is statistically indistinguishable from chance on the confirmation cohort.
- **Full Pool `top5_mass` ($N = 327$ segments, all 28 machines):**
  - Spearman $\rho = 0.0945$ ($p = 0.0880$), 95% Cluster-Bootstrap CI: `[-0.1382, 0.2825]`
  - AUROC = $0.5579$, 95% Cluster-Bootstrap CI: `[0.4239, 0.6653]`
