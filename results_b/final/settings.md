# Experimental Settings & Hyperparameters

**Configuration Frozen SHA-256**: `f09541fda4f2aac730456ceb0c954ca51750b343d6c42bce535cf2beae33ea95`

## 1. Detector Architecture & Training Settings
- **Model Architecture**: `GATAnomalyDetector` (PyTorch Geometric)
- **Input Dimension**: 6 (for 6-D temporal features) / 2 (for 2-D ablation: mean, std)
- **Hidden Width**: 64
- **Attention Heads**: 4
- **Layers**: 2 GATConv layers with ELU activations
- **Edge Feature Dimension**: 1
- **Dropout**: 0.20
- **Optimizer**: `torch.optim.Adam`
- **Learning Rate**: 0.001 ($1 \times 10^{-3}$)
- **Weight Decay**: 0.0001 ($1 \times 10^{-4}$)
- **Batch Size**: 64
- **Max Epochs**: 20
- **Early-Stopping Rule**: 10 consecutive epochs without validation loss improvement (`patience=10`)
- **Loss Function**: `torch.nn.BCEWithLogitsLoss()` (unweighted binary cross entropy)
- **Class Weighting**: None (`pos_weight=None`)
- **Device**: CPU

## 2. Sliding Window, Purging & Graph Construction
- **Window Length ($T$)**: 100 timesteps
- **Window Stride**: 1 timestep
- **Split Ratio**: 70% Train, 15% Validation, 15% Test (strictly contiguous temporal blocks)
- **Purge Gap**: 99 windows between Train/Val and Val/Test (guarantees zero boundary window overlap)
- **Node Feature List (6-D)**:
  1. Mean
  2. Standard deviation
  3. Skewness
  4. Kurtosis
  5. Maximum absolute deviation
  6. FFT dominant frequency magnitude
- **Node Feature List (2-D Ablation)**:
  1. Mean
  2. Standard deviation
- **Graph Construction Rule**:
  - Pearson absolute correlation threshold $|r| \ge 0.60$
  - Computed strictly on each machine's own **Train** block (no test leakage)
  - Self-loops added; edge weights set to $|r|$

## 3. Threshold Grid & Detection Inference
- **Grid Search**: 0.01 to 0.99 with step 0.01 (99 evaluation thresholds)
- **Optimization Criterion**: Maximize $F_1$ score on the validation block
- **Test Application**: Best threshold from validation applied directly to test block probabilities

## 4. Attribution Settings
- **Methods Benchmarked**:
  1. Integrated Gradients (IG): 50 interpolation steps, baseline = zero tensor
  2. GNNExplainer: 60 epochs per segment, learning rate = 0.05
  3. Raw GAT Attention: Mean attention weights from Layer 2
- **Fine Anomaly Window Margin**: $\pm 15$ timesteps around ground-truth interpretation segment boundary

## 5. Conformal Calibration Settings
- **Conformal Predictor**: Inductive Conformal Prediction
- **Nonconformity Score**: Smallest prefix length of IG sensor ranking required to reach containment target
- **Targets Evaluated**:
  - Containment $\rho \in \{1.00, 0.80\}$
  - Significance $\alpha \in \{0.10, 0.20\}$ (Nominal coverage: 90%, 80%)
- **Quantile Rule**: Finite-sample adjusted quantile $\hat{q} = \lceil (n + 1)(1 - \alpha) \rceil / n$
- **Protocols**:
  1. Pooled held-out: 1,000 random 50/50 splits (seed 42)
  2. Leave-One-Machine-Out (LOMO): 27 train machines, 1 test machine across all 28 folds

## 6. Selective Abstention Gate
- **Explanation Top-$K$ Evidence**: 5 sensors
- **Variance Floor**: Sensors with baseline std $< 0.001$ filtered out to prevent dead-sensor ranking
- **Abstention Mass Threshold ($\tau$)**: Minimum 70% cumulative IG mass required on Top-$K$ sensors
- **Evaluation Sweep**: $\tau \in [0.00, 0.95]$ with step 0.05
