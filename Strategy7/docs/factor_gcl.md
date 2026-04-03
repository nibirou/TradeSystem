# DFQ-FactorGCL in Strategy7

This document maps the report model design to the implementation added in Strategy7.

## 1. Report-to-Code Mapping

The implementation follows the report's four core blocks:

1. HyperGCN prior beta layer:
   - Formula: `Y = sigma(Dv^-1/2 H We De^-1 H^T Dv^-1/2 X Theta)`
   - Code: `HyperGCNLayer.forward(...)` in `factor_gcl_model.py`
   - Predefined beta matrix `beta_s` is built from static stock concepts (`industry_bucket`, `board_type`, plus `all`).

2. HyperGCN hidden beta layer with learnable factor prototypes:
   - Formula: `beta_h = sigmoid(X_t,1 * C^T)`
   - Code: `beta_hidden = sigmoid(x1 @ prototypes.T)`
   - Hidden factor prototype count is controlled by `--fgcl-num-factor` (default 48 as report).

3. Individual alpha residual block:
   - Formula: `alpha = phi_alpha(X_t,2)`
   - Code: `alpha_layer = Linear + LeakyReLU`, input is residual after prior/hidden layers.

4. Dual-path TRCL training objective:
   - Historical path predicts target return.
   - Future path uses independent feature extractor but shares prior/hidden/alpha modules and shared beta matrices from history path.
   - Loss:
     - `L_total = L_mse + gamma * (L_temporal_infoNCE + L_cross_section_infoNCE)`
   - Improved temporal InfoNCE implemented with:
     - positive: `sign(sim) * sim^2`
     - negatives: `sim^2`
     - temperature `tau`

## 2. Hyperparameters (Report-aligned Defaults)

Implemented CLI defaults aligned with report:

- `--fgcl-seq-len 30`
- `--fgcl-future-look 20`
- `--fgcl-hidden-size 128`
- `--fgcl-num-layers 2`
- `--fgcl-num-factor 48`
- `--fgcl-gamma 1.0`
- `--fgcl-tau 0.25`
- `--fgcl-epochs 200`
- `--fgcl-lr 9e-5`
- `--fgcl-early-stop 20`
- `--fgcl-smooth-steps 5`
- `--fgcl-per-epoch-batch 100`
- `--fgcl-batch-size -1`
- `--fgcl-label-transform csranknorm`

## 3. Strategy7 Adaptation Details

### 3.1 Input/Output contract

The model fully matches the stock-selection interface:

- `fit(train_df, factor_cols, target_col)`
- `predict_score(df, factor_cols)`
- `fill_values()`
- `save(folder, run_tag)`

### 3.2 Score-direction stabilization

Deep models may occasionally converge to a sign-inverted score (negative validation rankIC while preserving spread).
To reduce this instability, the implementation calibrates score direction after training:

- compute validation rankIC on the best-smoothed checkpoint
- if validation rankIC is negative, multiply raw score by `-1` before cross-sectional ranking

This keeps `pred_score` semantics stable (`higher score => stronger expected return`).

### 3.3 Sequence construction

For each stock code, rows are sorted by signal timestamp and transformed into:

- past sequence: `[t-seq_len+1, ..., t]`
- future sequence: `[t+1, ..., t+future_look]`

Only samples with complete past+future windows are used for TRCL training.

### 3.4 Score format in pipeline

The model predicts raw continuous scores, then converts to cross-sectional percentile rank per trading day to match Strategy7's score usage (`pred_score` in `[0,1]`).

## 4. New CLI and Scripts

### 4.1 New model type

Use:

- `--stock-model-type factor_gcl`

Aliases also accepted in factory:

- `factorgcl`
- `dfq_factorgcl`

### 4.2 Added scripts

- `scripts/run_strategy7_04_factorgcl_daily.ps1`
- `scripts/run_strategy7_04_factorgcl_daily.cmd`
- `scripts/run_strategy7_04_factorgcl_daily.sh`

## 5. Dependency

FactorGCL requires PyTorch. If torch is missing, model training raises a clear runtime error.

Install example:

```bash
pip install torch
```

## 6. Notes on Current Implementation Scope

To fit Strategy7's generic data schema and keep compatibility:

- Predefined concepts are built from available static fields (`industry_bucket`, `board_type`) rather than external thematic concept files.
- The report's "complete" end-to-end data engineering beyond Strategy7's current factor panel (e.g., proprietary concept tags) is not embedded.

The architecture, training logic, and objective design are reproduced and adapted to the existing framework interfaces.
