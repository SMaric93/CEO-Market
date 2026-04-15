# Estimation guide

## Notation

- `e`: succession event
- `f`: focal firm
- `i`: candidate
- `t`: succession year

## Main reduced-form specifications

\[
Y_e = \beta_1 \text{ReleaseCount}_{e,730d,60mi,outind}
    + \beta_2 \text{MaxReleasedTaskFit}_{e}
    + \beta_3 \text{BenchIndex}_{e}
    + \gamma' X_e + \text{FE} + \varepsilon_e
\]

In the package:

- `release_count_730d_60mi_outind` is the main access-shock variable
- `max_released_task_fit_z` is the best structured fit among released candidates in the main pool
- `bench_index_z` is the standardized internal-bench summary used in the baseline and heterogeneity tables
- `X_e` is constructed from focal-firm `t-1` variables only

The package also implements the tighter within-market-year specification:

\[
Y_e = \beta_1 \text{AvgReleasedTaskFit}_{e}
    + \beta_2 \text{MaxReleasedTaskFit}_{e}
    + \beta_3 \text{TopQuartileReleasedCount}_{e}
    + \beta_4 \text{BenchIndex}_{e}
    + \gamma' X_e + \text{FE}_{market \times year} + \text{FE}_{ff49} + \varepsilon_e
\]

## Outcome blocks

### Successor-origin outcomes

- `outsider_flag`
- `distant_external_flag`
- `log1p_distance_miles`
- `from_release_pool_flag`

### Realized-fit outcomes

- `realized_task_fit_z`
- `gap_accessible_task_fit_z`
- `text_fit_tfidf_cosine` when source text inputs are available

### Post-succession outcomes

For horizon `h in {1,3}` in the main paper tables:

\[
\Delta Y_{e,h} = \frac{1}{h}\sum_{s=1}^{h} Y_{f,t+s} - \frac{1}{2}\sum_{s=1}^{2} Y_{f,t-s}
\]

The event panel stores:

- `delta_roa_h1`, `delta_roa_h3`
- `delta_q_h1`, `delta_q_h3`
- `delta_roa_resid_h1`, `delta_roa_resid_h3`
- `delta_q_resid_h1`, `delta_q_resid_h3`
- optional `delta_tfp_op_h3`, `delta_tfp_lp_h3`

## Task-alignment fit

The main fit score follows the specification layer:

\[
\text{TaskFitRaw}_{if}
= -\frac{1}{5}\sum_{d=1}^{5} (N_{fd} - E_{id})^2 + \text{IndustryFit}_{if}
\]

with event-candidate standardization:

\[
\text{task_fit_z_if} = \frac{\text{TaskFitRaw}_{if} - \mu}{\sigma}
\]

Current dimensions:

1. innovation
2. turnaround
3. scale / complexity
4. operating / capital intensity
5. growth / market expectations

The package stores:

- `task_fit_raw_if`
- `task_fit_z_if`
- dimension-specific need and experience components
- event-level summaries including `realized_task_fit_z`, `max_released_task_fit_z`, and `gap_accessible_task_fit_z`

## Predictive fit

When enabled, predictive fit uses cross-fitting so the same succession event is never used for both training and scoring. The implemented decomposition follows:

- `PortableQuality_if`: out-of-fold prediction from a model on `Z_i`
- `PredictiveFit_if`: out-of-fold prediction from interaction terms `Z_i ⊗ X_f`

## IV specification

The default IV command estimates realized-fit mediation using:

- endogenous regressor: `realized_task_fit_z`
- instrument: `release_count_730d_60mi_outind`
- controls: `max_released_task_fit_z`, `bench_index_z`, and the core lagged control set

On very small synthetic samples, the command may return an unavailable/insufficient-sample record instead of forcing a rank-deficient 2SLS fit.
