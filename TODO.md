# TODO

Tracked v2 work and known limitations. Items here are intentional skips,
not bugs.

## LightGBM improvements

### Residual learning (highest expected ROI)
Predict `y(T) - y(T-24)` instead of raw `y(T)`. The model only learns the
deviation from yesterday; persistence is built into the architecture for
free. Typically the single biggest win on heavily autocorrelated time-series.

Implementation: change `TARGET` in `forecasting/features.py` to a residual,
add helper to reconstruct `y_pred = y(T-24) + residual_pred` at inference.

### Hyperparameter search
Run Optuna (50 trials) over `n_estimators`, `learning_rate`, `num_leaves`,
`min_data_in_leaf`, `feature_fraction`. Use val set as the objective.

### Switch `objective='mae'`
Default is squared error. Since we report MAE in the scoreboard, training
on MAE directly is a tiny code change with a small consistent gain.

### Add features
- Heating/cooling degree days (HDD, CDD) — derived from tmax/tmin
- 336h lag (2 weeks ago, same hour)
- Rolling weather stats (7-day mean tmax, etc.)
- Year as a numeric feature (captures the 22%/5y growth trend)

### Direct multi-output forecasting
One model per hour-of-day. 24 models total. Each predicts only its own hour
slot given the same features. Often beats single-target on calendar-heavy
patterns.

## Chronos improvements

### Fine-tune on the train split
Zero-shot is the default. Fine-tuning Chronos on 2020-2023 ERCOT history
typically gains 5-15% MAPE. Requires GPU (~$10-15 on Modal/RunPod H100).

### Try Chronos-Bolt vs original Chronos
Bolt is 2024's faster variant — often comparable accuracy at much lower
latency.

## Scoreboard rigor

- Walk-forward CV instead of single train/val/test split
- Report per-horizon metrics (1h, 6h, 12h, 24h) not just point estimates
- Coverage analysis on prediction intervals (Day 7 already planned)
