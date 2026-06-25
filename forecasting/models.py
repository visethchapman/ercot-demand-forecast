"""
Shared LightGBM training helper.

Single source of truth for hyperparameters so notebooks 03 and 05 don't drift.
"""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd


# Default hyperparameters — used by both point and quantile regression.
# Changes here propagate to every notebook that imports train_lgbm().
DEFAULT_PARAMS: dict = dict(
    n_estimators=1500,
    learning_rate=0.05,
    num_leaves=63,
    min_data_in_leaf=50,
    feature_fraction=0.9,
    bagging_fraction=0.9,
    bagging_freq=5,
    n_jobs=1,            # avoids OpenMP conflicts with PyTorch on macOS
    verbose=-1,
)

DEFAULT_EARLY_STOPPING_ROUNDS = 40


def train_lgbm(
    X_train,
    y_train,
    X_val,
    y_val,
    *,
    quantile: float | None = None,
    early_stopping_rounds: int = DEFAULT_EARLY_STOPPING_ROUNDS,
    **overrides,
) -> lgb.LGBMRegressor:
    """
    Train one LightGBM regressor with the project's default hyperparameters.

    If `quantile` is provided (e.g., 0.1), uses quantile regression at that
    alpha; otherwise fits the default mean (MSE) objective.

    Any keyword in `overrides` replaces the corresponding DEFAULT_PARAMS entry.
    """
    params = dict(DEFAULT_PARAMS)
    params.update(overrides)
    if quantile is not None:
        params["objective"] = "quantile"
        params["alpha"] = quantile

    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
    )
    return model
