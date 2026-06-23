"""
Feature engineering for the 24-hour-ahead ERCOT demand forecast.

For each row at timestamp T (target time), features describe information
available at the prediction time T - horizon_hours (e.g. 24h before T).

Leakage rules:
- Lag features shift demand by AT LEAST `horizon_hours` (never less).
- Rolling stats are computed on the lagged series, so they end exactly at
  T - horizon_hours.
- Weather uses observations from `date(T) - 1 day` (local) — yesterday's
  actuals as a proxy for "today's weather forecast" at prediction time.
- Calendar features describe the TARGET time T (because the model knows
  what hour/day/month it's predicting for).
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

import holidays


TARGET = "demand_mwh"

# Stable list of feature columns — keep in sync with build_features() output.
FEATURE_COLS: list[str] = [
    # Calendar (about the target time)
    "hour", "dow", "month", "is_weekend", "is_holiday",
    # Cyclical encoding (sin/cos pairs)
    "hour_sin", "hour_cos",
    "dow_sin",  "dow_cos",
    "month_sin", "month_cos",
    # Lags (timestamps relative to TARGET t)
    "lag_24h", "lag_25h", "lag_48h", "lag_72h", "lag_168h", "lag_192h",
    # Rolling stats ending at T-h (prediction time)
    "roll_24h_mean", "roll_24h_std", "roll_168h_mean",
    # Weather lagged 1 day vs target's local date
    "tmax_lag1", "tmin_lag1", "prcp_lag1", "awnd_lag1",
]


def build_features(
    demand_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    horizon_hours: int = 24,
    local_tz: str = "America/Chicago",
) -> pd.DataFrame:
    """Return a DataFrame with target + features for each demand row.

    demand_df must have columns: ts (TIMESTAMPTZ in UTC) and demand_mwh.
    weather_df must have columns: obs_date (python date), tmax_c, tmin_c,
    prcp_mm, awnd_ms.
    """
    h = horizon_hours
    df = demand_df.copy().sort_values("ts").reset_index(drop=True)

    # Target time, in local
    df["ts_local"] = df["ts"].dt.tz_convert(local_tz)

    # ---------- calendar (target time) ----------
    df["hour"] = df["ts_local"].dt.hour
    df["dow"] = df["ts_local"].dt.dayofweek
    df["month"] = df["ts_local"].dt.month
    df["is_weekend"] = (df["dow"] >= 5).astype(int)

    us_holidays = holidays.UnitedStates()
    df["is_holiday"] = df["ts_local"].dt.date.map(lambda d: int(d in us_holidays))

    # cyclical encoding
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dow"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dow"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

    # ---------- lag features (named relative to TARGET time T) ----------
    # All lags must be >= horizon_hours to avoid leakage at prediction time T-h.
    y = df["demand_mwh"]
    df["lag_24h"]  = y.shift(24)            # same hour yesterday = prediction time itself
    df["lag_25h"]  = y.shift(25)
    df["lag_48h"]  = y.shift(48)            # same hour 2 days ago
    df["lag_72h"]  = y.shift(72)            # same hour 3 days ago
    df["lag_168h"] = y.shift(168)           # SAME HOUR LAST WEEK — strong weekly signal
    df["lag_192h"] = y.shift(192)           # same hour, day before last week

    # rolling stats ending at T-h
    shifted = y.shift(h)
    df["roll_24h_mean"]  = shifted.rolling(24).mean()
    df["roll_24h_std"]   = shifted.rolling(24).std()
    df["roll_168h_mean"] = shifted.rolling(168).mean()

    # ---------- weather (yesterday's local-date observation) ----------
    # target's local date minus 1 day
    df["weather_date"] = df["ts_local"].dt.date.apply(
        lambda d: d - dt.timedelta(days=1)
    )

    w = weather_df[["obs_date", "tmax_c", "tmin_c", "prcp_mm", "awnd_ms"]].rename(
        columns={
            "obs_date":  "weather_date",
            "tmax_c":    "tmax_lag1",
            "tmin_c":    "tmin_lag1",
            "prcp_mm":   "prcp_lag1",
            "awnd_ms":   "awnd_lag1",
        }
    )
    df = df.merge(w, on="weather_date", how="left")

    df = df.drop(columns=["ts_local", "weather_date"])
    return df
