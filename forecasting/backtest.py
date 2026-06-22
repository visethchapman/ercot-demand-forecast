"""Temporal train / validation / test split — the leakage-prevention layer.

Rule: every train timestamp < every val timestamp < every test timestamp.
Enforced by assertions at the end of split_train_val_test(). If those
assertions fail, the split is wrong. Don't disable them.

Default boundaries for the 2020-2024 ERCOT dataset:
    train: 2020-01-01 → 2023-06-30   (3.5 years)
    val:   2023-07-01 → 2023-12-31   (6 months)
    test:  2024-01-01 → 2024-12-31   (1 year)
"""

from __future__ import annotations

import pandas as pd


TRAIN_END = "2023-07-01"
VAL_END   = "2024-01-01"
TEST_END  = "2025-01-01"


def split_train_val_test(
    df: pd.DataFrame,
    ts_col: str = "ts",
    train_end: str = TRAIN_END,
    val_end:   str = VAL_END,
    test_end:  str = TEST_END,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train, val, test) DataFrames partitioned by `ts_col`.

    Asserts strict temporal ordering between the three splits.
    """
    ts = pd.to_datetime(df[ts_col], utc=True)
    train_end_ts = pd.Timestamp(train_end, tz="UTC")
    val_end_ts   = pd.Timestamp(val_end,   tz="UTC")
    test_end_ts  = pd.Timestamp(test_end,  tz="UTC")

    train = df.loc[ts < train_end_ts].copy()
    val   = df.loc[(ts >= train_end_ts) & (ts < val_end_ts)].copy()
    test  = df.loc[(ts >= val_end_ts)   & (ts < test_end_ts)].copy()

    # Leakage assertions — DO NOT disable these to make a model "work."
    if len(train) and len(val):
        assert pd.to_datetime(train[ts_col], utc=True).max() \
             < pd.to_datetime(val[ts_col],   utc=True).min(), \
            "train/val temporal overlap — split is leaking!"
    if len(val) and len(test):
        assert pd.to_datetime(val[ts_col],  utc=True).max() \
             < pd.to_datetime(test[ts_col], utc=True).min(), \
            "val/test temporal overlap — split is leaking!"

    return train, val, test
