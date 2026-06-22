"""Data loaders for ERCOT hourly demand + station daily weather.

Reads from the Postgres container shared with the `energy-text2sql` project.
Loads `.env` from the repo root, where `DATABASE_URL` should be defined.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import psycopg
from dotenv import load_dotenv

_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=True)
DB_URL = os.environ["DATABASE_URL"]


def load_demand(region: str = "ERCO") -> pd.DataFrame:
    """Return hourly demand for one EIA balancing-authority region.

    Columns:
        ts (datetime64[ns, UTC])  — hour timestamp
        demand_mwh (float)        — demand in megawatt-hours
    """
    q = """
        SELECT period AS ts, value AS demand_mwh
        FROM eia.demand
        WHERE region = %s
        ORDER BY period
    """
    with psycopg.connect(DB_URL) as conn:
        df = pd.read_sql(q, conn, params=(region,))
    df["ts"] = pd.to_datetime(df["ts"], utc=True)
    return df


def load_weather(station_id: str = "USW00012960") -> pd.DataFrame:
    """Return daily weather for one NOAA station. Default = Houston Hobby."""
    q = """
        SELECT obs_date, tmax_c, tmin_c, prcp_mm, awnd_ms
        FROM noaa.daily_weather
        WHERE station_id = %s
        ORDER BY obs_date
    """
    with psycopg.connect(DB_URL) as conn:
        df = pd.read_sql(q, conn, params=(station_id,))
    df["obs_date"] = pd.to_datetime(df["obs_date"]).dt.date
    return df
