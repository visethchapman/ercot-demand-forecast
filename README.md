# ERCOT Demand Forecasting

A comparison of three forecasting methods on Texas grid (ERCOT) hourly
electricity demand:

1. **Naive baseline** — same-hour-last-week
2. **LightGBM** — gradient boosting with engineered features (lags,
   calendar, lagged Houston weather)
3. **Chronos** — Amazon's zero-shot time-series foundation model
   (HuggingFace)

Companion to [`energy-text2sql`](https://github.com/visethchapman/energy-text2sql) —
same dataset, different question. That repo *explores* the data
agentically. This one *predicts* from it.

## Status

Week 1, Day 1 — repo bootstrap + EDA.

## Stack

| Layer | Tech |
|---|---|
| Database | Postgres 16 (reused from `energy-text2sql`) |
| Modeling | LightGBM, Chronos via HuggingFace transformers |
| Runtime | Python 3.12 + `uv` |
| Notebooks | Jupyter |

## Data

| Source | Rows | Range |
|---|---|---|
| `eia.demand` (ERCO only) | 43,847 hours | 2020-01-01 → 2024-12-31 |
| `noaa.daily_weather` (Houston) | ~2,325 days | 2020-01-01 → 2026-05-13 |

## Roadmap

- Day 1: EDA — seasonality, demand vs temperature
- Day 2: Naive baselines + walk-forward backtest protocol
- Day 3–4: LightGBM with engineered features (quantile regression for uncertainty)
- Day 5–6: Chronos zero-shot
- Day 7: Calibration analysis (does 90% PI actually cover 90% of points?)
- Day 8: Comparison + scoreboard
- Day 9: README polish + demo plot

## Quick start

```bash
# 1. Make sure the energy-text2sql Postgres is running
cd ../energy-text2sql
docker compose up -d
cd -

# 2. Configure
cp .env.example .env

# 3. Install
uv sync

# 4. Open the EDA notebook
uv run jupyter lab notebooks/01_eda.ipynb
```

## Repository layout

```
.
├── notebooks/                # Narrative — one per phase
│   └── 01_eda.ipynb
├── forecasting/              # Reusable modules (testable, populated Day 2+)
│   ├── data.py               # load_demand(), load_weather()
│   ├── backtest.py           # walk-forward split — source of truth on leakage
│   ├── features.py           # lag, calendar, weather features
│   ├── metrics.py            # MAPE, RMSE, MAE, coverage
│   └── models.py             # method wrappers
├── results/
│   └── scoreboard.json       # final comparison numbers
├── pyproject.toml
└── README.md
```

## License

MIT.
