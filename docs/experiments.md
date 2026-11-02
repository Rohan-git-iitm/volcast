# Experiment log

## E1: Large-trade features ablation (2 Nov 2026)

**Question:** Do trade-level large-order features, which only raw trades can provide, improve next-hour volatility forecasts?

**Setup:**
- Data: BTCUSDT, ETHUSDT, SOLUSDT, 1 Jul to 30 Sep 2026, 1-minute bars built with Spark
- Large trade: quote size at or above the coin's 99th percentile, from the first 7 days only
- Model: LightGBM (500 trees, learning rate 0.03, 31 leaves, min 200 rows per leaf) on log-volatility, one model for all coins
- Split: last 14 days as test, 60-minute gap before it
- Variants:
  - No large-trade: volatility, activity and time features only
  - Absolute: plus large_share_15m, large_trades_15m, max_trade_15m
  - Surge: plus the same three, each divided by its own 24-hour average per coin

**Results (test set):**

| Variant | RMSPE | RMSE | Correlation |
|---|---|---|---|
| No large-trade | 0.2714 | 0.001532 | 0.7819 |
| Absolute | 0.2725 | 0.001521 | 0.7839 |
| Surge | 0.2711 | 0.001528 | 0.7836 |

Feature importance (gain) in the absolute variant also ranked all three large-trade features last.

**Conclusion:** All variants are within 0.4% on every metric, which is within noise for a single split. Large-trade information does not improve hourly forecasts beyond recent volatility, activity and time-of-day features. The features were removed from the model, and the planned live trade collector was dropped, so serving uses only 1-minute candles from the REST API. The bars stage still computes large-trade columns so this ablation can be rerun as a tracked DVC experiment.

**Baselines for reference (same split, before removal):** persistence RMSPE 0.347, HAR 0.303, LightGBM (absolute variant) 0.275.
