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

## E2: Additional candle-based features (2 Nov 2026)

**Question:** Do additional features from 1-minute candles improve LightGBM beyond the base set of 10?

**Setup:** same data, model and split as E1. Base: rv_15m, rv_1h, rv_4h, rv_24h, range_1h, volume_ratio, trades_ratio, taker_buy_ratio, hour, weekday. Each group added to the base on its own:
- Same hour yesterday: realized volatility of the forecast window's time slot 24 hours earlier
- Downside: realized volatility from negative returns only (last hour)
- Jumps: share of last hour's variance from jumps (realized variance minus bipower variation)
- Ratios: rv_15m / rv_24h and rv_1h / rv_24h
- Market: equal-weighted market volatility (1h, 4h), BTC volatility, coin-vs-market volatility, 4h correlation with BTC

**Results (test set):**

| Features | RMSPE | RMSE | Correlation |
|---|---|---|---|
| Base | 0.2736 | 0.001533 | 0.7807 |
| + same hour yesterday | 0.2714 | 0.001529 | 0.7825 |
| + downside | 0.2731 | 0.001528 | 0.7828 |
| + jumps | 0.2738 | 0.001526 | 0.7824 |
| + ratios | 0.2719 | 0.001518 | 0.7856 |
| + market | 0.2787 | 0.001528 | 0.7820 |
| All groups | 0.2740 | 0.001538 | 0.7790 |
| + same hour yesterday + ratios | **0.2702** | **0.001528** | **0.7828** |

**Conclusion:** Same-hour-yesterday and the volatility ratios each improved every metric; combined they lower RMSPE by 1.2%, so both were kept. Downside and jump features had no effect. Market features worsened RMSPE by 1.9%: each coin's own volatility already reflects market-wide moves, and features shared identically across coins likely let the model fit training-period market conditions that did not carry over. Adding every group at once performed no better than the base, showing that unhelpful features can cancel out helpful ones. Final feature set: base plus rv_same_hour_yday, vol_ratio_15m_24h, vol_ratio_1h_24h.
