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

## E3: Training loss and model refinements (2 Nov 2026)

**Question:** Can the training loss, a measurement-error feature, or a model ensemble reduce forecast error?

**Setup:** same data, final feature set from E2, same split. QLIKE added as a metric: $y^2/\hat{y}^2 - \ln(y^2/\hat{y}^2) - 1$ on variance, the standard volatility loss, robust to noise in the realized-volatility target (Patton 2011). Variants:
- Current: LightGBM on log-volatility (squared error)
- RMSPE-weighted: LightGBM on volatility, rows weighted by $1/y^2$ (targets floored at their 1st percentile)
- QLIKE-trained: LightGBM gamma objective on variance, equivalent to minimizing QLIKE
- Realized quarticity: current model plus $RQ = (60/3)\sum r^4$ over the last hour (HARQ idea, Bollerslev, Patton and Quaedvlieg 2016)
- Ensemble: weighted average of HAR and LightGBM, weight chosen on a validation week (last 7 days of training data)

**Results (test set):**

| Variant | RMSPE | QLIKE | Correlation |
|---|---|---|---|
| Current (log target) | 0.2702 | 0.2109 | 0.7828 |
| RMSPE-weighted | 0.2444 | 0.2637 | 0.7960 |
| QLIKE-trained | 0.3139 | 0.1969 | 0.7756 |
| + realized quarticity | 0.2698 | 0.2121 | 0.7821 |
| LightGBM alone, validation-reduced training | 0.2749 | 0.2153 | 0.7813 |
| Ensemble (0.9 LightGBM, 0.1 HAR) | 0.2731 | 0.2144 | 0.7819 |

**Conclusion:** Each training loss won its own metric by a wide margin. RMSPE training under-forecasts by design: its optimal forecast is $E[1/y]/E[1/y^2]$, which lies below average volatility, because over-forecasting a calm hour produces a large percentage error. QLIKE training is unbiased for variance (optimal $\hat{y}^2 = E[y^2]$). Since the application warns users about expected price swings, systematic under-forecasting is the most harmful failure, so QLIKE was adopted as the primary metric and LightGBM switched to QLIKE training. RMSPE and correlation are still reported. Realized quarticity had no effect, and the ensemble gained under 1% with a weight of 0.9 on LightGBM, so neither was adopted.

## E4: Drift metric choice (3-4 Nov 2026)

**Question:** Which input-drift measure is reliable for rolling-window features?

**Tried:** PSI on 10 training deciles, live window of about 90 minutes. Every coin and feature scored 1.3 to 8.3, with `rv_24h` at 8.28, the maximum PSI for 10 buckets (all live rows in one bucket).

**Why it failed:** the features are rolling windows, so consecutive minutes are nearly identical (strong autocorrelation). PSI and similar two-sample tests assume independent rows, so a short live window looks like a single point and is always flagged.

**Adopted:** per feature, the training percentile of the latest value, and the share of the last 24 hours outside the training 1st to 99th percentile (2% expected). On the same day, BTC volatility features sat at the 65th to 92nd percentile with out-of-range shares near 0, i.e. no drift.

**Decision:** percentile and out-of-range share for inputs; sustained live RMSPE against test RMSPE for performance.
