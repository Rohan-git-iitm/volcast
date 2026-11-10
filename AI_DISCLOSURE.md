# AI disclosure

This statement describes how AI tools were used in building VolCast, as required by the course policy.

## Tools used

Claude.

## How it was used

**1. Project scoping and proposal.** I discussed the rubric with Claude and reviewed several candidate project ideas it proposed. I chose volatility forecasting, decided to use Binance's public tick-level trade data, and validated the approach myself in a Kaggle notebook comparing persistence, HAR and LightGBM before committing to it. Claude helped draft the proposal; I edited it and removed numeric targets it had suggested (latency and error thresholds) because they were not yet supported by any measurement. A previous year's student project on GitHub was read for reference only; no code was taken from it.

**2. Explanations of concepts and tools.** I used Claude to explain parts of the stack as I built them, for example the MLflow registry and aliases, model serving, FastAPI and uvicorn, DVC's stage and caching behavior, Prometheus metric types and PromQL, and why PSI fails on autocorrelated features.

**3. Code generation.** Most of the code was drafted by Claude, one pipeline stage or component at a time, to my specifications: the DVC stages, feature engineering, training and evaluation, the backend and worker, monitoring configuration, the frontend, the scripts and the tests. For every piece, I asked for an explanation of what each file and function does, ran it myself in my Ubuntu VM, checked its output, and asked for changes where it did not behave as intended or did not fit the design.

**4. Debugging.** I used AI to diagnose problems, for example a Docker Compose image-name clash, a NaN that broke JSON responses (found by a test), missed scheduled runs while the machine slept, a full disk that interrupted the data refresh, and a drift reference that no longer matched the served model.

**5. Research.** I used Claude to research methods before choosing them. For model selection: forecasting baselines (persistence, and the HAR model, adapted here to 1-hour, 4-hour and 24-hour windows), loss functions for comparing volatility forecasts (QLIKE against squared-error losses), and gradient-boosted trees trained with a gamma objective, which matches QLIKE. For feature engineering: realized volatility over several windows, intraday and weekly seasonality, activity measures such as volume and trade counts, and large-trade features, which I then tested and dropped (experiment E1). For monitoring: on my request it searched the web to survey drift-detection methods (PSI, KS, Jensen-Shannon, Wasserstein, PCA reconstruction, domain classifiers, range checks, Page-Hinkley, ADWIN) with sources, which I used to choose the drift metric. In each case I made the final choice after testing the options on the data.

**6. Documentation.** Claude drafted the documents in `docs/` and the README from the work done and the decisions recorded during the project. I reviewed them, had them simplified where needed, and checked technical details against the code.

## My decisions and contributions

Design choices that shaped the system came from me or were made by me, such as:

- using raw Binance trades and keeping Spark for the bars stage
- dropping the large-trade features once the ablation showed no benefit (experiment E1)
- fetching only new candles each minute instead of the full window
- a rolling 92-day window refreshed daily, with retraining only on alert, so data preparation and training run at different times
- deleting old data only after new downloads are verified
- monitoring live QLIKE instead of RMSPE, for consistency with the metric the champion is selected on
- researching drift methods before replacing PSI, rather than accepting the first alternative
- simplifying the dashboard and frontend for readability

I ran every command and test, verified results against independent sources where possible (Spark bars against Binance's official candles, models against baselines, live forecasts against actual outcomes), and can explain each part of the code and the reasoning behind it.

## Impact

AI assistance sped up implementation, testing and documentation. It also introduced mistakes, which testing and review caught: an incorrect diagnosis of missing data, a few design flaws found when tracing failure cases, and details in the documents that did not match the code. The final system, its correctness and its design are my responsibility.
