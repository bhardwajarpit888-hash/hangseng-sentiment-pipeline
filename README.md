# Cross-Exchange Alternative Sentiment & n+1 Price Prediction Pipeline

A quantitative sentiment ingestion and machine learning pipeline that forecasts next-period asset returns and prices ($n+1$) across the **Hong Kong Stock Exchange (Hang Seng / HKEX)** and the **Korea Exchange (KOSPI / KRX)**.

The system ingests retail social sentiment from forum discussion boards via **Apify** and institutional financial news via **Exa.com**, aligns feeds to hourly UTC price bars, computes dynamic support/resistance zones, and evaluates predictive accuracy through walk-forward out-of-sample validation against naive persistence baselines.

---

## 1. System Architecture

```
hangseng_sentiment_pipeline/
├── config/
│   ├── settings.py              # Exchange definitions, tickers, Apify actor schemas & strict modes
│   └── __init__.py
├── data/
│   ├── market_data.py           # Hourly OHLCV market bar fetcher (UTC normalized, session filtered)
│   ├── scrapers/
│   │   ├── apify_client.py      # Apify scraping actor pipelines (LIHKG, Futu, Naver, Daum)
│   │   └── exa_news.py          # Exa.com neural search news ingestion
│   ├── processor.py             # UTC bar alignment, zero-lookahead feature fusion & lag engineering
│   └── cache/                   # Parquet cache storage for historical price/sentiment
├── sentiment/
│   ├── analyzer.py              # Multilingual domain lexicon (Cantonese/Korean/English) & position extractor
│   ├── engagement_weighting.py  # Follower reach, viral engagement & time-decay weighting engine
│   └── __init__.py
├── technical/
│   ├── indicators.py            # RSI, MACD, Bollinger Bands, ATR & volume dynamics
│   ├── levels.py                # Support/Resistance pivots, swing extrema & position decision engine
│   └── __init__.py
├── models/
│   ├── feature_engineering.py   # Matrix formatting & zero-leakage feature selection
│   ├── validation.py            # Walk-forward TimeSeriesSplit out-of-sample validation & baselines
│   ├── predictor.py             # Blended XGBoost + L2 Ridge Regression model
│   └── __init__.py
├── reporting/
│   ├── summary_table.py         # Rich terminal tables sorted by mentions & sentiment
│   ├── visualizer.py            # Interactive HTML dashboard with data source badges
│   ├── prediction_logger.py     # Parquet prediction logger and realization evaluator
│   └── __init__.py
├── tests/
│   └── test_pipeline.py         # Test suite (normalization, no-leakage, baseline, regression)
├── reports/
│   ├── sentiment_price_prediction_report.html  # Interactive visual report
│   ├── pipeline_summary.json                   # Output summary
│   └── predictions_log.parquet                 # Realized predictions log
├── app.py                       # Streamlit dashboard
├── main.py                      # Orchestrator CLI entrypoint
├── requirements.txt             # Production dependencies
└── .env.example                 # Credentials template
```

---

## 2. Quantitative Methodology & Formulations

### A. Data Ingestion & Cross-Exchange Coverage

Per exchange, the pipeline monitors 5 core benchmark equities across 2 independent localized social channels plus institutional news:

| Exchange | Tickers | Social Source 1 | Social Source 2 | News Source |
| :--- | :--- | :--- | :--- | :--- |
| **HKEX (Hang Seng)** | `0700.HK`, `9988.HK`, `3690.HK`, `1810.HK`, `0005.HK` | **LIHKG** (Financial Channel) | **Futu / Moomoo** (Stock Community) | **Exa.com** (Reuters, Bloomberg, SCMP, HKEJ) |
| **KRX (KOSPI)** | `005930.KS`, `000660.KS`, `035420.KS`, `035720.KS`, `005380.KS` | **Naver Finance** (종목토론실) | **Daum Finance** / Telegram Feeds | **Exa.com** (Bloomberg, Korea Times, KED Global) |

#### Apify Actor Configurations

| Social Channel | Apify Store Actor ID | Target URL Schema | Extraction Type |
| :--- | :--- | :--- | :--- |
| **LIHKG** | `apify/web-scraper` | `https://lihkg.com/category/5?page=1` | DOM cheerio / pageFunction |
| **Futu / Moomoo** | `apify/puppeteer-scraper` | `https://www.moomoo.com/community/stock/{code}` | Headless Chromium evaluation |
| **Naver Finance** | `apify/cheerio-scraper` | `https://finance.naver.com/item/board.naver?code={code}` | Table parser (종목토론실) |
| **Daum Finance** | `apify/cheerio-scraper` | `https://finance.daum.net/quotes/A{code}#talks` | Forum feed cheerio |
| **Telegram** | `vdrmmr/telegram-scraper` | `https://t.me/s/{channel}` | Public channel message extractor |

### B. Strict No-Leakage Time Alignment Rule

Each bar $k$ represents an interval $(t_{\text{close}, k-1}, t_{\text{close}, k}]$.
A post or article published at timestamp $t$ is mapped to bar $k$ if and only if:

$$t_{\text{close}, k-1} < t \le t_{\text{close}, k}$$

Features calculated for bar $k$ use information available strictly at or before $t_{\text{close}, k}$.
The target is the subsequent bar return:

$$R_{k+1} = \frac{\text{Close}_{k+1} - \text{Close}_k}{\text{Close}_k}$$

No future information enters the feature matrix of bar $k$. Warmup rows with missing indicator values are dropped rather than zero-filled.

### C. Engagement-Weighted Sentiment Index

Raw post counts are vulnerable to spam and noise. Each post $i$ is weighted by author authority (logarithmic follower scale), virality (likes, reposts, comments), and an exponential time decay ($\lambda = \ln(2) / t_{1/2}$, with $t_{1/2} = 3$ days):

$$W_i = \ln(1 + \text{Followers}_i) \cdot \left(1.0 + 0.25 \ln(1 + \text{Likes}_i) + 0.45 \ln(1 + \text{Reposts}_i) + 0.10 \ln(1 + \text{Views}_i)\right) \cdot e^{-\lambda \cdot \Delta t}$$

$$\bar{S}_{\text{weighted}} = \frac{\sum_{i=1}^{N} W_i \cdot S_i}{\sum_{i=1}^{N} W_i}$$

### D. Retail vs. Smart Money Divergence (RSMD)

Quantifies the spread between retail crowd sentiment and institutional reporting:

$$\Delta_{\text{RSMD}} = \bar{S}_{\text{weighted}} - \bar{S}_{\text{news}}$$

- **Overbought Distribution Trap ($\Delta_{\text{RSMD}} > +0.40$):** Retail euphoria while institutional news is neutral or negative, especially near Resistance $R_1$.
- **Capitulation Accumulation Zone ($\Delta_{\text{RSMD}} < -0.40$):** Retail panic while institutional fundamentals remain stable near Support $S_1$.

### E. Support & Resistance Levels

Computes classical floor pivots, Fibonacci pivots, and local swing extrema:

$$PP = \frac{\text{High} + \text{Low} + \text{Close}}{3}, \quad R_1 = 2 \cdot PP - \text{Low}, \quad S_1 = 2 \cdot PP - \text{High}$$

$$\text{Swing High} = \max_{t \in [T-10, T]} \text{High}_t, \quad \text{Swing Low} = \min_{t \in [T-10, T]} \text{Low}_t$$

The pipeline also parses specific price levels directly mentioned in forum text (e.g., target prices, stop bounds).

### F. Stated Retail Position vs. Suggested Stance

- **Stated Position**: Extracted directly from retail post text via regex matching phrases like `買入`, `long`, `물렸다`, `손절`, `매수`, `숏`.
- **Suggested Stance**: The model's risk-adjusted recommendation (`LONG`, `SHORT`, `NEUTRAL`) combining sentiment polarity, divergence, and technical pivot levels.

### G. Predictive Modeling & Walk-Forward Validation

- **Models**: Scikit-Learn `Pipeline(StandardScaler(), Ridge(alpha=1.0))` and `XGBRegressor`.
- **Validation**: Strict `TimeSeriesSplit(gap=1)` walk-forward cross validation. Requires minimum 40 training bars and 15 test bars per fold; otherwise reports `insufficient_data` and returns `None` for metrics.
- **Benchmarking**: Compares model accuracy against a naive persistence baseline ($\text{sign}(\hat{R}_{t+1}) = \text{sign}(R_t)$) and reports lift.
- **Statistical Tests**: Binomial test p-values ($H_0: p=0.50$) and 95% Wilson confidence intervals.
- **Sharpe Ratio**: Computed only when $n_{\text{test}} \ge 100$; otherwise suppressed as `None` to prevent spurious small-sample claims.

---

## 3. Results & Limitations

### Honest Assessment of Edge and Statistical Significance

1. **Sample Size Limitations**: In short lookback windows (e.g., 10 days of hourly bars), out-of-sample test samples range between 40 and 80 bars per asset. While directional accuracies of 50–55% are observed, high binomial p-values ($p > 0.05$) indicate that short-window directional lifts cannot be definitively distinguished from random chance without expanding to multi-month horizons.
2. **Persistence Baseline**: Financial asset returns exhibit negative autocorrelation at hourly frequencies during consolidation. Comparing against a naive persistence baseline reveals that raw hit rates near 50% often yield marginal net lift ($\pm 2\text{--}5\%$).
3. **Synthetic Mode Warning**: Simulation is **OFF by default**. If run with `--allow-simulation`, synthetic data is generated uniformly at random ($p = 0.33$ bull, bear, neut) without ticker bias, and all outputs are prominently labeled `⚠️ SIMULATED DATA WARNING`.

---

## 4. Setup & Execution

### Prerequisites

Python 3.10+ is required.

```bash
git clone https://github.com/bhardwajarpit888-hash/hangseng-sentiment-pipeline.git
cd hangseng-sentiment-pipeline
pip install -r requirements.txt
```

### Environment Configuration

Copy `.env.example` to `.env` and configure your API tokens:

```bash
cp .env.example .env
```

```ini
APIFY_API_TOKEN=your_apify_api_token
EXA_API_KEY=your_exa_api_key
LOOKBACK_DAYS=10
INTERVAL=1h
```

### Running the Pipeline (Live Mode)

```bash
python main.py --exchange ALL --lookback 10 --interval 1h
```

If API tokens are not provided, the pipeline halts with an informative error. To run offline with synthetic fallback feeds, pass:

```bash
python main.py --exchange ALL --lookback 10 --interval 1h --allow-simulation
```

### Realized Prediction Evaluation

Once subsequent market bars have printed, evaluate logged predictions against realized prices:

```bash
python main.py --evaluate
```

### Interactive Web Dashboard

```bash
streamlit run app.py
```

### Running the Test Suite

```bash
python -m unittest discover tests
```
*(All 6 unit tests covering normalization, zero-leakage, baseline comparison, and regression assertions pass).*
