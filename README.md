# Cross-Exchange Alternative Sentiment & n+1 Price Prediction Pipeline

A quantitative sentiment ingestion and machine learning pipeline that predicts next-period asset returns and prices ($n+1$) across the **Hong Kong Stock Exchange (Hang Seng / HKEX)** and the **Korea Exchange (KOSPI / KRX)**.

The system fuses high-frequency retail social sentiment scraped via **Apify** with institutional financial news ingested via **Exa.com**, aligns feeds to UTC price bars, computes dynamic support/resistance zones, and evaluates predictive accuracy through walk-forward out-of-sample validation.

---

## 1. System Architecture

```
hangseng_sentiment_pipeline/
├── config/
│   ├── __init__.py
│   └── settings.py              # Exchange definitions, tickers, API configs & fallback rules
├── data/
│   ├── market_data.py           # OHLCV price bar fetcher (yfinance, UTC normalized)
│   ├── scrapers/
│   │   ├── apify_client.py      # Apify scraping actors (LIHKG, Futu/Moomoo, Naver, Daum)
│   │   └── exa_news.py          # Exa.com neural search news ingestion
│   ├── processor.py             # UTC bar alignment, feature fusion & lag engineering
│   └── cache/                   # Parquet cache storage for price and sentiment feeds
├── sentiment/
│   ├── __init__.py
│   ├── analyzer.py              # Multilingual financial domain lexicon (Cantonese/Korean/English)
│   └── engagement_weighting.py  # Reach, virality & time-decay weighting engine
├── technical/
│   ├── __init__.py
│   ├── indicators.py            # RSI, MACD, Bollinger Bands, ATR & volume dynamics
│   └── levels.py                # Support/Resistance pivots & position decision engine
├── models/
│   ├── __init__.py
│   ├── feature_engineering.py   # Feature selection and matrix formatting
│   ├── validation.py            # Walk-forward TimeSeriesSplit out-of-sample validation
│   └── predictor.py             # Blended XGBoost + L2 Ridge Regression model
├── reporting/
│   ├── __init__.py
│   ├── summary_table.py         # Rich terminal tables sorted by mentions & sentiment
│   └── visualizer.py            # Self-contained interactive HTML dashboard
├── tests/
│   └── test_pipeline.py         # Test suite covering scrapers, NLP, levels & models
├── reports/
│   ├── sentiment_price_prediction_report.html  # Interactive visual report
│   └── pipeline_summary.json                   # Machine-readable output summary
├── app.py                       # Interactive Streamlit dashboard
├── main.py                      # Orchestrator CLI entrypoint
├── requirements.txt             # Project dependencies
└── .env.example                 # Environment credentials template
```

---

## 2. Quantitative Methodology & Formulations

### A. Data Ingestion & Cross-Exchange Coverage

Per exchange, the pipeline ingests 5 core liquid benchmark assets across minimum 2 independent social channels plus institutional news:

| Exchange | Tickers Ingested | Social Channel 1 | Social Channel 2 | Institutional News |
| :--- | :--- | :--- | :--- | :--- |
| **HKEX (Hang Seng)** | `0700.HK`, `9988.HK`, `3690.HK`, `1810.HK`, `0005.HK` | **LIHKG** (Financial Channel) | **Futu / Moomoo** (Stock Community) | **Exa.com** (Reuters, Bloomberg, SCMP, HKEJ) |
| **KRX (KOSPI)** | `005930.KS`, `000660.KS`, `035420.KS`, `035720.KS`, `005380.KS` | **Naver Finance** (종목토론실) | **Daum Finance** / Telegram Feeds | **Exa.com** (Bloomberg, Korea Times, KED Global) |

### B. Engagement-Weighted Sentiment Formulation

Raw social mentions are heavily susceptible to noise, retail spam, and bot activity. The pipeline computes an **Engagement-Weighted Sentiment Score** where each post $i$ is weighted by author authority (logarithmic follower reach), viral spread (likes, reposts, comments), and an exponential half-life time decay:

$$W_i = \ln(1 + \text{Followers}_i) \cdot \left(1.0 + 0.25 \ln(1 + \text{Likes}_i) + 0.45 \ln(1 + \text{Reposts}_i) + 0.10 \ln(1 + \text{Views}_i)\right) \cdot e^{-\lambda \cdot \Delta t}$$

$$\bar{S}_{\text{weighted}} = \frac{\sum_{i=1}^{N} W_i \cdot S_i}{\sum_{i=1}^{N} W_i}$$

Where:
- $\Delta t$: Age of the post in seconds relative to bar close.
- $\lambda = \frac{\ln(2)}{t_{1/2}}$: Half-life decay parameter ($t_{1/2} = 3$ days).

### C. Retail vs. Smart Money Divergence (RSMD)

To avoid buying into exhaustion tops or selling into capitulation lows, the pipeline computes the sentiment delta between retail crowds (Apify social channels) and institutional coverage (Exa news):

$$\Delta_{\text{RSMD}} = \bar{S}_{\text{weighted}} - \bar{S}_{\text{news}}$$

- **Overbought Distribution Trap ($\Delta_{\text{RSMD}} > +0.40$):** Extreme retail euphoria while institutional news is neutral or negative, especially near Resistance $R_1$.
- **Capitulation Accumulation Zone ($\Delta_{\text{RSMD}} < -0.40$):** Retail panic while institutional fundamentals remain solid, particularly around Support $S_1$.

### D. Support & Resistance Formulations

The technical engine calculates classical Floor Pivots and Fibonacci extensions over the lookback window:

$$PP = \frac{\text{High} + \text{Low} + \text{Close}}{3}$$

$$R_1 = 2 \cdot PP - \text{Low}, \quad S_1 = 2 \cdot PP - \text{High}$$

$$R_2 = PP + (\text{High} - \text{Low}), \quad S_2 = PP - (\text{High} - \text{Low})$$

$$R_{\text{Fib}} = PP + 0.618 \cdot (\text{High} - \text{Low}), \quad S_{\text{Fib}} = PP - 0.618 \cdot (\text{High} - \text{Low})$$

### E. Predictive Modeling at $n+1$

The target variable is the next period return $R_{t+1}$:

$$R_{t+1} = \frac{P_{t+1} - P_t}{P_t}$$

The model blends **Gradient Boosted Trees (XGBoost)** and **L2 Regularized Ridge Regression**:

$$\hat{R}_{t+1} = 0.65 \cdot \hat{R}_{\text{XGB}} + 0.35 \cdot \hat{R}_{\text{Ridge}}$$

$$\hat{P}_{n+1} = P_n \cdot (1 + \hat{R}_{t+1})$$

Prediction intervals are constructed using the normalized 14-period Average True Range ($ATR_{14}$):

$$\text{Bounds}_{n+1} = \hat{P}_{n+1} \pm (P_n \cdot ATR_{\text{norm}} \cdot 0.8)$$

---

## 3. Out-of-Sample Walk-Forward Validation

To ensure realistic estimates and prevent lookahead bias:
1. Historical samples are split chronologically using `TimeSeriesSplit(n_splits=3)`.
2. All feature transformations and lag attributes ($X_{t-1}$) are calculated strictly up to the training timestamp.
3. Evaluated metrics include:
   - **Directional Accuracy / Hit Ratio**: $\frac{1}{M} \sum \mathbb{I}(\text{sign}(\hat{R}_{t+1}) == \text{sign}(R_{t+1}))$
   - **Root Mean Squared Error (RMSE)**
   - **Mean Absolute Error (MAE)**
   - **Walk-Forward Strategy Sharpe Ratio & Cumulative Return**

---

## 4. Setup & Execution

### Prerequisites

Ensure Python 3.10+ is installed.

```bash
git clone <repo_url>
cd hangseng_sentiment_pipeline
pip install -r requirements.txt
```

### Environment Configuration

Copy `.env.example` to `.env` and insert your credentials:

```bash
cp .env.example .env
```

```ini
APIFY_API_TOKEN=your_apify_api_token
EXA_API_KEY=your_exa_api_key
LOOKBACK_DAYS=10
```

*(Note: The system contains high-fidelity fallback generators for testing offline or when API tokens are not provided, allowing immediate end-to-end evaluation out of the box).*

### Running the Pipeline

To run the complete pipeline across both exchanges:

```bash
python main.py --exchange ALL --lookback 10
```

To run specifically for Hang Seng / HKEX:

```bash
python main.py --exchange HKEX --lookback 10
```

### Launching the Interactive Web Dashboard

To launch the real-time visual dashboard:

```bash
streamlit run app.py
```

### Viewing the Static HTML Report

The pipeline automatically outputs an interactive HTML report to:
`reports/sentiment_price_prediction_report.html`

Open it directly in any modern browser to view the visual charts and rankings.

### Running Unit Tests

```bash
python -m unittest discover tests
```

---

## 5. Sample Output & Asset Ranking

```
=== HKEX Market Intelligence & Sentiment Summary (Sorted by Mentions) ===
┌──────────┬──────────────────────┬─────────┬──────────┬───────────┬──────────┬──────────┬────────────┬────────────┬─────────────┬─────────┐
│ Ticker   │ Asset Name           │ Price   │ Mentions │ Sentiment │ Direct.  │ Position │ S1 (Supp)  │ R1 (Res)   │ Target n+1  │ OOS Acc │
├──────────┼──────────────────────┼─────────┼──────────┼───────────┼──────────┼──────────┼────────────┼────────────┼─────────────┼─────────┤
│ 1810.HK  │ Xiaomi Corporation   │ 29.70   │ 80       │ +0.33     │ BULLISH  │ LONG     │ 27.63      │ 31.75      │ 30.33       │ 66.7%   │
│ 0005.HK  │ HSBC Holdings plc    │ 138.50  │ 80       │ +0.31     │ BULLISH  │ LONG     │ 132.90     │ 143.70     │ 137.69      │ 46.7%   │
│ 0700.HK  │ Tencent Holdings Ltd │ 376.00  │ 80       │ +0.25     │ BULLISH  │ LONG     │ 357.60     │ 412.80     │ 374.58      │ 53.3%   │
│ 3690.HK  │ Meituan              │ 70.50   │ 80       │ +0.25     │ BULLISH  │ LONG     │ 65.50      │ 79.25      │ 70.85       │ 46.7%   │
│ 9988.HK  │ Alibaba Group        │ 85.45   │ 80       │ +0.10     │ NEUTRAL  │ NEUTRAL  │ 77.23      │ 101.53     │ 84.04       │ 46.7%   │
└──────────┴──────────────────────┴─────────┴──────────┴───────────┴──────────┴──────────┴────────────┴────────────┴─────────────┴─────────┘
```
