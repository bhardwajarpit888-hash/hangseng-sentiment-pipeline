import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config.settings import EXCHANGE_CONFIGS
from main import PipelineOrchestrator

st.set_page_config(
    page_title="Hang Seng & KRX Sentiment Price Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        border-radius: 8px;
        padding: 16px;
        border: 1px solid #334155;
    }
    .badge-bull { color: #10b981; font-weight: bold; }
    .badge-bear { color: #ef4444; font-weight: bold; }
    .badge-neut { color: #f59e0b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=600)
def load_pipeline_data(lookback_days: int):
    orchestrator = PipelineOrchestrator(lookback_days=lookback_days)
    results = orchestrator.run_pipeline(target_exchange="ALL")
    return results


# Sidebar controls
st.sidebar.title("Market Sentiment Pipeline")
st.sidebar.caption("Alternative Data Ingestion & n+1 Forecast")

lookback = st.sidebar.slider("Lookback Horizon (Days)", min_value=5, max_value=30, value=10)
selected_exchange = st.sidebar.selectbox("Exchange", ["ALL", "HKEX", "KRX"])

if st.sidebar.button("Run / Refresh Pipeline", use_container_width=True):
    st.cache_data.clear()

with st.spinner("Ingesting social posts, Exa news, and training predictive models..."):
    pipeline_data = load_pipeline_data(lookback)

summary_df = pipeline_data["summary_df"]
results = pipeline_data["results"]

if selected_exchange != "ALL":
    summary_df = summary_df[summary_df["exchange"] == selected_exchange]

st.title("Cross-Exchange Sentiment & n+1 Price Prediction")
st.caption(f"Hong Kong Stock Exchange (Hang Seng) & Korea Exchange (KOSPI) | {lookback}-Day Ingestion Window")

# KPI Summary Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
total_mentions = summary_df["social_mentions"].sum()
avg_accuracy = summary_df["oos_accuracy"].mean() * 100
avg_sharpe = summary_df["sharpe"].mean()
bull_count = (summary_df["direction"] == "BULLISH").sum()

kpi1.metric("Total Social Posts", f"{total_mentions:,}")
kpi2.metric("OOS Directional Accuracy", f"{avg_accuracy:.1f}%")
kpi3.metric("Walk-Forward Sharpe", f"{avg_sharpe:.2f}")
kpi4.metric("Bullish Assets", f"{bull_count} / {len(summary_df)}")

# Main Tabs
tab_ranked, tab_deep_dive, tab_technical, tab_architecture = st.tabs([
    "Ranked Market Intelligence",
    "Asset Deep Dive",
    "Support & Resistance Levels",
    "Model Architecture"
])

with tab_ranked:
    st.subheader("Asset Rankings (Sorted by Social Discussion & Sentiment)")

    sort_col = st.selectbox("Sort By", ["social_mentions", "composite_sentiment", "current_price", "oos_accuracy"], index=0)
    display_df = summary_df.sort_values(by=sort_col, ascending=False)[[
        "symbol", "name", "exchange", "current_price", "currency",
        "social_mentions", "weighted_sentiment", "news_sentiment",
        "composite_sentiment", "direction", "position",
        "support_s1", "resistance_r1", "pred_price_n1", "oos_accuracy"
    ]].copy()

    display_df["oos_accuracy"] = (display_df["oos_accuracy"] * 100).round(1).astype(str) + "%"
    display_df["composite_sentiment"] = display_df["composite_sentiment"].round(3)
    display_df["weighted_sentiment"] = display_df["weighted_sentiment"].round(3)
    display_df["news_sentiment"] = display_df["news_sentiment"].round(3)

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    # Plotly Volume & Sentiment Chart
    fig_bar = go.Figure()
    fig_bar.add_trace(go.Bar(
        x=summary_df["symbol"],
        y=summary_df["social_mentions"],
        name="Social Mentions",
        marker_color="#3b82f6"
    ))
    fig_bar.add_trace(go.Scatter(
        x=summary_df["symbol"],
        y=summary_df["composite_sentiment"] * (summary_df["social_mentions"].max() * 0.8),
        name="Scaled Composite Sentiment",
        mode="lines+markers",
        line=dict(color="#10b981", width=3)
    ))
    fig_bar.update_layout(
        title="Social Discussion Volume vs Composite Sentiment Polarity",
        template="plotly_dark",
        height=380,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with tab_deep_dive:
    selected_symbol = st.selectbox("Select Asset to Inspect", summary_df["symbol"].tolist())
    asset_info = next((item for item in results if item["symbol"] == selected_symbol), None)

    if asset_info:
        rec = asset_info["recommendation"]
        pred = asset_info["prediction"]
        sent = asset_info["sentiment"]
        levels = rec.get("levels", {})

        col_a, col_b, col_c = st.columns([1, 1, 1.2])

        with col_a:
            st.markdown(f"### {asset_info['name']}")
            st.write(f"**Symbol:** `{selected_symbol}` | **Exchange:** `{asset_info['exchange']}`")
            st.write(f"**Sector:** {asset_info['sector']}")
            st.metric("Current Price", f"{asset_info['current_price']:,.2f} {asset_info['currency']}")
            st.metric("Predicted (n+1)", f"{pred['predicted_price_n1']:,.2f} {asset_info['currency']}", f"{pred['predicted_return_n1']*100:+.2f}%")

        with col_b:
            st.markdown("### Actionable Recommendation")
            st.write(f"**Position:** `{rec['position']}` (Confidence: {rec['confidence']})")
            st.write(f"**Target (TP):** `{rec['take_profit']:,.2f}` | **Stop Loss (SL):** `{rec['stop_loss']:,.2f}`")
            st.write(f"**Risk/Reward Ratio:** `{rec['risk_reward_ratio']}x`")
            st.info(f"**Quantitative Rationale:** {rec['rationale']}")

        with col_c:
            st.markdown("### Alternative Sentiment Breakdown")
            st.write(f"• **Social Mentions:** {sent['social_mentions']} posts")
            st.write(f"• **Weighted Social Sentiment:** `{sent['weighted_social_sentiment']:+.3f}`")
            st.write(f"• **Exa News Sentiment:** `{sent['news_sentiment']:+.3f}`")
            st.write(f"• **Retail vs Institutional Divergence:** `{sent['retail_smart_divergence']:+.3f}`")
            st.write(f"• **Total Viral Engagement:** {sent.get('total_likes', 0)} likes, {sent.get('total_reposts', 0)} reposts")

with tab_technical:
    st.subheader("Support, Resistance & Boundary Analysis")
    tech_table = []
    for r in results:
        lv = r["recommendation"].get("levels", {})
        tech_table.append({
            "Symbol": r["symbol"],
            "Current": r["current_price"],
            "Support S2": lv.get("s2", 0),
            "Support S1": lv.get("s1", 0),
            "Pivot Point": lv.get("pivot", 0),
            "Resistance R1": lv.get("r1", 0),
            "Resistance R2": lv.get("r2", 0),
            "ATR (10)": lv.get("atr", 0),
            "10-Day High": lv.get("high_10d", 0),
            "10-Day Low": lv.get("low_10d", 0),
        })
    st.dataframe(pd.DataFrame(tech_table), use_container_width=True, hide_index=True)

with tab_architecture:
    st.markdown("""
    ### Pipeline Architecture & Validation Framework
    
    1. **Data Ingestion**:
       - **Apify Scrapers**: Target localized retail investor hubs (LIHKG and Futu/Moomoo for Hang Seng; Naver Finance 종목토론실 and Daum Finance for KRX).
       - **Exa.com Ingestion**: Neural search across Tier-1 financial publishers (Reuters, Bloomberg, SCMP, Korea Times, KED Global).
       - **OHLCV Market Bars**: UTC normalized daily/hourly intervals via Yahoo Finance.
    
    2. **Engagement Weighting Algorithm**:
       $$W_i = \\ln(1 + \\text{Followers}_i) \\cdot \\left(1 + 0.25 \\ln(1 + \\text{Likes}_i) + 0.45 \\ln(1 + \\text{Shares}_i)\\right) \\cdot e^{-\\lambda \\cdot \\Delta t}$$
    
    3. **Retail vs. Smart Money Divergence (RSMD)**:
       $$\\Delta_{\\text{RSMD}} = \\text{Sentiment}_{\\text{Social}} - \\text{Sentiment}_{\\text{News}}$$
       Identifies overbought distribution traps when retail sentiment decouples positively from institutional news.
    
    4. **Rigorous Out-of-Sample Validation**:
       - Implements `TimeSeriesSplit` walk-forward evaluation.
       - Blended ensemble: L2 Regularized Ridge Regression + Gradient Boosted Trees (XGBoost).
    """)
