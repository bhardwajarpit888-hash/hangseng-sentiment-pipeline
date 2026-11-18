import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from sentiment.analyzer import MultilingualFinancialSentimentAnalyzer
from sentiment.engagement_weighting import EngagementWeightingEngine
from technical.indicators import enrich_technical_features

logger = logging.getLogger(__name__)


class DataAlignmentProcessor:
    """
    Aligns asynchronous unstructured social feeds and news streams
    with structured OHLCV market bars in strict UTC time coordinates.
    Computes lag features to strictly prevent lookahead bias.
    """

    def __init__(self):
        self.sentiment_analyzer = MultilingualFinancialSentimentAnalyzer()
        self.weighting_engine = EngagementWeightingEngine()

    def align_and_engineer_features(
        self,
        symbol: str,
        price_df: pd.DataFrame,
        social_posts: List[Dict[str, Any]],
        news_articles: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        if price_df.empty:
            return pd.DataFrame()

        df = price_df.copy()
        if "return" not in df.columns:
            df["return"] = df["close"].pct_change().fillna(0.0)
        if "log_return" not in df.columns:
            df["log_return"] = np.log(df["close"] / df["close"].shift(1)).fillna(0.0)
        if "target_return_n1" not in df.columns:
            df["target_return_n1"] = df["return"].shift(-1)
        if "target_close_n1" not in df.columns:
            df["target_close_n1"] = df["close"].shift(-1)
        if "target_direction_n1" not in df.columns:
            df["target_direction_n1"] = (df["target_return_n1"] > 0).astype(int)

        df = enrich_technical_features(df)

        # Score social posts and news if not already scored
        social_posts = self.sentiment_analyzer.score_batch(social_posts)
        news_articles = self.sentiment_analyzer.score_batch(news_articles)

        # Convert to DataFrames for time-based joining
        social_df = pd.DataFrame(social_posts)
        news_df = pd.DataFrame(news_articles)

        if not social_df.empty:
            social_df["ts"] = pd.to_datetime(social_df["timestamp_utc"], utc=True)
        if not news_df.empty:
            news_df["ts"] = pd.to_datetime(news_df["timestamp_utc"], utc=True)

        bar_dates = df.index.tz_convert("UTC") if df.index.tz else df.index.tz_localize("UTC")

        # Aggregate sentiment metrics for each price bar interval
        bar_social_sentiment = []
        bar_weighted_sentiment = []
        bar_mentions = []
        bar_news_sentiment = []
        bar_news_count = []
        bar_divergence = []

        for i in range(len(df)):
            current_bar_time = bar_dates[i]
            prev_bar_time = bar_dates[i - 1] if i > 0 else (current_bar_time - pd.Timedelta(days=1))

            # Filter posts strictly up to the current bar time
            if not social_df.empty:
                mask = (social_df["ts"] > prev_bar_time) & (social_df["ts"] <= current_bar_time)
                bar_posts = social_posts_subset = social_df[mask].to_dict("records")
            else:
                bar_posts = []

            if not news_df.empty:
                n_mask = (news_df["ts"] > prev_bar_time) & (news_df["ts"] <= current_bar_time)
                bar_news = news_df[n_mask].to_dict("records")
            else:
                bar_news = []

            metrics = self.weighting_engine.aggregate_sentiment_metrics(
                bar_posts,
                bar_news,
                ref_time=current_bar_time.to_pydatetime()
            )

            bar_social_sentiment.append(metrics.get("raw_social_sentiment", 0.0))
            bar_weighted_sentiment.append(metrics.get("weighted_social_sentiment", 0.0))
            bar_mentions.append(metrics.get("social_mentions", 0))
            bar_news_sentiment.append(metrics.get("news_sentiment", 0.0))
            bar_news_count.append(metrics.get("news_articles", 0))
            bar_divergence.append(metrics.get("retail_smart_divergence", 0.0))

        df["social_sentiment"] = bar_social_sentiment
        df["weighted_sentiment"] = bar_weighted_sentiment
        df["social_mentions"] = bar_mentions
        df["news_sentiment"] = bar_news_sentiment
        df["news_count"] = bar_news_count
        df["retail_smart_divergence"] = bar_divergence

        # Sentiment momentum and surge dynamics
        df["sentiment_momentum"] = df["weighted_sentiment"].diff().fillna(0.0)
        df["mentions_ma3"] = df["social_mentions"].rolling(3, min_periods=1).mean()
        df["mention_surge"] = (df["social_mentions"] / df["mentions_ma3"].replace(0, 1.0) - 1.0).fillna(0.0)

        # Lagged features to eliminate any lookahead leakage
        feature_cols = [
            "return", "log_return", "ratio_sma_5", "ratio_sma_10",
            "rsi_14", "macd_hist", "atr_norm", "vol_ratio",
            "social_sentiment", "weighted_sentiment", "social_mentions",
            "news_sentiment", "retail_smart_divergence", "sentiment_momentum", "mention_surge"
        ]

        for col in feature_cols:
            df[f"{col}_lag1"] = df[col].shift(1).fillna(0.0)

        return df
