import logging
from typing import List, Dict, Any, Tuple
from datetime import datetime, timezone, timedelta
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

    ALIGNMENT RULE:
    Each bar k has a designated close timestamp t_close_k.
    A social post or news article published at timestamp t belongs to bar k
    if and only if: t_close_{k-1} < t <= t_close_k.
    Therefore, all features calculated for bar k use data strictly available
    at or before t_close_k.
    The prediction target is the return of bar k+1: R_{k+1} = (Close_{k+1} - Close_k) / Close_k.
    This guarantees zero lookahead leakage into future bars.
    """

    def __init__(self, use_transformer: bool = False):
        self.sentiment_analyzer = MultilingualFinancialSentimentAnalyzer(use_transformer=use_transformer)
        self.weighting_engine = EngagementWeightingEngine()

    def align_and_engineer_features(
        self,
        symbol: str,
        price_df: pd.DataFrame,
        social_posts: List[Dict[str, Any]],
        news_articles: List[Dict[str, Any]],
        bar_interval_minutes: int = 60
    ) -> pd.DataFrame:
        if price_df.empty or len(price_df) < 5:
            return pd.DataFrame()

        df = price_df.copy()

        # Returns and forward target definition
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

        # Technical indicators computed strictly on past & current close prices
        df = enrich_technical_features(df)

        # Score social and news items
        social_posts = self.sentiment_analyzer.score_batch(social_posts)
        news_articles = self.sentiment_analyzer.score_batch(news_articles)

        social_df = pd.DataFrame(social_posts)
        news_df = pd.DataFrame(news_articles)

        if not social_df.empty:
            social_df["ts"] = pd.to_datetime(social_df["timestamp_utc"], utc=True)
        if not news_df.empty:
            news_df["ts"] = pd.to_datetime(news_df["timestamp_utc"], utc=True)

        # Determine bar open and close timestamps
        bar_index = df.index.tz_convert("UTC") if df.index.tz else df.index.tz_localize("UTC")
        delta = timedelta(minutes=bar_interval_minutes)
        bar_close_times = [t + delta for t in bar_index]
        df["bar_close_utc"] = bar_close_times

        bar_social_sentiment = []
        bar_weighted_sentiment = []
        bar_mentions = []
        bar_news_sentiment = []
        bar_news_count = []
        bar_divergence = []

        for i in range(len(df)):
            t_close_curr = bar_close_times[i]
            t_close_prev = bar_close_times[i - 1] if i > 0 else (t_close_curr - delta)

            # Strict time partition: t_close_prev < t <= t_close_curr
            if not social_df.empty:
                s_mask = (social_df["ts"] > t_close_prev) & (social_df["ts"] <= t_close_curr)
                bar_posts = social_df[s_mask].to_dict("records")
            else:
                bar_posts = []

            if not news_df.empty:
                n_mask = (news_df["ts"] > t_close_prev) & (news_df["ts"] <= t_close_curr)
                bar_news = news_df[n_mask].to_dict("records")
            else:
                bar_news = []

            metrics = self.weighting_engine.aggregate_sentiment_metrics(
                bar_posts,
                bar_news,
                ref_time=t_close_curr.to_pydatetime()
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

        # Rolling sentiment momentum and attention surge
        df["sentiment_momentum"] = df["weighted_sentiment"].diff().fillna(0.0)
        df["mentions_ma3"] = df["social_mentions"].rolling(3, min_periods=1).mean()
        df["mention_surge"] = (df["social_mentions"] / df["mentions_ma3"].replace(0, 1.0) - 1.0).fillna(0.0)

        # Drop the warmup rows where technical indicators have NaNs instead of filling fake zeros
        # This keeps our training data pure without fabricated zero-fill noise
        df = df.dropna(subset=["return", "rsi_14", "macd_hist", "ratio_sma_5"])

        return df
