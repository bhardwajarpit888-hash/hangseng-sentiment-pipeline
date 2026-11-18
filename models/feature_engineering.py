from typing import List, Tuple
import pandas as pd
import numpy as np

FEATURE_COLUMNS = [
    "return_lag1",
    "ratio_sma_5_lag1",
    "ratio_sma_10_lag1",
    "rsi_14_lag1",
    "macd_hist_lag1",
    "atr_norm_lag1",
    "vol_ratio_lag1",
    "weighted_sentiment",
    "social_mentions",
    "news_sentiment",
    "retail_smart_divergence",
    "sentiment_momentum",
    "mention_surge"
]


def prepare_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Extracts features X and targets y:
    - y_return: target return at n+1
    - y_close: target close price at n+1
    - y_direction: target direction (1=Bullish, 0=Bearish)
    """
    clean_df = df.dropna(subset=["target_return_n1", "target_close_n1"]).copy()

    for col in FEATURE_COLUMNS:
        if col not in clean_df.columns:
            clean_df[col] = 0.0

    X = clean_df[FEATURE_COLUMNS].copy()
    y_return = clean_df["target_return_n1"].copy()
    y_close = clean_df["target_close_n1"].copy()
    y_direction = clean_df["target_direction_n1"].copy()

    return X, y_return, y_close, y_direction
