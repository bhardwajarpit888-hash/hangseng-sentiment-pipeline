from typing import List, Tuple
import pandas as pd
import numpy as np

# Feature list: All features in row k are strictly finalized at or before bar k close.
# Target return is bar k+1 return: (Close_{k+1} - Close_k) / Close_k.
FEATURE_COLUMNS = [
    "return",
    "ratio_sma_5",
    "ratio_sma_10",
    "rsi_14",
    "macd_hist",
    "atr_norm",
    "vol_ratio",
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
    - y_return: target return of bar k+1
    - y_close: target close price of bar k+1
    - y_direction: target direction (1=Bullish, 0=Bearish)
    Drops initial rows containing NaN warmups to prevent fake zero fills.
    """
    required_cols = FEATURE_COLUMNS + ["target_return_n1", "target_close_n1"]
    existing_cols = [c for c in required_cols if c in df.columns]

    clean_df = df.dropna(subset=existing_cols).copy()

    for col in FEATURE_COLUMNS:
        if col not in clean_df.columns:
            clean_df[col] = 0.0

    X = clean_df[FEATURE_COLUMNS].copy()
    y_return = clean_df["target_return_n1"].copy()
    y_close = clean_df["target_close_n1"].copy()
    y_direction = (y_return > 0).astype(int)

    return X, y_return, y_close, y_direction
