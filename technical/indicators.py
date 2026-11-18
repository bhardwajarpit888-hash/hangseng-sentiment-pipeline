import numpy as np
import pandas as pd


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd_line = exp1 - exp2
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def compute_bollinger_bands(series: pd.Series, period: int = 20, num_std: float = 2.0):
    sma = series.rolling(window=period, min_periods=5).mean()
    rstd = series.rolling(window=period, min_periods=5).std().fillna(0.0)
    upper = sma + (num_std * rstd)
    lower = sma - (num_std * rstd)
    return upper, sma, lower


def enrich_technical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Enriches market price bars with technical oscillators and momentum indicators."""
    df = df.copy()
    close = df["close"]

    # Moving averages
    df["sma_5"] = close.rolling(5, min_periods=1).mean()
    df["sma_10"] = close.rolling(10, min_periods=1).mean()
    df["ratio_sma_5"] = close / df["sma_5"].replace(0, np.nan) - 1.0
    df["ratio_sma_10"] = close / df["sma_10"].replace(0, np.nan) - 1.0

    # RSI
    df["rsi_14"] = compute_rsi(close, 14)

    # MACD
    macd, sig, hist = compute_macd(close)
    df["macd_hist"] = hist.fillna(0.0)

    # Volatility
    high = df["high"]
    low = df["low"]
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    df["atr_norm"] = (tr.rolling(10, min_periods=1).mean() / close).fillna(0.01)

    # Volume momentum
    vol_sma = df["volume"].rolling(5, min_periods=1).mean()
    df["vol_ratio"] = (df["volume"] / vol_sma.replace(0, np.nan)).fillna(1.0)

    return df
