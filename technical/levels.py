import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional


class SupportResistanceEngine:
    """
    Computes key market levels:
    1. Classical Floor Pivots (PP, S1, R1, S2, R2)
    2. Fibonacci Pivots (0.618 extensions)
    3. Swing High / Swing Low (local extrema price rejections)
    4. ATR Volatility Bands
    """

    def calculate_levels(self, df: pd.DataFrame, post_mentioned: Optional[List[float]] = None) -> Dict[str, Any]:
        if df.empty or len(df) < 2:
            return {
                "pivot": 0.0, "s1": 0.0, "s2": 0.0, "s3": 0.0,
                "r1": 0.0, "r2": 0.0, "r3": 0.0,
                "fib_s618": 0.0, "fib_r618": 0.0,
                "swing_high": 0.0, "swing_low": 0.0,
                "post_mentioned_levels": post_mentioned or [],
                "atr": 0.0
            }

        recent_bars = df.tail(min(30, len(df)))
        high = float(recent_bars["high"].max())
        low = float(recent_bars["low"].min())
        close = float(recent_bars["close"].iloc[-1])

        # Classical Floor Pivots
        pp = (high + low + close) / 3.0
        r1 = (2.0 * pp) - low
        s1 = (2.0 * pp) - high
        r2 = pp + (high - low)
        s2 = pp - (high - low)
        r3 = high + 2.0 * (pp - low)
        s3 = low - 2.0 * (high - pp)

        # Fibonacci Pivots
        range_hl = max(1e-6, high - low)
        fib_r618 = pp + (0.618 * range_hl)
        fib_s618 = pp - (0.618 * range_hl)

        # Swing Extrema: local highs and lows over rolling window
        swing_window = min(10, len(df))
        swing_high = float(df["high"].tail(swing_window).max())
        swing_low = float(df["low"].tail(swing_window).min())

        # Average True Range (14-period or available history)
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["close"].shift(1)).abs()
        tr3 = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr_window = min(14, len(tr))
        atr = float(tr.tail(atr_window).mean()) if len(tr) > 0 else (high - low) * 0.05

        return {
            "pivot": round(pp, 2),
            "s1": round(s1, 2),
            "s2": round(s2, 2),
            "s3": round(s3, 2),
            "r1": round(r1, 2),
            "r2": round(r2, 2),
            "r3": round(r3, 2),
            "fib_s618": round(fib_s618, 2),
            "fib_r618": round(fib_r618, 2),
            "swing_high": round(swing_high, 2),
            "swing_low": round(swing_low, 2),
            "post_mentioned_levels": post_mentioned or [],
            "atr": round(atr, 2),
            "lookback_high": round(high, 2),
            "lookback_low": round(low, 2),
        }


class PositionRecommendationEngine:
    """
    Separates stated retail positions (extracted from post text)
    from suggested quantitative model stance (derived from risk/reward, pivots & sentiment).
    """

    def __init__(self):
        self.levels_calc = SupportResistanceEngine()

    def generate_recommendation(
        self,
        symbol: str,
        price_df: pd.DataFrame,
        sentiment_metrics: Dict[str, Any]
    ) -> Dict[str, Any]:
        if price_df.empty:
            return {}

        current_price = float(price_df["close"].iloc[-1])
        post_levels = sentiment_metrics.get("post_mentioned_levels", [])
        levels = self.levels_calc.calculate_levels(price_df, post_mentioned=post_levels)

        sentiment_score = sentiment_metrics.get("composite_sentiment", 0.0)
        direction = sentiment_metrics.get("sentiment_direction", "NEUTRAL")
        divergence = sentiment_metrics.get("retail_smart_divergence", 0.0)
        atr = levels["atr"]

        # Retail crowd's stated position
        stated_pos = sentiment_metrics.get("dominant_stated_position", "NONE")
        stated_counts = sentiment_metrics.get("stated_position_counts", {"long": 0, "short": 0, "flat": 0})

        # Moving average trend check
        ma_window = min(20, len(price_df))
        ma_trend = float(price_df["close"].tail(ma_window).mean()) if len(price_df) >= 3 else current_price
        trend_bullish = current_price >= ma_trend

        # Quantitative suggested stance
        suggested_stance = "NEUTRAL"
        confidence = "MEDIUM"
        size = 0.0
        rationale = ""

        if direction == "BULLISH" and sentiment_score > 0.12:
            if divergence > 0.40 and current_price >= levels["r1"]:
                suggested_stance = "NEUTRAL"
                confidence = "LOW"
                rationale = "Extreme retail hype divergence near resistance R1; distribution exhaustion risk. Maintain neutral stance."
            else:
                suggested_stance = "LONG"
                size = 1.0 if sentiment_score > 0.30 and trend_bullish else 0.5
                confidence = "HIGH" if sentiment_score > 0.30 else "MEDIUM"
                rationale = f"Composite sentiment (+{sentiment_score:.2f}) aligning above pivot {levels['pivot']}."

        elif direction == "BEARISH" and sentiment_score < -0.12:
            if divergence < -0.40 and current_price <= levels["s1"]:
                suggested_stance = "NEUTRAL"
                confidence = "LOW"
                rationale = "Capitulation divergence near support S1; potential short-squeeze bounce. Advise neutral."
            else:
                suggested_stance = "SHORT"
                size = 1.0 if sentiment_score < -0.30 and not trend_bullish else 0.5
                confidence = "HIGH" if sentiment_score < -0.30 else "MEDIUM"
                rationale = f"Bearish sentiment pressure ({sentiment_score:.2f}) below pivot {levels['pivot']}."
        else:
            suggested_stance = "NEUTRAL"
            size = 0.0
            confidence = "LOW"
            rationale = "Neutral sentiment bounds and price consolidating within pivot range."

        # Risk parameters
        if suggested_stance == "LONG":
            tp = round(max(levels["r1"], levels["swing_high"], current_price + 1.5 * atr), 2)
            sl = round(min(levels["s1"], current_price - 1.0 * atr), 2)
            rr = round((tp - current_price) / max(0.01, (current_price - sl)), 2)
        elif suggested_stance == "SHORT":
            tp = round(min(levels["s1"], levels["swing_low"], current_price - 1.5 * atr), 2)
            sl = round(max(levels["r1"], current_price + 1.0 * atr), 2)
            rr = round((current_price - tp) / max(0.01, (sl - current_price)), 2)
        else:
            tp = round(levels["r1"], 2)
            sl = round(levels["s1"], 2)
            rr = 1.0

        return {
            "current_price": current_price,
            "suggested_stance": suggested_stance,
            "stated_position": stated_pos,
            "stated_position_counts": stated_counts,
            "position": suggested_stance,  # backward compatibility alias
            "position_size": size,
            "confidence": confidence,
            "take_profit": tp,
            "stop_loss": sl,
            "risk_reward_ratio": rr,
            "levels": levels,
            "rationale": rationale
        }
