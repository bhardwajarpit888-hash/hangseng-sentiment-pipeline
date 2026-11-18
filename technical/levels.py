import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple


class SupportResistanceEngine:
    """
    Computes key market levels including Classic Floor Pivots, Fibonacci Pivots,
    local swing extrema, and ATR volatility boundaries.
    """

    def calculate_levels(self, df: pd.DataFrame) -> Dict[str, float]:
        if df.empty or len(df) < 2:
            return {
                "pivot": 0.0, "s1": 0.0, "s2": 0.0, "s3": 0.0,
                "r1": 0.0, "r2": 0.0, "r3": 0.0,
                "fib_s618": 0.0, "fib_r618": 0.0,
                "atr": 0.0
            }

        recent_bars = df.tail(10)
        high = float(recent_bars["high"].max())
        low = float(recent_bars["low"].min())
        close = float(recent_bars["close"].iloc[-1])

        # Classical Pivot Points
        pp = (high + low + close) / 3.0
        r1 = (2.0 * pp) - low
        s1 = (2.0 * pp) - high
        r2 = pp + (high - low)
        s2 = pp - (high - low)
        r3 = high + 2.0 * (pp - low)
        s3 = low - 2.0 * (high - pp)

        # Fibonacci Pivots
        range_hl = high - low
        fib_r618 = pp + (0.618 * range_hl)
        fib_s618 = pp - (0.618 * range_hl)

        # ATR computation
        tr1 = df["high"] - df["low"]
        tr2 = (df["high"] - df["close"].shift(1)).abs()
        tr3 = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = float(tr.tail(10).mean()) if len(tr) > 0 else (high - low) * 0.1

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
            "atr": round(atr, 2),
            "high_10d": round(high, 2),
            "low_10d": round(low, 2),
        }


class PositionRecommendationEngine:
    """
    Synthesizes multi-source sentiment with technical support/resistance levels
    to generate actionable risk-managed position recommendations.
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
        levels = self.levels_calc.calculate_levels(price_df)

        sentiment_score = sentiment_metrics.get("composite_sentiment", 0.0)
        direction = sentiment_metrics.get("sentiment_direction", "NEUTRAL")
        divergence = sentiment_metrics.get("retail_smart_divergence", 0.0)
        atr = levels["atr"]

        # 5-day and 10-day trend
        ma5 = float(price_df["close"].tail(5).mean()) if len(price_df) >= 5 else current_price
        trend_bullish = current_price >= ma5

        # Position sizing logic
        position = "NEUTRAL"
        confidence = "MEDIUM"
        size = 0.0
        rationale = ""

        # Bullish setup: Positive sentiment supported by technical structure
        if direction == "BULLISH" and sentiment_score > 0.15:
            if divergence > 0.45 and current_price >= levels["r1"]:
                # Retail euphoria trap near resistance
                position = "NEUTRAL"
                confidence = "LOW"
                rationale = "Extreme retail hype divergence near resistance level R1; potential exhaustion trap. Advise waiting for pullback."
            else:
                position = "LONG"
                size = 1.0 if sentiment_score > 0.35 and trend_bullish else 0.5
                confidence = "HIGH" if sentiment_score > 0.35 else "MEDIUM"
                rationale = f"Strong multi-source sentiment (+{sentiment_score:.2f}) aligning with price consolidating above pivot {levels['pivot']}."

        # Bearish setup: Negative sentiment breaking support
        elif direction == "BEARISH" and sentiment_score < -0.15:
            if divergence < -0.45 and current_price <= levels["s1"]:
                # Extreme retail panic near support - potential contrarian bounce
                position = "NEUTRAL"
                confidence = "LOW"
                rationale = "Severe retail capitulation at support level S1; high likelihood of short-squeeze bounce. Sideline until confirmation."
            else:
                position = "SHORT"
                size = 1.0 if sentiment_score < -0.35 and not trend_bullish else 0.5
                confidence = "HIGH" if sentiment_score < -0.35 else "MEDIUM"
                rationale = f"Negative sentiment inflection ({sentiment_score:.2f}) coupled with overhead pressure below pivot {levels['pivot']}."

        else:
            position = "NEUTRAL"
            size = 0.0
            confidence = "LOW"
            rationale = "Balanced sentiment signals and price trading within historical pivot bounds."

        # Risk parameters
        if position == "LONG":
            tp = round(max(levels["r1"], current_price + 1.5 * atr), 2)
            sl = round(min(levels["s1"], current_price - 1.0 * atr), 2)
            rr = round((tp - current_price) / max(0.01, (current_price - sl)), 2)
        elif position == "SHORT":
            tp = round(min(levels["s1"], current_price - 1.5 * atr), 2)
            sl = round(max(levels["r1"], current_price + 1.0 * atr), 2)
            rr = round((current_price - tp) / max(0.01, (sl - current_price)), 2)
        else:
            tp = round(levels["r1"], 2)
            sl = round(levels["s1"], 2)
            rr = 1.0

        return {
            "current_price": current_price,
            "position": position,
            "position_size": size,
            "confidence": confidence,
            "take_profit": tp,
            "stop_loss": sl,
            "risk_reward_ratio": rr,
            "levels": levels,
            "rationale": rationale
        }
