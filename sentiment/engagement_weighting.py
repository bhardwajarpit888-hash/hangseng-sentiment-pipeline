import math
from typing import List, Dict, Any
from datetime import datetime, timezone
import pandas as pd
import numpy as np


class EngagementWeightingEngine:
    """
    Computes engagement-weighted sentiment indices by weighting posts based on
    reach (author followers), viral interactions (likes, reposts, comments/views),
    and time decay across the lookback horizon.
    """

    def __init__(self, half_life_days: float = 3.0):
        self.half_life_seconds = half_life_days * 86400

    def compute_post_weight(self, post: Dict[str, Any], ref_time: datetime) -> float:
        followers = max(0, int(post.get("author_followers", 0)))
        likes = max(0, int(post.get("likes", 0)))
        reposts = max(0, int(post.get("reposts", 0)))
        views = max(0, int(post.get("views", 0)))

        # Follower reach factor: dampened log scale
        reach_factor = math.log10(1 + followers) + 1.0

        # Virality factor
        viral_factor = 1.0 + (0.25 * math.log1p(likes)) + (0.45 * math.log1p(reposts)) + (0.10 * math.log1p(views))

        # Time decay factor: recent posts matter exponentially more
        post_ts = pd.to_datetime(post.get("timestamp_utc"), utc=True).to_pydatetime()
        age_seconds = max(0, (ref_time - post_ts).total_seconds())
        decay_factor = math.exp(-0.693147 * (age_seconds / self.half_life_seconds))

        return reach_factor * viral_factor * decay_factor

    def aggregate_sentiment_metrics(
        self,
        social_posts: List[Dict[str, Any]],
        news_articles: List[Dict[str, Any]],
        ref_time: datetime = None
    ) -> Dict[str, Any]:
        """
        Combines social and news streams into a unified set of quantitative sentiment signals.
        """
        if ref_time is None:
            ref_time = datetime.now(timezone.utc)

        total_social_mentions = len(social_posts)
        total_news_count = len(news_articles)

        if total_social_mentions == 0:
            return {
                "social_mentions": 0,
                "news_articles": total_news_count,
                "raw_social_sentiment": 0.0,
                "weighted_social_sentiment": 0.0,
                "news_sentiment": 0.0,
                "composite_sentiment": 0.0,
                "sentiment_direction": "NEUTRAL",
                "bullish_ratio": 0.33,
                "bearish_ratio": 0.33,
                "neutral_ratio": 0.34,
                "retail_smart_divergence": 0.0,
                "engagement_volume_score": 0.0,
                "dominant_stated_position": "NONE",
                "stated_position_counts": {"long": 0, "short": 0, "flat": 0},
                "post_mentioned_levels": []
            }

        # Calculate social weighted sentiment
        weights = []
        raw_scores = []
        likes_acc = 0
        reposts_acc = 0
        views_acc = 0
        bull_cnt = 0
        bear_cnt = 0
        neut_cnt = 0

        pos_long = 0
        pos_short = 0
        pos_flat = 0
        mentioned_levels_set = set()

        for p in social_posts:
            score = p.get("sentiment_score", 0.0)
            raw_scores.append(score)
            w = self.compute_post_weight(p, ref_time)
            weights.append(w)

            likes_acc += p.get("likes", 0)
            reposts_acc += p.get("reposts", 0)
            views_acc += p.get("views", 0)

            # Extract stated retail stance
            st_pos = p.get("stated_position")
            if st_pos == "LONG":
                pos_long += 1
            elif st_pos == "SHORT":
                pos_short += 1
            elif st_pos == "FLAT":
                pos_flat += 1

            # Extract mentioned levels
            for lv in p.get("mentioned_levels", []):
                mentioned_levels_set.add(lv)

            if score >= 0.10:
                bull_cnt += 1
            elif score <= -0.10:
                bear_cnt += 1
            else:
                neut_cnt += 1

        if pos_long > pos_short and pos_long >= 2:
            dominant_stated = "LONG"
        elif pos_short > pos_long and pos_short >= 2:
            dominant_stated = "SHORT"
        elif pos_flat > 0 and pos_flat >= pos_long and pos_flat >= pos_short:
            dominant_stated = "FLAT"
        else:
            dominant_stated = "NONE"

        raw_social_mean = float(np.mean(raw_scores))
        sum_weights = sum(weights)
        if sum_weights > 0:
            weighted_social_sentiment = float(np.average(raw_scores, weights=weights))
        else:
            weighted_social_sentiment = raw_social_mean

        # News sentiment aggregation
        if total_news_count > 0:
            news_scores = [n.get("sentiment_score", 0.0) for n in news_articles]
            news_sentiment = float(np.mean(news_scores))
        else:
            news_sentiment = 0.0

        # Composite sentiment (65% weighted social, 35% institutional news)
        composite = (0.65 * weighted_social_sentiment) + (0.35 * news_sentiment)

        # Creative divergence: Retail crowd vs Institutional smart money
        # Positive = Retail is more bullish than news/institutions
        # Negative = Retail is panicking while news is calm/bullish
        rs_divergence = weighted_social_sentiment - news_sentiment

        if composite >= 0.12:
            direction = "BULLISH"
        elif composite <= -0.12:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        return {
            "social_mentions": total_social_mentions,
            "news_articles": total_news_count,
            "raw_social_sentiment": round(raw_social_mean, 4),
            "weighted_social_sentiment": round(weighted_social_sentiment, 4),
            "news_sentiment": round(news_sentiment, 4),
            "composite_sentiment": round(composite, 4),
            "sentiment_direction": direction,
            "bullish_ratio": round(bull_cnt / total_social_mentions, 4),
            "bearish_ratio": round(bear_cnt / total_social_mentions, 4),
            "neutral_ratio": round(neut_cnt / total_social_mentions, 4),
            "retail_smart_divergence": round(rs_divergence, 4),
            "total_likes": likes_acc,
            "total_reposts": reposts_acc,
            "total_views": views_acc,
            "engagement_volume_score": round(math.log10(1 + likes_acc * 2 + reposts_acc * 3 + views_acc * 0.1), 3),
            "dominant_stated_position": dominant_stated,
            "stated_position_counts": {"long": pos_long, "short": pos_short, "flat": pos_flat},
            "post_mentioned_levels": sorted(list(mentioned_levels_set))[:5]
        }
