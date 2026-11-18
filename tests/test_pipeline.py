import unittest
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from config.settings import EXCHANGE_CONFIGS
from sentiment.analyzer import MultilingualFinancialSentimentAnalyzer
from sentiment.engagement_weighting import EngagementWeightingEngine
from technical.levels import SupportResistanceEngine, PositionRecommendationEngine
from data.scrapers.apify_client import ApifySocialScraper
from data.scrapers.exa_news import ExaNewsIngestor
from data.processor import DataAlignmentProcessor
from models.predictor import PricePredictionModel


class TestSentimentPredictionPipeline(unittest.TestCase):

    def setUp(self):
        self.sentiment_analyzer = MultilingualFinancialSentimentAnalyzer()
        self.weighting_engine = EngagementWeightingEngine()
        self.levels_engine = SupportResistanceEngine()
        self.apify_scraper = ApifySocialScraper()
        self.exa_ingestor = ExaNewsIngestor()

    def test_multilingual_sentiment_analyzer(self):
        # Cantonese / Traditional Chinese
        hk_res = self.sentiment_analyzer.analyze_text("騰訊 業績超預期，今日大成交破位，準備見新高！")
        self.assertEqual(hk_res["direction"], "BULLISH")
        self.assertGreater(hk_res["score"], 0.2)

        # Korean
        kr_res = self.sentiment_analyzer.analyze_text("SK하이닉스 실적 서프라이즈 대박! 외인 순매수 유입 중 떡상 각입니다.")
        self.assertEqual(kr_res["direction"], "BULLISH")
        self.assertGreater(kr_res["score"], 0.2)

        # Bearish Korean
        kr_bear = self.sentiment_analyzer.analyze_text("삼성전자 지지선 무너졌네요... 외인들 대량 매도폭탄 던지는 중.")
        self.assertEqual(kr_bear["direction"], "BEARISH")
        self.assertLess(kr_bear["score"], -0.2)

    def test_engagement_weighting(self):
        now = datetime.now(timezone.utc)
        viral_post = {
            "timestamp_utc": now.isoformat(),
            "author_followers": 25000,
            "likes": 450,
            "reposts": 80,
            "views": 12000,
            "sentiment_score": 0.8
        }
        small_post = {
            "timestamp_utc": now.isoformat(),
            "author_followers": 10,
            "likes": 1,
            "reposts": 0,
            "views": 25,
            "sentiment_score": 0.1
        }

        w_viral = self.weighting_engine.compute_post_weight(viral_post, now)
        w_small = self.weighting_engine.compute_post_weight(small_post, now)
        self.assertGreater(w_viral, w_small * 3.0)

    def test_support_resistance_levels(self):
        dates = pd.date_range("2026-11-01", periods=10, freq="B", tz="UTC")
        sample_df = pd.DataFrame({
            "open": [100.0 + i for i in range(10)],
            "high": [105.0 + i for i in range(10)],
            "low": [98.0 + i for i in range(10)],
            "close": [102.0 + i for i in range(10)],
            "volume": [1_000_000 for _ in range(10)]
        }, index=dates)

        levels = self.levels_engine.calculate_levels(sample_df)
        self.assertIn("pivot", levels)
        self.assertIn("s1", levels)
        self.assertIn("r1", levels)
        self.assertLess(levels["s1"], levels["pivot"])
        self.assertGreater(levels["r1"], levels["pivot"])

    def test_apify_and_exa_scrapers(self):
        ticker_cfg = EXCHANGE_CONFIGS["HKEX"].tickers["0700.HK"]
        posts = self.apify_scraper.scrape_ticker_socials(ticker_cfg, ["LIHKG", "FUTU_MOOMOO"], lookback_days=5, max_posts_per_source=5)
        self.assertGreaterEqual(len(posts), 5)
        self.assertEqual(posts[0]["ticker"], "0700.HK")
        self.assertIn("timestamp_utc", posts[0])

        news = self.exa_ingestor.fetch_recent_news(ticker_cfg, lookback_days=5, max_articles=3)
        self.assertGreaterEqual(len(news), 2)
        self.assertEqual(news[0]["ticker"], "0700.HK")

    def test_prediction_model(self):
        dates = pd.date_range("2026-10-01", periods=25, freq="B", tz="UTC")
        np.random.seed(42)
        prices = 100.0 * np.cumprod(1 + np.random.normal(0.001, 0.015, size=25))

        df = pd.DataFrame({
            "open": prices * 0.99,
            "high": prices * 1.01,
            "low": prices * 0.98,
            "close": prices,
            "volume": np.random.uniform(1e6, 5e6, size=25)
        }, index=dates)

        processor = DataAlignmentProcessor()
        ticker_cfg = EXCHANGE_CONFIGS["HKEX"].tickers["0700.HK"]
        posts = self.apify_scraper.scrape_ticker_socials(ticker_cfg, ["LIHKG"], lookback_days=10, max_posts_per_source=10)
        news = self.exa_ingestor.fetch_recent_news(ticker_cfg, lookback_days=10, max_articles=5)

        aligned = processor.align_and_engineer_features("0700.HK", df, posts, news)
        predictor = PricePredictionModel()
        pred_res = predictor.fit_and_predict_n1("0700.HK", aligned)

        self.assertIn("predicted_price_n1", pred_res)
        self.assertIn("oos_validation", pred_res)
        self.assertGreater(pred_res["predicted_price_n1"], 0.0)
        self.assertIn("directional_accuracy", pred_res["oos_validation"])


if __name__ == "__main__":
    unittest.main()
