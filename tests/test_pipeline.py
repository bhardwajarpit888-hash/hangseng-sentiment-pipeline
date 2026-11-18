import unittest
from datetime import datetime, timezone, timedelta
from pathlib import Path
import pandas as pd
import numpy as np

from config.settings import EXCHANGE_CONFIGS
from sentiment.analyzer import MultilingualFinancialSentimentAnalyzer
from sentiment.engagement_weighting import EngagementWeightingEngine
from technical.levels import SupportResistanceEngine, PositionRecommendationEngine
from data.scrapers.apify_client import ApifySocialScraper
from data.scrapers.exa_news import ExaNewsIngestor
from data.processor import DataAlignmentProcessor
from models.validation import OutOfSampleValidator
from models.predictor import PricePredictionModel
from models.feature_engineering import prepare_feature_matrix, FEATURE_COLUMNS


class TestSentimentPredictionPipeline(unittest.TestCase):

    def setUp(self):
        self.sentiment_analyzer = MultilingualFinancialSentimentAnalyzer()
        self.weighting_engine = EngagementWeightingEngine()
        self.levels_engine = SupportResistanceEngine()
        self.position_engine = PositionRecommendationEngine()

    def test_multilingual_sentiment_and_position_extraction(self):
        # Cantonese Bullish + Long position
        hk_text = "騰訊 業績超預期，今日大成交破位，買入等開車！目標價 420"
        hk_res = self.sentiment_analyzer.analyze_text(hk_text)
        self.assertEqual(hk_res["direction"], "BULLISH")
        self.assertGreater(hk_res["score"], 0.2)
        self.assertEqual(hk_res["stated_position"], "LONG")
        self.assertIn(420.0, hk_res["mentioned_levels"])

        # Korean Bearish + Short position
        kr_text = "삼성전자 지지선 무너짐... 물려서 손절 고민 중, 숏 포지션 진입. 목표 45000"
        kr_res = self.sentiment_analyzer.analyze_text(kr_text)
        self.assertEqual(kr_res["direction"], "BEARISH")
        self.assertLess(kr_res["score"], -0.2)
        self.assertEqual(kr_res["stated_position"], "SHORT")

        # Lexicon benchmark agreement test
        bench = self.sentiment_analyzer.evaluate_benchmark_agreement()
        self.assertTrue(bench["benchmark_passed"])
        self.assertGreaterEqual(bench["agreement_percentage"], 85.0)

    def test_apify_item_normalization_across_sources(self):
        scraper = ApifySocialScraper(allow_simulation=True)
        ticker_cfg = EXCHANGE_CONFIGS["HKEX"].tickers["0700.HK"]

        # Sample LIHKG item
        lihkg_item = {
            "title": "700 今日大成交破位！",
            "url": "https://lihkg.com/thread/123456",
            "replies": 35,
            "timestamp": "2026-11-18T10:00:00Z"
        }
        norm_lihkg = scraper._normalize_apify_item(lihkg_item, ticker_cfg, "LIHKG", data_source="live")
        self.assertIsNotNone(norm_lihkg)
        self.assertEqual(norm_lihkg["ticker"], "0700.HK")
        self.assertEqual(norm_lihkg["source"], "LIHKG")
        self.assertEqual(norm_lihkg["views"], 35)
        self.assertEqual(norm_lihkg["data_source"], "live")

        # Sample Naver Finance item
        naver_item = {
            "title": "삼전 외인 순매수 폭발 떡상 가자",
            "views": 250,
            "dateStr": "2026.11.18 14:30"
        }
        krx_cfg = EXCHANGE_CONFIGS["KRX"].tickers["005930.KS"]
        norm_naver = scraper._normalize_apify_item(naver_item, krx_cfg, "NAVER_FINANCE", data_source="live")
        self.assertIsNotNone(norm_naver)
        self.assertEqual(norm_naver["views"], 250)
        self.assertEqual(norm_naver["exchange"], "KRX")

    def test_simulation_disabled_by_default(self):
        # Must raise RuntimeError if API client is absent and simulation is not explicitly permitted
        strict_scraper = ApifySocialScraper(api_token="", allow_simulation=False)
        strict_scraper.client = None
        ticker_cfg = EXCHANGE_CONFIGS["HKEX"].tickers["0700.HK"]

        # If cache does not exist, it must raise RuntimeError
        non_existent_cfg = EXCHANGE_CONFIGS["HKEX"].tickers["1810.HK"]
        # Clear any potential mock cache for test isolation
        test_cache = Path("data/cache/social_TEST_10d.parquet")
        if test_cache.exists():
            test_cache.unlink()

        # Run with allow_simulation=True and assert data_source="simulated"
        sim_scraper = ApifySocialScraper(api_token="", allow_simulation=True)
        sim_posts = sim_scraper.scrape_ticker_socials(ticker_cfg, ["LIHKG"], lookback_days=2, max_posts_per_source=5)
        self.assertTrue(all(p["data_source"] == "simulated" for p in sim_posts))

    def test_strict_alignment_and_no_lookahead_leakage(self):
        # Create 10 hourly bars
        base_time = datetime(2026, 11, 10, 2, 0, tzinfo=timezone.utc)
        dates = [base_time + timedelta(hours=i) for i in range(10)]
        prices = [100.0, 101.0, 100.5, 102.0, 103.0, 102.5, 104.0, 105.0, 104.5, 106.0]

        df = pd.DataFrame({
            "open": prices,
            "high": [p + 0.5 for p in prices],
            "low": [p - 0.5 for p in prices],
            "close": prices,
            "volume": [1_000_000] * 10
        }, index=dates)

        # Post 1: occurs at 02:30 UTC -> belongs to bar 0 (close 03:00 UTC)
        # Post 2: occurs at 05:45 UTC -> belongs to bar 3 (close 06:00 UTC)
        # Post 3: occurs in future at 15:00 UTC -> MUST NOT appear in bar 0-9
        posts = [
            {"timestamp_utc": (base_time + timedelta(minutes=30)).isoformat(), "text": "Bar 0 event", "likes": 5, "author_followers": 100},
            {"timestamp_utc": (base_time + timedelta(hours=3, minutes=45)).isoformat(), "text": "Bar 3 event", "likes": 10, "author_followers": 200},
            {"timestamp_utc": (base_time + timedelta(hours=20)).isoformat(), "text": "Future event", "likes": 50, "author_followers": 1000}
        ]

        processor = DataAlignmentProcessor()
        aligned = processor.align_and_engineer_features("TEST", df, posts, [], bar_interval_minutes=60)

        # Bar 0 should capture Post 1 (social_mentions == 1)
        self.assertEqual(aligned["social_mentions"].iloc[0], 1)
        # Future post at hour 20 must NOT leak into any bar
        self.assertEqual(aligned["social_mentions"].sum(), 2)

    def test_validator_insufficient_data_handling(self):
        validator = OutOfSampleValidator(min_train=40, min_test=15)
        # Small DataFrame with only 10 rows
        tiny_X = pd.DataFrame(np.random.randn(10, 5))
        tiny_y = pd.Series(np.random.randn(10))

        def dummy_builder():
            from sklearn.linear_model import Ridge
            return Ridge()

        res = validator.evaluate_walk_forward(dummy_builder, tiny_X, tiny_y)
        self.assertEqual(res["status"], "insufficient_data")
        self.assertIsNone(res["directional_accuracy"])
        self.assertIsNone(res["mean_rmse"])
        self.assertIsNone(res["mean_mae"])
        self.assertIsNone(res["sharpe_ratio"])

    def test_regression_no_hardcoded_metrics_in_codebase(self):
        # Assert that legacy hardcoded numbers (0.625, 1.45, 0.64, 1.52, 1.2) are NOT in model outputs
        predictor = PricePredictionModel(min_train=100, min_test=30)
        tiny_df = pd.DataFrame({
            "close": [100.0, 101.0, 102.0],
            "weighted_sentiment": [0.2, 0.3, 0.1]
        })
        heuristic_res = predictor._heuristic_prediction("TEST", tiny_df)

        self.assertEqual(heuristic_res["symbol"], "TEST")
        self.assertEqual(heuristic_res["model"], "heuristic")
        self.assertIsNone(heuristic_res["oos_validation"]["directional_accuracy"])
        self.assertIsNone(heuristic_res["oos_validation"]["sharpe_ratio"])

        # Check raw numbers
        oos = heuristic_res["oos_validation"]
        for bad_metric in [0.625, 1.45, 0.64, 1.52, 1.2]:
            self.assertNotEqual(oos.get("directional_accuracy"), bad_metric)
            self.assertNotEqual(oos.get("sharpe_ratio"), bad_metric)


if __name__ == "__main__":
    unittest.main()
