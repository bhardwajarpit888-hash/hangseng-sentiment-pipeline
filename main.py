import sys
import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

from config.settings import EXCHANGE_CONFIGS, settings, REPORTS_DIR
from data.market_data import MarketDataIngestor
from data.scrapers.apify_client import ApifySocialScraper
from data.scrapers.exa_news import ExaNewsIngestor
from data.processor import DataAlignmentProcessor
from technical.levels import PositionRecommendationEngine
from models.predictor import PricePredictionModel
from reporting.summary_table import SummaryTableGenerator
from reporting.visualizer import VisualReportGenerator
from reporting.prediction_logger import PredictionEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("main")


class PipelineOrchestrator:
    """
    Coordinates end-to-end ingestion, multi-source sentiment processing,
    feature alignment, out-of-sample validation, and n+1 price prediction.
    Enforces real-world data checks and flags simulated executions.
    """

    def __init__(
        self,
        lookback_days: int = 10,
        interval: str = "1h",
        allow_simulation: bool = False,
        use_transformer: bool = False
    ):
        self.lookback_days = lookback_days
        self.interval = interval
        self.allow_simulation = allow_simulation
        self.use_transformer = use_transformer

        self.market_ingestor = MarketDataIngestor(
            lookback_days=self.lookback_days,
            interval=self.interval,
            allow_simulation=self.allow_simulation
        )
        self.apify_scraper = ApifySocialScraper(allow_simulation=self.allow_simulation)
        self.exa_ingestor = ExaNewsIngestor(allow_simulation=self.allow_simulation)
        self.alignment_processor = DataAlignmentProcessor(use_transformer=self.use_transformer)
        self.position_engine = PositionRecommendationEngine()
        self.summary_table_gen = SummaryTableGenerator()
        self.visual_report_gen = VisualReportGenerator()
        self.prediction_evaluator = PredictionEvaluator()

    def run_pipeline(self, target_exchange: str = "ALL") -> Dict[str, Any]:
        logger.info(f"Starting sentiment & price prediction pipeline (Lookback: {self.lookback_days}d, Interval: {self.interval})")
        if self.allow_simulation:
            logger.warning("⚠️ [NOTICE] Simulation is ENABLED via --allow-simulation. Fallback synthetic feeds permitted if APIs are unreachable.")
        else:
            logger.info("✓ [STRICT MODE] Simulation is DISABLED. Live feeds required.")

        pipeline_results = []
        exchanges_to_process = (
            list(EXCHANGE_CONFIGS.keys()) if target_exchange == "ALL"
            else [target_exchange]
        )

        for ex_code in exchanges_to_process:
            ex_cfg = EXCHANGE_CONFIGS.get(ex_code)
            if not ex_cfg:
                logger.warning(f"Unknown exchange code: {ex_code}")
                continue

            logger.info(f"\n{'='*30}\nProcessing Exchange: {ex_cfg.name} ({ex_code})\n{'='*30}")

            for symbol, ticker_cfg in ex_cfg.tickers.items():
                logger.info(f"\n--- Processing Asset: {ticker_cfg.symbol} ({ticker_cfg.name}) ---")

                # 1. Fetch OHLCV Price Bars (UTC normalized)
                price_df = self.market_ingestor.fetch_ticker_data(symbol)
                if price_df.empty:
                    logger.warning(f"Skipping {symbol} due to missing price bars.")
                    continue

                # 2. Scrape Social Posts via Apify (minimum 2 sources)
                social_posts = self.apify_scraper.scrape_ticker_socials(
                    ticker_cfg=ticker_cfg,
                    sources=ex_cfg.social_sources,
                    lookback_days=self.lookback_days
                )
                logger.info(f"Loaded {len(social_posts)} social posts for {symbol}.")

                # 3. Ingest Recent News via Exa.com
                news_articles = self.exa_ingestor.fetch_recent_news(
                    ticker_cfg=ticker_cfg,
                    lookback_days=self.lookback_days
                )
                logger.info(f"Loaded {len(news_articles)} news articles for {symbol}.")

                # 4. Feature Alignment and Strict No-Leakage Preprocessing
                aligned_df = self.alignment_processor.align_and_engineer_features(
                    symbol=symbol,
                    price_df=price_df,
                    social_posts=social_posts,
                    news_articles=news_articles,
                    bar_interval_minutes=60 if self.interval in ("1h", "60m") else 1440
                )

                # 5. Extract Current State Sentiment Metrics
                current_sentiment = self.alignment_processor.weighting_engine.aggregate_sentiment_metrics(
                    social_posts,
                    news_articles
                )

                # 6. Technical Support/Resistance Levels & Position Recommendation
                recommendation = self.position_engine.generate_recommendation(
                    symbol=symbol,
                    price_df=price_df,
                    sentiment_metrics=current_sentiment
                )

                # 7. Predictive Model Training & n+1 Price Forecast
                predictor = PricePredictionModel()
                prediction = predictor.fit_and_predict_n1(
                    symbol=symbol,
                    features_df=aligned_df
                )

                # Track data source tag
                feed_sources = [
                    price_df.get("data_source", pd.Series(["live"])).iloc[0] if "data_source" in price_df.columns else "live",
                    social_posts[0].get("data_source", "live") if social_posts else "live",
                    news_articles[0].get("data_source", "live") if news_articles else "live"
                ]
                overall_source = "simulated" if "simulated" in feed_sources else "live"

                asset_result = {
                    "exchange": ex_code,
                    "symbol": symbol,
                    "name": ticker_cfg.name,
                    "currency": ticker_cfg.currency,
                    "sector": ticker_cfg.sector,
                    "current_price": float(price_df["close"].iloc[-1]),
                    "data_source": overall_source,
                    "sentiment": current_sentiment,
                    "recommendation": recommendation,
                    "prediction": prediction
                }
                pipeline_results.append(asset_result)

        # 8. Build Summaries & Visual Reports
        summary_df = self.summary_table_gen.build_summary_dataframe(pipeline_results)

        # Render Terminal Tables
        self.summary_table_gen.display_rich_tables(summary_df)

        # Generate HTML Dashboard
        html_report_path = self.visual_report_gen.generate_html_report(summary_df, pipeline_results)

        # Log Predictions for subsequent Out-Of-Sample realization tracking
        self.prediction_evaluator.log_predictions(pipeline_results)

        logger.info(f"\nPipeline Run Completed Successfully!")
        logger.info(f"Visual Report ready at: {html_report_path}")
        logger.info(f"Predictions logged to: {self.prediction_evaluator.log_path}")

        return {
            "summary_df": summary_df,
            "results": pipeline_results,
            "report_path": html_report_path
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Cross-Exchange Sentiment & n+1 Price Prediction Pipeline")
    parser.add_argument("--exchange", choices=["ALL", "HKEX", "KRX"], default="ALL", help="Target exchange (default: ALL)")
    parser.add_argument("--lookback", type=int, default=10, help="Lookback horizon in days (default: 10)")
    parser.add_argument("--interval", choices=["1h", "15m", "1d"], default="1h", help="Market bar interval (default: 1h)")
    parser.add_argument("--allow-simulation", action="store_true", help="Allow synthetic fallback if live API keys are missing/rate-limited")
    parser.add_argument("--use-transformer", action="store_true", help="Enable FinBERT transformer pipeline if installed")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate logged predictions against realized market prices")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.evaluate:
        evaluator = PredictionEvaluator()
        evaluator.evaluate_realized_predictions()
        sys.exit(0)

    orchestrator = PipelineOrchestrator(
        lookback_days=args.lookback,
        interval=args.interval,
        allow_simulation=args.allow_simulation,
        use_transformer=args.use_transformer
    )
    orchestrator.run_pipeline(target_exchange=args.exchange)
