import logging
import os
import random
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
import pandas as pd

from config.settings import settings, TickerConfig, CACHE_DIR

logger = logging.getLogger(__name__)


class ExaNewsIngestor:
    """
    Ingests financial and market news using Exa.com neural search.
    Tags all output records with `data_source` ("live" or "simulated").
    Simulation is OFF by default and will raise RuntimeError unless --allow-simulation is given.
    """

    def __init__(self, api_key: Optional[str] = None, allow_simulation: bool = False):
        self.api_key = api_key or settings.exa_api_key
        self.allow_simulation = allow_simulation
        self.client = None

        if settings.has_exa_key:
            try:
                from exa_py import Exa
                self.client = Exa(api_key=self.api_key)
                logger.info("Exa client initialized with active API key.")
            except Exception as e:
                logger.error(f"Failed to initialize Exa client: {e}")

    def fetch_recent_news(
        self,
        ticker_cfg: TickerConfig,
        lookback_days: int = 10,
        max_articles: int = 10
    ) -> List[Dict[str, Any]]:
        cache_path = CACHE_DIR / f"news_{ticker_cfg.symbol.replace('.', '_')}_{lookback_days}d.parquet"

        articles = []
        if self.client:
            try:
                articles = self._execute_exa_search(ticker_cfg, lookback_days, max_articles)
                logger.info(f"Retrieved {len(articles)} LIVE news articles from Exa for {ticker_cfg.symbol}.")
            except Exception as e:
                logger.error(f"Exa search failed for {ticker_cfg.symbol}: {e}")
                if not self.allow_simulation:
                    if cache_path.exists():
                        logger.info(f"Loading cached news articles for {ticker_cfg.symbol}")
                        return pd.read_parquet(cache_path).to_dict("records")
                    raise RuntimeError(
                        f"Live Exa news search failed for {ticker_cfg.symbol} ({e}) and simulation is disabled. "
                        "Pass --allow-simulation to allow synthetic fallback."
                    ) from e

        if not articles:
            if not self.allow_simulation:
                if cache_path.exists():
                    logger.info(f"Loading cached news articles for {ticker_cfg.symbol}")
                    return pd.read_parquet(cache_path).to_dict("records")
                raise RuntimeError(
                    f"No Exa API key or live news returned for {ticker_cfg.symbol}. "
                    "Simulation is OFF by default. To run with synthetic fallback, provide --allow-simulation."
                )

            logger.warning(f"⚠️ [SIMULATION WARNING] Using synthetic news articles for {ticker_cfg.symbol}.")
            articles = self._generate_simulated_news(ticker_cfg, lookback_days, max_articles)

        df = pd.DataFrame(articles)
        if not df.empty:
            df.to_parquet(cache_path)

        return articles

    def _execute_exa_search(
        self,
        ticker_cfg: TickerConfig,
        lookback_days: int,
        limit: int
    ) -> List[Dict[str, Any]]:
        start_date = (datetime.now(timezone.utc) - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        query = f"{ticker_cfg.name} {ticker_cfg.symbol} financial performance earnings stock analyst target price"

        domain_filter = [
            "bloomberg.com",
            "reuters.com",
            "scmp.com",
            "hkej.com",
            "koreatimes.co.kr",
            "kedglobal.com",
            "investing.com",
            "cnbc.com",
            "marketwatch.com"
        ]

        logger.info(f"Querying Exa API for {ticker_cfg.symbol} (start_published_date={start_date})...")
        response = self.client.search_and_contents(
            query=query,
            type="neural",
            use_autoprompt=True,
            num_results=limit,
            start_published_date=start_date,
            include_domains=domain_filter,
            text={"max_characters": 1200}
        )

        articles = []
        for res in response.results:
            pub_date = res.published_date or datetime.now(timezone.utc).isoformat()
            articles.append({
                "ticker": ticker_cfg.symbol,
                "asset_name": ticker_cfg.name,
                "exchange": ticker_cfg.exchange,
                "source": "EXA_NEWS",
                "data_source": "live",
                "title": res.title or "",
                "url": res.url,
                "publisher": self._extract_domain(res.url),
                "timestamp_utc": pd.to_datetime(pub_date, utc=True).isoformat(),
                "text": res.text or "",
                "score": float(getattr(res, "score", 0.0) or 0.0)
            })

        return articles

    def _extract_domain(self, url: str) -> str:
        try:
            from urllib.parse import urlparse
            return urlparse(url).netloc.replace("www.", "")
        except Exception:
            return "financial_news"

    def _generate_simulated_news(
        self,
        ticker_cfg: TickerConfig,
        lookback_days: int,
        count: int
    ) -> List[Dict[str, Any]]:
        now = datetime.now(timezone.utc)
        articles = []

        is_hk = (ticker_cfg.exchange == "HKEX")
        publishers = [
            "reuters.com", "bloomberg.com", "scmp.com", "investing.com"
        ] if is_hk else [
            "reuters.com", "bloomberg.com", "kedglobal.com", "koreatimes.co.kr"
        ]

        # Neutral sample headlines
        generic_headlines = [
            (f"{ticker_cfg.name} Releases Operational and Quarterly Updates", 0.1),
            (f"Analysts Review {ticker_cfg.symbol} Valuation and Market Stance", 0.0),
            (f"Sector Trends and Macro Headwinds Analyzed for {ticker_cfg.name}", -0.05),
            (f"{ticker_cfg.name} Market Activity in Line with Benchmark Index", 0.0)
        ]

        for i in range(min(count, len(generic_headlines))):
            title, score = generic_headlines[i]
            delta_days = random.uniform(0.5, lookback_days)
            art_time = now - timedelta(days=delta_days)
            publisher = random.choice(publishers)

            articles.append({
                "ticker": ticker_cfg.symbol,
                "asset_name": ticker_cfg.name,
                "exchange": ticker_cfg.exchange,
                "source": "EXA_NEWS",
                "data_source": "simulated",
                "title": title,
                "url": f"https://{publisher}/article/{ticker_cfg.symbol.lower().replace('.', '')}-{random.randint(100000, 999999)}",
                "publisher": publisher,
                "timestamp_utc": art_time.isoformat(),
                "text": f"{title}. Research analysts note sector conditions and macroeconomic developments.",
                "score": float(score)
            })

        articles.sort(key=lambda x: x["timestamp_utc"])
        return articles
