import logging
import os
import random
import time
from datetime import datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
import pandas as pd

from config.settings import settings, EXCHANGE_CONFIGS, TickerConfig, CACHE_DIR, APIFY_SOURCE_SPECS

logger = logging.getLogger(__name__)


class ApifySocialScraper:
    """
    Scrapes retail investor social platforms for Hong Kong (LIHKG, Futu/Moomoo)
    and South Korea (Naver Finance 종목토론실, Daum Finance) via Apify actors.
    Strictly tags every record with `data_source` ("live" or "simulated").
    Simulation is OFF by default.
    """

    def __init__(self, api_token: Optional[str] = None, allow_simulation: bool = False):
        self.api_token = api_token or settings.apify_api_token
        self.allow_simulation = allow_simulation
        self.client = None

        if settings.has_apify_key:
            try:
                from apify_client import ApifyClient
                self.client = ApifyClient(self.api_token)
                logger.info("ApifyClient initialized with user token.")
            except Exception as e:
                logger.error(f"Failed to initialize ApifyClient: {e}")

    def scrape_ticker_socials(
        self,
        ticker_cfg: TickerConfig,
        sources: List[str],
        lookback_days: int = 10,
        max_posts_per_source: int = 40
    ) -> List[Dict[str, Any]]:
        cache_path = CACHE_DIR / f"social_{ticker_cfg.symbol.replace('.', '_')}_{lookback_days}d.parquet"

        posts = []
        if self.client:
            try:
                posts = self._run_apify_actors(ticker_cfg, sources, lookback_days, max_posts_per_source)
                logger.info(f"Retrieved {len(posts)} LIVE social posts from Apify for {ticker_cfg.symbol}.")
            except Exception as e:
                logger.error(f"Apify live scraping execution failed for {ticker_cfg.symbol}: {e}")
                if not self.allow_simulation:
                    if cache_path.exists():
                        logger.info(f"Loading cached social records for {ticker_cfg.symbol}")
                        return pd.read_parquet(cache_path).to_dict("records")
                    raise RuntimeError(
                        f"Live Apify scraping failed for {ticker_cfg.symbol} ({e}) and simulation is disabled. "
                        "Pass --allow-simulation to allow synthetic fallback."
                    ) from e

        if not posts:
            if not self.allow_simulation:
                if cache_path.exists():
                    logger.info(f"Loading cached social records for {ticker_cfg.symbol}")
                    return pd.read_parquet(cache_path).to_dict("records")
                raise RuntimeError(
                    f"No Apify API key or live posts available for {ticker_cfg.symbol}. "
                    "Simulation is OFF by default. To run with synthetic fallback, provide --allow-simulation."
                )

            logger.warning(f"⚠️ [SIMULATION WARNING] Using unbiased synthetic social stream for {ticker_cfg.symbol}.")
            posts = self._generate_simulated_social_stream(ticker_cfg, sources, lookback_days, max_posts_per_source)

        df = pd.DataFrame(posts)
        if not df.empty:
            df.to_parquet(cache_path)

        return posts

    def _run_apify_actors(
        self,
        ticker_cfg: TickerConfig,
        sources: List[str],
        lookback_days: int,
        limit: int
    ) -> List[Dict[str, Any]]:
        results = []
        max_retries = 2

        for source in sources:
            spec = APIFY_SOURCE_SPECS.get(source)
            if not spec:
                logger.warning(f"No Apify spec defined for source {source}")
                continue

            actor_id = spec["actor_id"]
            url_template = spec["url_template"]
            start_url = url_template.format(
                raw_code=ticker_cfg.raw_code,
                channel=ticker_cfg.symbol.lower().replace(".", "")
            )

            # Build valid Apify actor input schema
            run_input: Dict[str, Any] = {
                "startUrls": [{"url": start_url}],
                "maxRequestsPerCrawl": max(5, limit // 2),
            }
            if spec.get("page_function"):
                run_input["pageFunction"] = spec["page_function"]

            # Retry with exponential backoff
            actor_success = False
            for attempt in range(1, max_retries + 1):
                try:
                    logger.info(f"Calling Apify Actor '{actor_id}' for {ticker_cfg.symbol} on {source} (attempt {attempt})...")
                    run = self.client.actor(actor_id).call(run_input=run_input, timeout_secs=75)
                    dataset = self.client.dataset(run["defaultDatasetId"])

                    for item in dataset.iterate_items():
                        parsed = self._normalize_apify_item(item, ticker_cfg, source, data_source="live")
                        if parsed:
                            results.append(parsed)
                    actor_success = True
                    break
                except Exception as err:
                    logger.warning(f"Apify call attempt {attempt} failed for {source}: {err}")
                    if attempt < max_retries:
                        time.sleep(2 ** attempt)

            if not actor_success:
                logger.error(f"All {max_retries} live Apify attempts failed for {ticker_cfg.symbol} on {source}.")

        return results

    def _normalize_apify_item(
        self,
        item: Dict[str, Any],
        ticker_cfg: TickerConfig,
        source: str,
        data_source: str = "live"
    ) -> Optional[Dict[str, Any]]:
        """Normalizes source-specific Apify outputs into standardized records."""
        try:
            # Handle nested or flat payloads
            raw_time = item.get("timestamp") or item.get("createdAt") or item.get("date") or item.get("dateStr")
            if raw_time:
                ts = pd.to_datetime(raw_time, utc=True)
            else:
                ts = datetime.now(timezone.utc)

            text = str(item.get("text") or item.get("title") or item.get("content") or "").strip()
            if not text:
                return None

            likes = int(item.get("likes") or item.get("upvotes") or 0)
            reposts = int(item.get("reposts") or item.get("shares") or 0)
            views = int(item.get("views") or item.get("replies") or item.get("comments") or 0)
            followers = int(item.get("author_followers") or item.get("userScore") or 100)

            return {
                "ticker": ticker_cfg.symbol,
                "asset_name": ticker_cfg.name,
                "exchange": ticker_cfg.exchange,
                "source": source,
                "data_source": data_source,
                "timestamp_utc": ts.isoformat(),
                "text": text,
                "likes": max(0, likes),
                "reposts": max(0, reposts),
                "views": max(0, views),
                "author_followers": max(0, followers),
            }
        except Exception as e:
            logger.debug(f"Error normalizing Apify item: {e}")
            return None

    def _generate_simulated_social_stream(
        self,
        ticker_cfg: TickerConfig,
        sources: List[str],
        lookback_days: int,
        count_per_source: int
    ) -> List[Dict[str, Any]]:
        """
        Generates simulated social posts strictly tagged data_source='simulated'.
        NO bias_map: sentiment is generated uniformly at random to prevent circularity.
        """
        now = datetime.now(timezone.utc)
        posts = []

        # Standard phrases
        is_hk = (ticker_cfg.exchange == "HKEX")
        bull_phrases = [
            f"{ticker_cfg.name} 業績超預期，今日大成交破位，買入等開車！",
            f"{ticker_cfg.keywords[0]} 突破阻力位，看多，長線持有！",
            f"南下資金大幅買入 {ticker_cfg.cashtags[0]}，多頭強勢。"
        ] if is_hk else [
            f"{ticker_cfg.name} 실적 서프라이즈! 외인 순매수 유입 롱 포지션 진입.",
            f"{ticker_cfg.keywords[0]} 저항선 돌파 성공, 거래량 동반 매수 추천.",
            f"{ticker_cfg.cashtags[0]} 기관 매수세 유입 중, 상승 모멘텀 기대."
        ]

        bear_phrases = [
            f"{ticker_cfg.name} 跌穿支持位，止蝕沽空，避險為上。",
            f"外資大幅沽出 {ticker_cfg.keywords[0]}，成交放量下跌，睇淡。",
            f"{ticker_cfg.cashtags[0]} 上方阻力沉重，減持空倉觀望。"
        ] if is_hk else [
            f"{ticker_cfg.name} 지지선 이탈... 물려서 손절 고민 중, 숏 관점.",
            f"외인들 {ticker_cfg.keywords[0]} 대량 매도폭탄, 하방 압력 지속.",
            f"{ticker_cfg.cashtags[0]} 매물대 두꺼움, 단기 하락세 주의."
        ]

        neut_phrases = [
            f"{ticker_cfg.name} 區間窄幅震盪，未有明確突破方向。",
            f"{ticker_cfg.keywords[0]} 觀望為主，等待宏觀數據公布。"
        ] if is_hk else [
            f"{ticker_cfg.name} 박스권 횡보 중, 방향성 확인 필요.",
            f"{ticker_cfg.keywords[0]} 변동성 축소 국면, 관망 추천."
        ]

        # STRICT: NO bias_map. Pure uniform distribution (33% bull, 33% bear, 34% neut)
        for source in sources:
            for _ in range(count_per_source):
                delta_seconds = random.randint(300, lookback_days * 86400)
                post_time = now - timedelta(seconds=delta_seconds)

                rnd = random.random()
                if rnd < 0.33:
                    text = random.choice(bull_phrases)
                elif rnd < 0.66:
                    text = random.choice(bear_phrases)
                else:
                    text = random.choice(neut_phrases)

                posts.append({
                    "ticker": ticker_cfg.symbol,
                    "asset_name": ticker_cfg.name,
                    "exchange": ticker_cfg.exchange,
                    "source": source,
                    "data_source": "simulated",
                    "timestamp_utc": post_time.isoformat(),
                    "text": text,
                    "likes": random.randint(0, 50),
                    "reposts": random.randint(0, 15),
                    "views": random.randint(50, 1000),
                    "author_followers": random.randint(50, 2000),
                })

        posts.sort(key=lambda x: x["timestamp_utc"])
        return posts
