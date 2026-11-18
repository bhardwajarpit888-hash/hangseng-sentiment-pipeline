import logging
from typing import Dict, Optional
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
import yfinance as yf
from config.settings import EXCHANGE_CONFIGS, settings, CACHE_DIR

logger = logging.getLogger(__name__)


class MarketDataIngestor:
    """
    Ingests and normalizes OHLCV market bars from HKEX and KRX via yfinance.
    Normalizes all timestamps to UTC. Handles exchange market hours:
    HKEX: 01:30 - 08:00 UTC
    KRX:  00:00 - 06:30 UTC
    """

    def __init__(self, lookback_days: int = 10, interval: str = "1h", allow_simulation: bool = False):
        self.lookback_days = lookback_days
        self.interval = interval
        self.allow_simulation = allow_simulation
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def fetch_ticker_data(
        self,
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> pd.DataFrame:
        if end_date is None:
            end_date = datetime.now(timezone.utc)
        if start_date is None:
            start_date = end_date - timedelta(days=max(25, self.lookback_days + 15))

        cache_file = CACHE_DIR / f"price_{symbol.replace('.', '_')}_{self.interval}.parquet"

        try:
            logger.info(f"Fetching market bars for {symbol} (interval={self.interval}, lookback={self.lookback_days}d)...")
            ticker = yf.Ticker(symbol)
            # Use start/end for precise windowing
            df = ticker.history(
                start=start_date.strftime("%Y-%m-%d"),
                end=(end_date + timedelta(days=1)).strftime("%Y-%m-%d"),
                interval=self.interval,
                auto_adjust=True
            )

            if df.empty or len(df) < 5:
                logger.warning(f"yfinance returned insufficient data for {symbol} ({len(df)} rows).")
                if not self.allow_simulation:
                    raise RuntimeError(
                        f"Live market data fetch failed for {symbol} and simulation is disabled. "
                        "Pass --allow-simulation to permit synthetic baseline fallback."
                    )
                logger.warning(f"⚠️ [SIMULATION WARNING] Generating synthetic fallback market bars for {symbol}.")
                df = self._generate_fallback_bars(symbol, start_date, end_date)
                data_source = "simulated"
            else:
                data_source = "live"

            df = self._normalize_dataframe(df, symbol, data_source=data_source)
            df.to_parquet(cache_file)
            return df

        except Exception as e:
            if not self.allow_simulation:
                if cache_file.exists():
                    logger.info(f"Using cached market data for {symbol} from {cache_file}")
                    return pd.read_parquet(cache_file)
                raise RuntimeError(
                    f"Market data failure for {symbol}: {e}. Simulation disabled (--allow-simulation not set)."
                ) from e

            logger.warning(f"⚠️ [SIMULATION WARNING] Market data fetch failed ({e}). Generating fallback bars.")
            df = self._generate_fallback_bars(symbol, start_date, end_date)
            df = self._normalize_dataframe(df, symbol, data_source="simulated")
            return df

    def _normalize_dataframe(self, df: pd.DataFrame, symbol: str, data_source: str = "live") -> pd.DataFrame:
        df = df.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Standardize timezone to UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        # Standardize column naming
        df.columns = [col.lower().replace(" ", "_") for col in df.columns]
        required_cols = ["open", "high", "low", "close", "volume"]
        for col in required_cols:
            if col not in df.columns:
                df[col] = 0.0

        df = df[required_cols].copy()
        df = df.dropna(subset=["close"])
        df = df[df["close"] > 0]

        # Exchange session filtering
        is_hk = symbol.endswith(".HK")
        if is_hk:
            # HKEX: 01:30 - 08:00 UTC
            mask = (df.index.time >= datetime.strptime("01:30", "%H:%M").time()) & \
                   (df.index.time <= datetime.strptime("08:00", "%H:%M").time())
            if mask.sum() > 10:
                df = df[mask]
        else:
            # KRX: 00:00 - 06:30 UTC
            mask = (df.index.time >= datetime.strptime("00:00", "%H:%M").time()) & \
                   (df.index.time <= datetime.strptime("06:30", "%H:%M").time())
            if mask.sum() > 10:
                df = df[mask]

        df["symbol"] = symbol
        df["data_source"] = data_source
        df["return"] = df["close"].pct_change()
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

        # Forward target return for n+1
        df["target_return_n1"] = df["return"].shift(-1)
        df["target_close_n1"] = df["close"].shift(-1)
        df["target_direction_n1"] = (df["target_return_n1"] > 0).astype(int)

        return df

    def _generate_fallback_bars(self, symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
        """Synthetic baseline generator invoked ONLY when --allow-simulation is active."""
        if not self.allow_simulation:
            raise RuntimeError("Simulation disabled. Refusing to generate synthetic market bars.")

        freq = "1h" if self.interval in ("1h", "60m") else "1d"
        dates = pd.date_range(start=start, end=end, freq=freq, tz="UTC")
        if len(dates) == 0:
            dates = pd.date_range(end=datetime.now(timezone.utc), periods=80, freq=freq, tz="UTC")

        # Derive seed deterministically from symbol hash without hardcoded directional bias
        np.random.seed(abs(hash(symbol)) % (2**31 - 1))
        p0 = 100.0
        daily_ret = np.random.normal(0.0, 0.008, size=len(dates))
        close_series = p0 * np.cumprod(1 + daily_ret)

        records = []
        for d, c in zip(dates, close_series):
            noise = np.random.uniform(0.002, 0.008)
            h = c * (1 + noise)
            l = c * (1 - noise)
            o = np.random.uniform(l, h)
            v = int(np.random.uniform(500_000, 5_000_000))
            records.append({"date": d, "open": o, "high": h, "low": l, "close": c, "volume": v})

        df = pd.DataFrame(records).set_index("date")
        return df

    def fetch_all_exchanges(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        data = {}
        for ex_code, ex_cfg in EXCHANGE_CONFIGS.items():
            data[ex_code] = {}
            for symbol in ex_cfg.tickers.keys():
                df = self.fetch_ticker_data(symbol)
                data[ex_code][symbol] = df
        return data
