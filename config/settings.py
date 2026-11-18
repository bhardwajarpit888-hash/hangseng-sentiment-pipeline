import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Base paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
STORAGE_DIR = DATA_DIR / "storage"
REPORTS_DIR = PROJECT_ROOT / "reports"

load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class TickerConfig:
    symbol: str
    name: str
    exchange: str
    currency: str
    keywords: List[str]
    cashtags: List[str]
    sector: str


@dataclass
class ExchangeConfig:
    code: str
    name: str
    timezone: str
    social_sources: List[str]
    tickers: Dict[str, TickerConfig]


EXCHANGE_CONFIGS: Dict[str, ExchangeConfig] = {
    "HKEX": ExchangeConfig(
        code="HKEX",
        name="Hong Kong Stock Exchange (Hang Seng)",
        timezone="Asia/Hong_Kong",
        social_sources=["LIHKG", "FUTU_MOOMOO"],
        tickers={
            "0700.HK": TickerConfig(
                symbol="0700.HK",
                name="Tencent Holdings Ltd.",
                exchange="HKEX",
                currency="HKD",
                keywords=["騰訊", "腾讯", "Tencent", "700", "企鵝"],
                cashtags=["#0700", "$0700.HK", "#騰訊控股"],
                sector="Technology / Internet"
            ),
            "9988.HK": TickerConfig(
                symbol="9988.HK",
                name="Alibaba Group Holding",
                exchange="HKEX",
                currency="HKD",
                keywords=["阿里巴巴", "阿里", "Alibaba", "9988", "淘寶"],
                cashtags=["#9988", "$9988.HK", "#阿里巴巴"],
                sector="E-commerce / Cloud"
            ),
            "3690.HK": TickerConfig(
                symbol="3690.HK",
                name="Meituan",
                exchange="HKEX",
                currency="HKD",
                keywords=["美團", "美团", "Meituan", "3690", "外賣"],
                cashtags=["#3690", "$3690.HK", "#美團點評"],
                sector="Consumer Services / Tech"
            ),
            "1810.HK": TickerConfig(
                symbol="1810.HK",
                name="Xiaomi Corporation",
                exchange="HKEX",
                currency="HKD",
                keywords=["小米", "Xiaomi", "1810", "雷軍", "SU7"],
                cashtags=["#1810", "$1810.HK", "#小米集團"],
                sector="Hardware / EV"
            ),
            "0005.HK": TickerConfig(
                symbol="0005.HK",
                name="HSBC Holdings plc",
                exchange="HKEX",
                currency="HKD",
                keywords=["滙豐", "汇丰", "HSBC", "0005", "大笨象"],
                cashtags=["#0005", "$0005.HK", "#滙豐控股"],
                sector="Financial Services / Banking"
            ),
        }
    ),
    "KRX": ExchangeConfig(
        code="KRX",
        name="Korea Exchange (KOSPI)",
        timezone="Asia/Seoul",
        social_sources=["NAVER_FINANCE", "DAUM_FINANCE"],
        tickers={
            "005930.KS": TickerConfig(
                symbol="005930.KS",
                name="Samsung Electronics Co.",
                exchange="KRX",
                currency="KRW",
                keywords=["삼성전자", "삼전", "Samsung", "005930", "반도체"],
                cashtags=["#005930", "$005930.KS", "#삼성전자"],
                sector="Semiconductors / Tech"
            ),
            "000660.KS": TickerConfig(
                symbol="000660.KS",
                name="SK Hynix Inc.",
                exchange="KRX",
                currency="KRW",
                keywords=["SK하이닉스", "하이닉스", "Hynix", "000660", "HBM"],
                cashtags=["#000660", "$000660.KS", "#SK하이닉스"],
                sector="Semiconductors / Memory"
            ),
            "035420.KS": TickerConfig(
                symbol="035420.KS",
                name="NAVER Corporation",
                exchange="KRX",
                currency="KRW",
                keywords=["네이버", "NAVER", "035420", "라인", "치지직"],
                cashtags=["#035420", "$035420.KS", "#네이버"],
                sector="Internet / AI"
            ),
            "035720.KS": TickerConfig(
                symbol="035720.KS",
                name="Kakao Corp.",
                exchange="KRX",
                currency="KRW",
                keywords=["카카오", "Kakao", "035720", "카카오톡", "카뱅"],
                cashtags=["#035720", "$035720.KS", "#카카오"],
                sector="Social Media / Fintech"
            ),
            "005380.KS": TickerConfig(
                symbol="005380.KS",
                name="Hyundai Motor Co.",
                exchange="KRX",
                currency="KRW",
                keywords=["현대자동차", "현대차", "Hyundai", "005380", "아이오닉"],
                cashtags=["#005380", "$005380.KS", "#현대차"],
                sector="Automotive / EV"
            ),
        }
    )
}


@dataclass
class AppSettings:
    apify_api_token: str = field(default_factory=lambda: os.getenv("APIFY_API_TOKEN", ""))
    exa_api_key: str = field(default_factory=lambda: os.getenv("EXA_API_KEY", ""))
    lookback_days: int = 10
    interval: str = "1d"  # or 1h for intraday resolution
    cv_splits: int = 4
    random_seed: int = 42
    use_mock_fallback: bool = True  # Allows offline/demo runs without crashing on missing keys

    @property
    def has_apify_key(self) -> bool:
        return bool(self.apify_api_token and self.apify_api_token.strip() and not self.apify_api_token.startswith("your_"))

    @property
    def has_exa_key(self) -> bool:
        return bool(self.exa_api_key and self.exa_api_key.strip() and not self.exa_api_key.startswith("your_"))


settings = AppSettings()
