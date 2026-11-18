import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from dotenv import load_dotenv

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
    raw_code: str  # e.g. "00700", "005930"


@dataclass
class ExchangeConfig:
    code: str
    name: str
    timezone: str
    session_utc_open: str   # HH:MM
    session_utc_close: str  # HH:MM
    social_sources: List[str]
    tickers: Dict[str, TickerConfig]


# Source-specific Apify Actor Configurations and Start URL patterns
APIFY_SOURCE_SPECS: Dict[str, Dict[str, Any]] = {
    "LIHKG": {
        "actor_id": "apify/web-scraper",
        "url_template": "https://lihkg.com/category/5?page=1",
        "page_function": """
            async function pageFunction(context) {
                const { $ } = context;
                const posts = [];
                $('div[data-testid="post-row"], div._2cNsnoTdbn54Pq83Empty, ._2cNsnoTdbn54Pq83Empty').each((i, el) => {
                    const title = $(el).find('span[property="headline"], ._20jN3DqE94Xz97Wk_w7b2e').text().trim();
                    const url = $(el).find('a').attr('href');
                    const replies = parseInt($(el).find('span[title="replies"], .replies').text()) || 0;
                    if (title) posts.push({ title, text: title, url, replies, timestamp: new Date().toISOString() });
                });
                return posts;
            }
        """
    },
    "FUTU_MOOMOO": {
        "actor_id": "apify/puppeteer-scraper",
        "url_template": "https://www.moomoo.com/community/stock/{raw_code}",
        "page_function": """
            async function pageFunction(context) {
                const { page } = context;
                await page.waitForSelector('.post-card, .moment-item', { timeout: 8000 }).catch(() => {});
                return await page.evaluate(() => {
                    return Array.from(document.querySelectorAll('.post-card, .moment-item')).map(el => ({
                        text: el.innerText,
                        likes: parseInt(el.querySelector('.like-count')?.innerText || '0'),
                        timestamp: new Date().toISOString()
                    }));
                });
            }
        """
    },
    "NAVER_FINANCE": {
        "actor_id": "apify/cheerio-scraper",
        "url_template": "https://finance.naver.com/item/board.naver?code={raw_code}",
        "page_function": """
            async function pageFunction(context) {
                const { $ } = context;
                const posts = [];
                $('table.type2 tbody tr').each((i, el) => {
                    const titleEl = $(el).find('td.title a');
                    const title = titleEl.text().trim();
                    const views = parseInt($(el).find('td:nth-child(4)').text()) || 0;
                    const dateStr = $(el).find('td:nth-child(2)').text().trim();
                    if (title) posts.push({ title, text: title, views, dateStr, timestamp: new Date().toISOString() });
                });
                return posts;
            }
        """
    },
    "DAUM_FINANCE": {
        "actor_id": "apify/cheerio-scraper",
        "url_template": "https://finance.daum.net/quotes/A{raw_code}#talks",
        "page_function": """
            async function pageFunction(context) {
                const { $ } = context;
                const posts = [];
                $('div.box_talk li, table.tbl_talk tbody tr').each((i, el) => {
                    const text = $(el).find('a.txt_tit, .title').text().trim();
                    if (text) posts.push({ title: text, text, timestamp: new Date().toISOString() });
                });
                return posts;
            }
        """
    },
    "TELEGRAM": {
        "actor_id": "vdrmmr/telegram-scraper",
        "url_template": "https://t.me/s/{channel}",
        "page_function": None
    }
}


EXCHANGE_CONFIGS: Dict[str, ExchangeConfig] = {
    "HKEX": ExchangeConfig(
        code="HKEX",
        name="Hong Kong Stock Exchange (Hang Seng)",
        timezone="Asia/Hong_Kong",
        session_utc_open="01:30",
        session_utc_close="08:00",
        social_sources=["LIHKG", "FUTU_MOOMOO"],
        tickers={
            "0700.HK": TickerConfig(
                symbol="0700.HK",
                name="Tencent Holdings Ltd.",
                exchange="HKEX",
                currency="HKD",
                keywords=["騰訊", "腾讯", "Tencent", "700", "企鵝"],
                cashtags=["#0700", "$0700.HK", "#騰訊控股"],
                sector="Technology / Internet",
                raw_code="00700"
            ),
            "9988.HK": TickerConfig(
                symbol="9988.HK",
                name="Alibaba Group Holding",
                exchange="HKEX",
                currency="HKD",
                keywords=["阿里巴巴", "阿里", "Alibaba", "9988", "淘寶"],
                cashtags=["#9988", "$9988.HK", "#阿里巴巴"],
                sector="E-commerce / Cloud",
                raw_code="09988"
            ),
            "3690.HK": TickerConfig(
                symbol="3690.HK",
                name="Meituan",
                exchange="HKEX",
                currency="HKD",
                keywords=["美團", "美团", "Meituan", "3690", "外賣"],
                cashtags=["#3690", "$3690.HK", "#美團點評"],
                sector="Consumer Services / Tech",
                raw_code="03690"
            ),
            "1810.HK": TickerConfig(
                symbol="1810.HK",
                name="Xiaomi Corporation",
                exchange="HKEX",
                currency="HKD",
                keywords=["小米", "Xiaomi", "1810", "雷軍", "SU7"],
                cashtags=["#1810", "$1810.HK", "#小米集團"],
                sector="Hardware / EV",
                raw_code="01810"
            ),
            "0005.HK": TickerConfig(
                symbol="0005.HK",
                name="HSBC Holdings plc",
                exchange="HKEX",
                currency="HKD",
                keywords=["滙豐", "汇丰", "HSBC", "0005", "大笨象"],
                cashtags=["#0005", "$0005.HK", "#滙豐控股"],
                sector="Financial Services / Banking",
                raw_code="00005"
            ),
        }
    ),
    "KRX": ExchangeConfig(
        code="KRX",
        name="Korea Exchange (KOSPI)",
        timezone="Asia/Seoul",
        session_utc_open="00:00",
        session_utc_close="06:30",
        social_sources=["NAVER_FINANCE", "DAUM_FINANCE"],
        tickers={
            "005930.KS": TickerConfig(
                symbol="005930.KS",
                name="Samsung Electronics Co.",
                exchange="KRX",
                currency="KRW",
                keywords=["삼성전자", "삼전", "Samsung", "005930", "반도체"],
                cashtags=["#005930", "$005930.KS", "#삼성전자"],
                sector="Semiconductors / Tech",
                raw_code="005930"
            ),
            "000660.KS": TickerConfig(
                symbol="000660.KS",
                name="SK Hynix Inc.",
                exchange="KRX",
                currency="KRW",
                keywords=["SK하이닉스", "하이닉스", "Hynix", "000660", "HBM"],
                cashtags=["#000660", "$000660.KS", "#SK하이닉스"],
                sector="Semiconductors / Memory",
                raw_code="000660"
            ),
            "035420.KS": TickerConfig(
                symbol="035420.KS",
                name="NAVER Corporation",
                exchange="KRX",
                currency="KRW",
                keywords=["네이버", "NAVER", "035420", "라인", "치지직"],
                cashtags=["#035420", "$035420.KS", "#네이버"],
                sector="Internet / AI",
                raw_code="035420"
            ),
            "035720.KS": TickerConfig(
                symbol="035720.KS",
                name="Kakao Corp.",
                exchange="KRX",
                currency="KRW",
                keywords=["카카오", "Kakao", "035720", "카카오톡", "카뱅"],
                cashtags=["#035720", "$035720.KS", "#카카오"],
                sector="Social Media / Fintech",
                raw_code="035720"
            ),
            "005380.KS": TickerConfig(
                symbol="005380.KS",
                name="Hyundai Motor Co.",
                exchange="KRX",
                currency="KRW",
                keywords=["현대자동차", "현대차", "Hyundai", "005380", "아이오닉"],
                cashtags=["#005380", "$005380.KS", "#현대차"],
                sector="Automotive / EV",
                raw_code="005380"
            ),
        }
    )
}


@dataclass
class AppSettings:
    apify_api_token: str = field(default_factory=lambda: os.getenv("APIFY_API_TOKEN", ""))
    exa_api_key: str = field(default_factory=lambda: os.getenv("EXA_API_KEY", ""))
    lookback_days: int = 10
    interval: str = "1h"
    cv_splits: int = 4
    min_train_rows: int = 40
    min_test_rows: int = 15
    random_seed: int = 42
    allow_simulation: bool = False  # STRICT: simulation is OFF by default

    @property
    def has_apify_key(self) -> bool:
        return bool(self.apify_api_token and self.apify_api_token.strip() and not self.apify_api_token.startswith("your_"))

    @property
    def has_exa_key(self) -> bool:
        return bool(self.exa_api_key and self.exa_api_key.strip() and not self.exa_api_key.startswith("your_"))


settings = AppSettings()
