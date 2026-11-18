import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


class MultilingualFinancialSentimentAnalyzer:
    """
    Multilingual financial sentiment engine tailored for Hang Seng (Cantonese / Traditional Chinese),
    KRX (Korean), and international financial English news.
    Supports rule-based domain lexicons, position extraction from retail discourse,
    price target extraction, and optional transformer backend.
    """

    def __init__(self, use_transformer: bool = False):
        self.vader = SentimentIntensityAnalyzer()
        self.use_transformer = use_transformer
        self.transformer_pipeline = None

        if self.use_transformer:
            self._init_transformer()

        self._init_financial_lexicons()
        self._init_position_patterns()

    def _init_transformer(self):
        try:
            from transformers import pipeline
            logger.info("Initializing FinBERT / multilingual transformer pipeline...")
            self.transformer_pipeline = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert"
            )
        except Exception as e:
            logger.warning(f"Transformer model unavailable ({e}). Falling back to multilingual lexicon + VADER.")
            self.transformer_pipeline = None

    def _init_financial_lexicons(self):
        # Hong Kong / Cantonese retail & financial terms
        self.hk_bullish_terms = {
            "破位": 0.8, "大成交": 0.6, "阻力位破": 0.85, "牛軍": 0.75, "建倉": 0.65,
            "開車": 0.7, "派息": 0.5, "超預期": 0.9, "長線買入": 0.8, "底背馳": 0.75,
            "金叉": 0.8, "反彈": 0.6, "掃入": 0.7, "買入": 0.75, "回購": 0.65, "增持": 0.7
        }
        self.hk_bearish_terms = {
            "穿位": -0.85, "止蝕": -0.8, "沽盤": -0.7, "外資沽": -0.8, "陰跌": -0.75,
            "熊軍": -0.75, "見頂": -0.8, "回調": -0.5, "利淡": -0.7, "輸到": -0.9,
            "唔敢睇": -0.6, "減持": -0.7, "沽空": -0.85, "死叉": -0.8, "跌穿": -0.8
        }

        # Korean financial retail terms
        self.kr_bullish_terms = {
            "떡상": 0.9, "서프라이즈": 0.85, "외인 순매수": 0.8, "저항선 돌파": 0.85,
            "골든크로스": 0.8, "줍줍": 0.7, "급등": 0.8, "쌍끌이": 0.75, "상향": 0.7,
            "모멘텀": 0.6, "호재": 0.8, "매수": 0.6, "자사주": 0.65, "신고가": 0.85
        }
        self.kr_bearish_terms = {
            "무너짐": -0.85, "손절": -0.8, "매도폭탄": -0.9, "매물대": -0.6,
            "차익실현": -0.5, "악재": -0.8, "하방": -0.7, "암반수": -0.85,
            "폭락": -0.9, "데드크로스": -0.8, "개미털기": -0.5, "신저가": -0.85, "물렸다": -0.7
        }

    def _init_position_patterns(self):
        # Regex patterns to detect stated retail positions in posts
        self.long_pattern = re.compile(
            r'(?:買入|建倉|掃貨|long|call|牛證|做多|多頭|매수|줍줍|물타기|롱\s*포지션|탑승)',
            re.IGNORECASE
        )
        self.short_pattern = re.compile(
            r'(?:沽出|沽空|止蝕|減持|short|put|熊證|做空|空頭|손절|매도|숏\s*포지션|탈출)',
            re.IGNORECASE
        )
        self.flat_pattern = re.compile(
            r'(?:觀望|清倉|空倉|平倉|sideline|cash|flat|관망|현금화|포지션\s*정리)',
            re.IGNORECASE
        )

        # Regex to capture price targets or technical levels mentioned in text
        self.level_pattern = re.compile(
            r'(?:目標價|目標|阻力|支持|穿|跌到|睇|target|TP|SL|support|resist|지지선|저항선|목표가)\s*[:：]?\s*([0-9]+(?:\.[0-9]+)?)',
            re.IGNORECASE
        )

    def extract_stated_position(self, text: str) -> Optional[str]:
        if not text:
            return None
        has_long = bool(self.long_pattern.search(text))
        has_short = bool(self.short_pattern.search(text))
        has_flat = bool(self.flat_pattern.search(text))

        if has_long and not has_short:
            return "LONG"
        elif has_short and not has_long:
            return "SHORT"
        elif has_flat:
            return "FLAT"
        return None

    def extract_mentioned_levels(self, text: str, current_price: Optional[float] = None) -> List[float]:
        if not text:
            return []
        matches = self.level_pattern.findall(text)
        levels = []
        for m in matches:
            try:
                val = float(m)
                # Filter out absurd or unrelated numbers (e.g. years, percentages)
                if current_price is not None:
                    if 0.2 * current_price <= val <= 5.0 * current_price:
                        levels.append(round(val, 2))
                elif val > 0:
                    levels.append(round(val, 2))
            except ValueError:
                continue
        return levels

    def analyze_text(self, text: str) -> Dict[str, Any]:
        if not text or not text.strip():
            return {
                "score": 0.0,
                "direction": "NEUTRAL",
                "confidence": 0.0,
                "stated_position": None,
                "mentioned_levels": []
            }

        cleaned = text.strip()
        stated_pos = self.extract_stated_position(cleaned)
        levels = self.extract_mentioned_levels(cleaned)

        # 1. Lexicon scanning
        lexicon_score = 0.0
        matches = 0

        for term, weight in self.hk_bullish_terms.items():
            if term in cleaned:
                lexicon_score += weight
                matches += 1
        for term, weight in self.hk_bearish_terms.items():
            if term in cleaned:
                lexicon_score += weight
                matches += 1

        for term, weight in self.kr_bullish_terms.items():
            if term in cleaned:
                lexicon_score += weight
                matches += 1
        for term, weight in self.kr_bearish_terms.items():
            if term in cleaned:
                lexicon_score += weight
                matches += 1

        # 2. VADER score
        vader_res = self.vader.polarity_scores(cleaned)
        vader_compound = vader_res["compound"]

        # 3. Optional transformer
        transformer_score = None
        if self.transformer_pipeline:
            try:
                # FinBERT outputs: positive, negative, neutral
                res = self.transformer_pipeline(cleaned[:512])[0]
                lbl = res["label"].lower()
                conf = res["score"]
                if lbl == "positive":
                    transformer_score = conf
                elif lbl == "negative":
                    transformer_score = -conf
                else:
                    transformer_score = 0.0
            except Exception:
                transformer_score = None

        # Composite polarity determination
        if transformer_score is not None:
            composite = (0.5 * transformer_score) + (0.5 * (lexicon_score / matches if matches > 0 else vader_compound))
        elif matches > 0:
            avg_lex = lexicon_score / matches
            composite = (0.7 * avg_lex) + (0.3 * vader_compound) if abs(vader_compound) > 0.05 else avg_lex
        else:
            composite = vader_compound

        composite = max(-1.0, min(1.0, composite))

        if composite >= 0.10:
            direction = "BULLISH"
        elif composite <= -0.10:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        return {
            "score": round(composite, 4),
            "direction": direction,
            "confidence": round(abs(composite), 4),
            "stated_position": stated_pos,
            "mentioned_levels": levels,
            "matched_terms": matches
        }

    def score_batch(self, items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        for item in items:
            text = item.get("text", "")
            if not text and "title" in item:
                text = item["title"]
            analysis = self.analyze_text(text)
            item["sentiment_score"] = analysis["score"]
            item["sentiment_direction"] = analysis["direction"]
            item["sentiment_confidence"] = analysis["confidence"]
            item["stated_position"] = analysis["stated_position"]
            item["mentioned_levels"] = analysis["mentioned_levels"]
        return items

    def evaluate_benchmark_agreement(self) -> Dict[str, Any]:
        """
        Evaluates agreement between the financial lexicon and VADER polarity
        on a calibrated benchmark set of multilingual financial phrases.
        """
        benchmark_phrases = [
            ("騰訊 業績超預期，今日大成交破位，買入等開車！", "BULLISH"),
            ("外資大幅沽出 阿里巴巴，成交放量下跌，睇淡。", "BEARISH"),
            ("小米 區間窄幅震盪，未有明確突破方向。", "NEUTRAL"),
            ("SK하이닉스 실적 서프라이즈! 외인 순매수 유입 롱 포지션 진입.", "BULLISH"),
            ("삼성전자 지지선 이탈... 물려서 손절 고민 중, 숏 관점.", "BEARISH"),
            ("카카오 박스권 횡보 중, 방향성 확인 필요.", "NEUTRAL"),
            ("Goldman Sachs raises target price on robust cloud margins", "BULLISH"),
            ("Regulatory headwinds and margin compression pressure earnings", "BEARISH")
        ]

        correct = 0
        total = len(benchmark_phrases)

        for text, true_label in benchmark_phrases:
            res = self.analyze_text(text)
            if res["direction"] == true_label:
                correct += 1

        agreement_pct = round((correct / total) * 100, 1)
        return {
            "sample_size": total,
            "agreement_percentage": agreement_pct,
            "benchmark_passed": bool(agreement_pct >= 85.0)
        }
