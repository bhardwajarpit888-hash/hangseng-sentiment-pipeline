import re
import logging
from typing import Dict, Any, List
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger(__name__)


class MultilingualFinancialSentimentAnalyzer:
    """
    Multilingual financial sentiment engine tailored for Hang Seng (Cantonese / Traditional Chinese),
    KRX (Korean), and international financial English news.
    Combines rule-based financial domain lexicons with VADER polarity scoring.
    """

    def __init__(self):
        self.vader = SentimentIntensityAnalyzer()
        self._init_financial_lexicons()

    def _init_financial_lexicons(self):
        # Hong Kong / Cantonese retail & financial terms
        self.hk_bullish_terms = {
            "破位": 0.8, "大成交": 0.6, "阻力位破": 0.85, "牛軍": 0.75, "建倉": 0.65,
            "開車": 0.7, "派息": 0.5, "超預期": 0.9, "長線買入": 0.8, "底背馳": 0.75,
            "金叉": 0.8, "反彈": 0.6, "掃入": 0.7, "平": 0.4, "回購": 0.65, "增持": 0.7
        }
        self.hk_bearish_terms = {
            "穿位": -0.85, "止蝕": -0.8, "沽盤": -0.7, "外資沽": -0.8, "陰跌": -0.75,
            "熊軍": -0.75, "見頂": -0.8, "回調": -0.5, "利淡": -0.7, "輸到": -0.9,
            "唔敢睇": -0.6, "減持": -0.7, "大成交沽": -0.85, "死叉": -0.8
        }

        # Korean financial retail terms
        self.kr_bullish_terms = {
            "떡상": 0.9, "서프라이즈": 0.85, "외인 순매수": 0.8, "저항선 돌파": 0.85,
            "골든크로스": 0.8, "줍줍": 0.7, "급등": 0.8, "쌍끌이": 0.75, "상향": 0.7,
            "모멘텀": 0.6, "호재": 0.8, "매수": 0.5, "자사주": 0.65, "신고가": 0.85
        }
        self.kr_bearish_terms = {
            "무너짐": -0.85, "손절": -0.8, "매도폭탄": -0.9, "매물대": -0.6,
            "차익실현": -0.5, "악재": -0.8, "하방": -0.7, "암반수": -0.85,
            "폭락": -0.9, "데드크로스": -0.8, "개미털기": -0.5, "신저가": -0.85
        }

    def analyze_text(self, text: str, language_hint: str = "auto") -> Dict[str, Any]:
        """
        Analyzes a single post or headline text and returns composite polarity and category.
        """
        if not text or not text.strip():
            return {"score": 0.0, "direction": "NEUTRAL", "confidence": 0.0}

        cleaned = text.strip()
        lexicon_score = 0.0
        matches = 0

        # Scan Cantonese / Chinese financial terms
        for term, weight in self.hk_bullish_terms.items():
            if term in cleaned:
                lexicon_score += weight
                matches += 1
        for term, weight in self.hk_bearish_terms.items():
            if term in cleaned:
                lexicon_score += weight
                matches += 1

        # Scan Korean financial terms
        for term, weight in self.kr_bullish_terms.items():
            if term in cleaned:
                lexicon_score += weight
                matches += 1
        for term, weight in self.kr_bearish_terms.items():
            if term in cleaned:
                lexicon_score += weight
                matches += 1

        # VADER English / general sentiment
        vader_res = self.vader.polarity_scores(cleaned)
        vader_compound = vader_res["compound"]

        # Blend scores
        if matches > 0:
            avg_lexicon = lexicon_score / matches
            # If both lexicon and VADER are present, weight domain lexicon heavily
            if abs(vader_compound) > 0.05:
                composite = (avg_lexicon * 0.7) + (vader_compound * 0.3)
            else:
                composite = avg_lexicon
        else:
            composite = vader_compound

        # Clamp between -1.0 and 1.0
        composite = max(-1.0, min(1.0, composite))

        if composite >= 0.10:
            direction = "BULLISH"
        elif composite <= -0.10:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        confidence = round(abs(composite), 4)

        return {
            "score": round(composite, 4),
            "direction": direction,
            "confidence": confidence,
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
        return items
