"""
WEEK 2: Intelligence Layer #3 — Crisis Detection (CRITICAL SAFETY MODULE)
Goal: Identify high-risk messages requiring immediate intervention.

Author Comment:
    Build a crisis detection system combining rule-based keywords + ML classifier.
    CRITICAL SAFETY MODULE — must have 95%+ recall, false positives are acceptable.
    Keywords: suicide, self harm, kill myself, end it, don't want to live, hurt myself, etc.
    If crisis detected: set is_crisis=True and return helplines dict.
    Include Pakistani helplines: Umang (0311-7786264), Rozan (051-2890505).
    Input: user message
    Output: {"is_crisis": bool, "confidence": float, "helplines": dict}
"""

import re
from typing import Dict, List

import torch
from transformers import pipeline


class CrisisDetector:
    """Detect crisis indicators with high recall for safety."""

    CRISIS_KEYWORDS = {
        "suicide": [
            "suicide",
            "kill myself",
            "end my life",
            "not worth living",
            "take my life",
            "end it all",
            "don't want to live",
            "dont want to live",
        ],
        "self_harm": [
            "self harm",
            "cut myself",
            "hurt myself",
            "self-injury",
            "self injure",
            "slash wrist",
            "hurt themselves",
        ],
        "violence": [
            "hurt someone",
            "harm others",
            "attack them",
            "kill them",
            "violent",
        ],
        "substance_overdose": [
            "overdose",
            "od",
            "too many pills",
            "all the pills",
            "take everything",
        ],
    }

    CRISIS_LABELS = [
        "suicidal ideation",
        "self harm",
        "violence",
        "overdose",
    ]

    HELPLINES = {
        "international": {
            "international_lifeline": "1-800-273-8255",
            "crisis_text_line": "Text HOME to 741741",
        },
        "pakistan": {
            "umang": "0311-7786264",
            "rozan": "051-2890505",
            "befrienders_pakistan": "0333-3964873",
        },
        "other": {
            "findahelpline": "https://www.findahelpline.com",
        },
    }

    RISK_LEVELS = {
        "low": 0,
        "medium": 1,
        "high": 2,
        "critical": 3,
    }

    def __init__(self, use_ml=True):
        print("[Crisis] Initializing detector...")
        self.crisis_patterns = self._compile_patterns()
        self.use_ml = use_ml
        self.classifier = None

        if use_ml:
            try:
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=0 if torch.cuda.is_available() else -1,
                )
                print("[Crisis] ✓ ML crisis classifier loaded")
            except Exception as e:
                print(f"[Crisis] ⚠ ML classifier unavailable: {e}")
                self.classifier = None
                self.use_ml = False

        print("[Crisis] ✓ Crisis detector ready")
        print("[Crisis] ⚠ SAFETY MODE: 95%+ recall, false positives acceptable")

    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        patterns = {}
        for crisis_type, keywords in self.CRISIS_KEYWORDS.items():
            pattern_str = "|".join(re.escape(kw) for kw in keywords)
            patterns[crisis_type] = re.compile(pattern_str, re.IGNORECASE)
        return patterns

    def detect_keywords(self, text: str) -> Dict[str, bool]:
        detections = {}
        for crisis_type, pattern in self.crisis_patterns.items():
            detections[crisis_type] = bool(pattern.search(text))
        return detections

    def detect_ml(self, text: str) -> Dict[str, float]:
        if self.classifier is None:
            return {}

        try:
            result = self.classifier(text, self.CRISIS_LABELS, multi_class=True)
            return {label: score for label, score in zip(result["labels"], result["scores"])}
        except Exception as e:
            print(f"[Crisis] ⚠ ML detection failed: {e}")
            return {}

    def assess_risk_level(self, text: str, keyword_matches: Dict[str, bool], ml_scores: Dict[str, float]) -> str:
        text_lower = text.lower()
        if keyword_matches.get("suicide") or ml_scores.get("suicidal ideation", 0.0) >= 0.3:
            if "now" in text_lower or "tonight" in text_lower or "im going to" in text_lower:
                return "critical"
            return "high"

        if keyword_matches.get("self_harm") or ml_scores.get("self harm", 0.0) >= 0.3:
            return "high"

        if keyword_matches.get("substance_overdose") or ml_scores.get("overdose", 0.0) >= 0.3:
            return "high"

        if keyword_matches.get("violence") or ml_scores.get("violence", 0.0) >= 0.3:
            return "high"

        if sum(keyword_matches.values()) >= 2 or max(ml_scores.values(), default=0.0) >= 0.45:
            return "critical"

        if any(keyword_matches.values()) or max(ml_scores.values(), default=0.0) > 0.2:
            return "medium"

        return "low"

    def analyze(self, text: str) -> Dict:
        keyword_matches = self.detect_keywords(text)
        ml_scores = self.detect_ml(text) if self.use_ml else {}

        risk_level = self.assess_risk_level(text, keyword_matches, ml_scores)
        is_crisis = risk_level in ["high", "critical"]

        keyword_confidence = 0.8 + 0.05 * sum(keyword_matches.values()) if any(keyword_matches.values()) else 0.0
        ml_confidence = max(ml_scores.values()) if ml_scores else 0.0
        confidence = min(1.0, max(keyword_confidence, ml_confidence))

        combined_indicators = {
            **{k: 0.85 if v else 0.0 for k, v in keyword_matches.items()},
            **ml_scores,
        }
        top_indicator = None
        if combined_indicators:
            top_indicator = max(combined_indicators.items(), key=lambda x: x[1])[0]

        result = {
            "text": text,
            "is_crisis": is_crisis,
            "crisis_indicators": keyword_matches,
            "ml_scores": ml_scores,
            "top_indicator": top_indicator,
            "risk_level": risk_level,
            "risk_score": self.RISK_LEVELS.get(risk_level, 0),
            "confidence": confidence,
        }

        if is_crisis:
            result["helplines"] = self.HELPLINES
            result["recommendations"] = self._get_recommendations(risk_level)
        else:
            result["helplines"] = None
            result["recommendations"] = []

        return result

    @staticmethod
    def _get_recommendations(risk_level: str) -> List[str]:
        recommendations_map = {
            "low": [],
            "medium": [
                "⚠ Escalate to human therapist if risk increases",
                "📞 Have helpline numbers ready",
            ],
            "high": [
                "🚨 ALERT: Contact qualified human therapist immediately",
                "📞 Provide crisis helpline: Umang (0311-7786264) or Rozan (051-2890505)",
                "📱 Encourage immediate professional contact",
            ],
            "critical": [
                "🚨 CRITICAL: Emergency response required",
                "📞 Pakistan: Call Umang 0311-7786264 or Rozan 051-2890505",
                "🏥 International: Call 1-800-273-8255 (US) or Emergency Services",
                "⚠ This conversation may need to be reported to authorities if in danger",
            ],
        }
        return recommendations_map.get(risk_level, [])


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("WEEK 2: Testing Crisis Detector")
    print("=" * 60)

    detector = CrisisDetector()
    test_cases = [
        ("I'm feeling a bit down today", "normal"),
        ("I think about killing myself sometimes", "high"),
        ("I've decided to end my life tonight", "critical"),
        ("My friend hurt themselves yesterday", "medium"),
    ]

    for msg, expected in test_cases:
        result = detector.analyze(msg)
        print(f"\nMessage: {msg}")
        print(f"  Risk Level: {result['risk_level']} (expected: {expected})")
        print(f"  Is Crisis: {result['is_crisis']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        if result['recommendations']:
            print(f"  Recommendations: {result['recommendations'][0]}")
