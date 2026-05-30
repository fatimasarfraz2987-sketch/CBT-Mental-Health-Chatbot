"""
WEEK 2: Intelligence Layer #1 — Sentiment Analysis
Goal: Build a sentiment analyzer combining VADER scores with a BERT classifier.

Author Comment:
    Build a sentiment analyzer that combines VADER scores with a fine-tuned BERT classifier.
    Input: raw user message string
    Output: dict with keys: {"emotion": str, "vader_compound": float, "confidence": float}
    Emotions to detect: sad, anxious, angry, hopeless, neutral, positive
    Use j-hartmann/emotion-english-distilroberta-base from HuggingFace for BERT part.
"""

from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import torch
import nltk

try:
    nltk.data.find("sentiment/vader_lexicon")
except LookupError:
    nltk.download("vader_lexicon", quiet=True)


class SentimentAnalyzer:
    """Multi-model sentiment and emotion analysis for therapy conversations."""

    EMOTION_MAP = {
        "sadness": "sad",
        "fear": "anxious",
        "anger": "angry",
        "joy": "positive",
        "neutral": "neutral",
        "surprise": "neutral",
        "disgust": "angry",
    }

    def __init__(self, use_bert=True):
        print("[Sentiment] Initializing analyzer...")
        self.vader = SentimentIntensityAnalyzer()
        self.use_bert = use_bert
        self.emotion_pipeline = None

        if use_bert:
            try:
                self.emotion_pipeline = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    device=0 if torch.cuda.is_available() else -1,
                    return_all_scores=False,
                )
                print("[Sentiment] ✓ BERT emotion classifier loaded")
            except Exception as e:
                print(f"[Sentiment] ⚠ Failed to load BERT: {e}")
                self.use_bert = False

    def analyze_vader(self, text):
        scores = self.vader.polarity_scores(text)
        return {
            "method": "VADER",
            "compound": scores["compound"],
            "positive": scores["pos"],
            "neutral": scores["neu"],
            "negative": scores["neg"],
        }

    def analyze_bert(self, text):
        try:
            result = self.emotion_pipeline(text[:512])[0]
            return {
                "method": "BERT",
                "emotion": result["label"],
                "confidence": result["score"],
            }
        except Exception as e:
            print(f"[Sentiment] ⚠ BERT analysis failed: {e}")
            return {"method": "BERT", "emotion": "neutral", "confidence": 0.0}

    def analyze(self, text):
        vader_result = self.analyze_vader(text)
        compound = vader_result["compound"]

        if self.use_bert and self.emotion_pipeline is not None:
            bert_result = self.analyze_bert(text)
            bert_emotion = bert_result["emotion"].lower()
            emotion = self.EMOTION_MAP.get(bert_emotion, "neutral")
            if emotion == "sad" and compound < -0.5:
                emotion = "hopeless"
            confidence = bert_result["confidence"]
        else:
            if compound <= -0.5:
                emotion = "sad"
                confidence = min(1.0, abs(compound))
            elif compound >= 0.5:
                emotion = "positive"
                confidence = min(1.0, compound)
            else:
                emotion = "neutral"
                confidence = 0.5

        if emotion == "sad" and compound <= -0.7:
            emotion = "hopeless"

        overall_sentiment = (
            "POSITIVE" if compound > 0.05 else "NEGATIVE" if compound < -0.05 else "NEUTRAL"
        )

        return {
            "emotion": emotion,
            "vader_compound": compound,
            "confidence": confidence,
            "overall_sentiment": overall_sentiment,
            "vader": vader_result,
        }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("WEEK 2: Testing Sentiment Analyzer")
    print("=" * 60)

    analyzer = SentimentAnalyzer(use_bert=False)
    test_messages = [
        "I'm feeling great and hopeful!",
        "I feel terrible and don't see any point",
        "I'm anxious about everything",
        "I'm just fine, nothing special",
    ]

    for msg in test_messages:
        result = analyzer.analyze(msg)
        print(f"\nMessage: {msg}")
        print(f"  Emotion: {result['emotion']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        print(f"  VADER compound: {result['vader_compound']:.2f}")
