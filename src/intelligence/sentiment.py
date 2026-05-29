"""
WEEK 2: Intelligence Layer #1 — Sentiment Analysis
Goal: Build a sentiment analyzer combining VADER scores with BERT classifier.

Author Comment:
    Build a sentiment analyzer that combines VADER scores with a fine-tuned BERT classifier.
    Input: raw user message string
    Output: dict with keys: {"emotion": str, "vader_compound": float, "confidence": float}
    Emotions to detect: sad, anxious, angry, hopeless, neutral, positive
    Use j-hartmann/emotion-english-distilroberta-base from HuggingFace for BERT part.
    
    Let Copilot generate the implementation — review each suggested block before accepting.
"""
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import torch
import nltk

# Download VADER lexicon
try:
    nltk.data.find('sentiment/vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon', quiet=True)


class SentimentAnalyzer:
    """Multi-model sentiment and emotion analysis for therapy conversations."""
    
    # Emotion mapping from BERT outputs
    EMOTION_MAP = {
        'sadness': 'sad',
        'fear': 'anxious',
        'anger': 'angry',
        'joy': 'positive',
        'neutral': 'neutral',
        'surprise': 'neutral',
        'disgust': 'angry'
    }
    
    def __init__(self, use_bert=True):
        """
        Initialize sentiment analyzer.
        
        Args:
            use_bert: Use BERT emotion classifier in addition to VADER
        """
        print("[Sentiment] Initializing analyzer...")
        self.vader = SentimentIntensityAnalyzer()
        self.use_bert = use_bert
        
        if use_bert:
            try:
                self.emotion_pipeline = pipeline(
                    "text-classification",
                    model="j-hartmann/emotion-english-distilroberta-base",
                    device=0 if torch.cuda.is_available() else -1
                )
                print("[Sentiment] ✓ BERT emotion classifier loaded")
            except Exception as e:
                print(f"[Sentiment] ⚠ Failed to load BERT: {e}")
                self.use_bert = False
    
    def analyze_vader(self, text):
        """
        Analyze sentiment using VADER (fast, rule-based).
        
        Args:
            text: User message text
            
        Returns:
            Dictionary with VADER scores
        """
        scores = self.vader.polarity_scores(text)
        return {
            'method': 'VADER',
            'compound': scores['compound'],
            'positive': scores['pos'],
            'neutral': scores['neu'],
            'negative': scores['neg']
        }
    
    def analyze_bert(self, text):
        """
        Analyze emotion using BERT classifier.
        
        Args:
            text: User message text (max 512 tokens)
            
        Returns:
            BERT emotion classification result
        """
        try:
            # Truncate to 512 tokens (BERT limit)
            result = self.emotion_pipeline(text[:512])[0]
            return {
                'method': 'BERT',
                'emotion': result['label'],
                'confidence': result['score']
            }
        except Exception as e:
            print(f"[Sentiment] ⚠ BERT analysis failed: {e}")
            return {'method': 'BERT', 'emotion': 'neutral', 'confidence': 0.0}
    
    def analyze(self, text):
        """
        Full sentiment and emotion analysis.
        
        Args:
            text: User message
            
        Returns:
            Combined analysis with emotion, confidence, and VADER scores
        """
        vader_result = self.analyze_vader(text)
        
        result = {
            'text': text,
            'vader': vader_result
        }
        
        if self.use_bert:
            bert_result = self.analyze_bert(text)
            result['bert'] = bert_result
            
            # Map BERT emotion
            emotion = self.EMOTION_MAP.get(bert_result['emotion'], 'neutral')
            confidence = bert_result['confidence']
        else:
            # Fallback: determine emotion from VADER only
            compound = vader_result['compound']
            if compound < -0.5:
                emotion = 'sad'
                confidence = abs(compound)
            elif compound > 0.5:
                emotion = 'positive'
                confidence = compound
            else:
                emotion = 'neutral'
                confidence = 0.5
        
        result['emotion'] = emotion
        result['confidence'] = confidence
        
        return result


if __name__ == "__main__":
    print("\n" + "="*60)
    print("WEEK 2: Testing Sentiment Analyzer")
    print("="*60)
    
    analyzer = SentimentAnalyzer(use_bert=False)  # Set to True if BERT model available
    
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
        print(f"  VADER compound: {result['vader']['compound']:.2f}")
