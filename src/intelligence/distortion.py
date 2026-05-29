"""
WEEK 2: Intelligence Layer #2 — Cognitive Distortion Detection
Goal: Identify thinking patterns (10 distortion types) in user messages.

Author Comment:
    Build a cognitive distortion classifier.
    10 distortion types: catastrophizing, black_and_white, overgeneralization,
    mind_reading, fortune_telling, emotional_reasoning, should_statements,
    labeling, personalization, magnification
    Input: user message string
    Output: dict with top distortion type and confidence score
    Use a fine-tuned BERT or zero-shot classification from HuggingFace.
    
    Let Copilot generate the implementation — review each suggested block before accepting.
"""
from transformers import pipeline
import torch
import re


class CognitiveDistortionDetector:
    """Detect cognitive distortions in user messages."""
    
    # Define 10 cognitive distortion patterns
    DISTORTIONS = {
        'catastrophizing': ['disaster', 'catastrophe', 'worst', 'ruined', 'never recover', 'terrible', 'horrible'],
        'black_and_white': ['always', 'never', 'perfect', 'failure', 'completely', 'total', 'all or nothing'],
        'overgeneralization': ['everyone', 'nobody', 'always', 'constantly', 'every time', 'all'],
        'mind_reading': ['they think', 'know they', 'they must', 'probably thinking', 'i know what they'],
        'fortune_telling': ['will happen', 'going to', 'definitely will', 'it\'s certain'],
        'emotional_reasoning': ['feel like', 'feel that', 'my feelings tell me', 'because i feel'],
        'should_statements': ['should', 'must', 'have to', 'ought to', 'supposed to'],
        'labeling': ['i\'m a', 'i\'m just a', 'i\'m worthless', 'i\'m failure', 'i\'m stupid'],
        'personalization': ['my fault', 'because of me', 'i caused', 'i made them', 'it\'s my doing'],
        'magnification': ['so bad', 'terrible', 'worst', 'unbearable', 'can\'t stand it']
    }
    
    def __init__(self, use_ml=False):
        """
        Initialize distortion detector.
        
        Args:
            use_ml: Use zero-shot classification for more sophisticated detection
        """
        print("[Distortion] Initializing detector...")
        self.use_ml = use_ml
        
        if use_ml:
            try:
                self.classifier = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",
                    device=0 if torch.cuda.is_available() else -1
                )
                print("[Distortion] ✓ BART zero-shot classifier loaded")
            except Exception as e:
                print(f"[Distortion] ⚠ Failed to load BART: {e}")
                self.use_ml = False
    
    def detect_keywords(self, text):
        """
        Rule-based detection using keyword matching.
        
        Args:
            text: User message
            
        Returns:
            List of detected distortions with confidence
        """
        text_lower = text.lower()
        detected = {}
        
        for distortion, keywords in self.DISTORTIONS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected[distortion] = 0.8  # High confidence for keyword match
                    break
        
        return detected
    
    def detect_ml(self, text):
        """
        ML-based detection using zero-shot classification.
        
        Args:
            text: User message
            
        Returns:
            Distortion labels and confidence scores
        """
        distortion_types = list(self.DISTORTIONS.keys())
        
        try:
            result = self.classifier(
                text,
                distortion_types,
                multi_class=True
            )
            return dict(zip(result['labels'], result['scores']))
        except Exception as e:
            print(f"[Distortion] ⚠ ML detection failed: {e}")
            return {}
    
    def analyze(self, text):
        """
        Full distortion analysis.
        
        Args:
            text: User message
            
        Returns:
            Detected distortions with top match and confidence
        """
        keyword_distortions = self.detect_keywords(text)
        
        result = {
            'text': text,
            'keyword_matches': keyword_distortions
        }
        
        if self.use_ml:
            ml_distortions = self.detect_ml(text)
            result['ml_matches'] = ml_distortions
            
            # Combine results (keyword matches override ML)
            all_distortions = {**ml_distortions, **keyword_distortions}
        else:
            all_distortions = keyword_distortions
        
        # Get top distortion
        if all_distortions:
            top_distortion = max(all_distortions.items(), key=lambda x: x[1])
            result['top_distortion'] = top_distortion[0]
            result['confidence'] = top_distortion[1]
        else:
            result['top_distortion'] = None
            result['confidence'] = 0.0
        
        return result


if __name__ == "__main__":
    print("\n" + "="*60)
    print("WEEK 2: Testing Cognitive Distortion Detector")
    print("="*60)
    
    detector = CognitiveDistortionDetector(use_ml=False)
    
    test_messages = [
        "I always mess everything up. I'm a complete failure.",
        "Everyone thinks I'm stupid.",
        "If this doesn't work out perfectly, it's a disaster.",
        "I should be able to handle this, I must be strong",
        "This is my fault. I caused all of this."
    ]
    
    for msg in test_messages:
        result = detector.analyze(msg)
        print(f"\nMessage: {msg}")
        print(f"  Top Distortion: {result['top_distortion']}")
        print(f"  Confidence: {result['confidence']:.2f}")
