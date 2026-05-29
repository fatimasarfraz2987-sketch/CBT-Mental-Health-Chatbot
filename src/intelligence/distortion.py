"""
Cognitive distortion classifier.
Identifies thinking patterns like catastrophizing, all-or-nothing thinking, etc.
"""
from transformers import pipeline
import torch


class CognitiveDdistortionDetector:
    """Detect cognitive distortions in user messages."""
    
    # Dictionary of distortion patterns
    DISTORTION_KEYWORDS = {
        'catastrophizing': ['disaster', 'catastrophe', 'worst', 'ruined', 'never recover', 'terrible'],
        'all_or_nothing': ['always', 'never', 'perfect', 'failure', 'completely', 'total'],
        'overgeneralization': ['everyone', 'always', 'never', 'constantly', 'nobody'],
        'mind_reading': ['they think', 'know they', 'they must', 'probably think'],
        'emotional_reasoning': ['feel like', 'feel that', 'my feelings tell me'],
        'should_statements': ['should', 'must', 'have to', 'ought to'],
        'personalization': ['my fault', 'because of me', 'i caused', 'i made them']
    }
    
    def __init__(self, use_ml=True):
        """
        Initialize distortion detector.
        
        Args:
            use_ml: Use ML model for more sophisticated detection
        """
        self.use_ml = use_ml
        if use_ml:
            self.classifier = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli",
                device=0 if torch.cuda.is_available() else -1
            )
    
    def detect_keywords(self, text):
        """
        Rule-based detection using keyword matching.
        
        Args:
            text: User message
            
        Returns:
            List of detected distortions
        """
        text_lower = text.lower()
        detected = []
        
        for distortion, keywords in self.DISTORTION_KEYWORDS.items():
            for keyword in keywords:
                if keyword in text_lower:
                    detected.append(distortion)
                    break
        
        return list(set(detected))
    
    def detect_ml(self, text):
        """
        ML-based detection using zero-shot classification.
        
        Args:
            text: User message
            
        Returns:
            Distortion labels and confidence scores
        """
        distortion_types = list(self.DISTORTION_KEYWORDS.keys())
        
        result = self.classifier(
            text,
            distortion_types,
            multi_class=True
        )
        
        return {
            'labels': result['labels'],
            'scores': result['scores']
        }
    
    def analyze(self, text):
        """
        Full distortion analysis.
        
        Args:
            text: User message
            
        Returns:
            Detected distortions with confidence
        """
        keyword_distortions = self.detect_keywords(text)
        
        result = {
            'text': text,
            'keyword_distortions': keyword_distortions
        }
        
        if self.use_ml:
            ml_result = self.detect_ml(text)
            result['ml_analysis'] = ml_result
            
            # Combine results
            high_confidence_distortions = [
                label for label, score in zip(ml_result['labels'], ml_result['scores'])
                if score > 0.3
            ]
            result['detected_distortions'] = list(set(keyword_distortions + high_confidence_distortions))
        else:
            result['detected_distortions'] = keyword_distortions
        
        return result


if __name__ == "__main__":
    detector = CognitiveDdistortionDetector(use_ml=False)
    
    test_messages = [
        "I always mess everything up. I'm a complete failure.",
        "Everyone thinks I'm stupid.",
        "If this doesn't work out perfectly, it's a disaster."
    ]
    
    for msg in test_messages:
        result = detector.analyze(msg)
        print(f"Message: {result['text']}")
        print(f"Distortions: {result['detected_distortions']}\n")
