"""
Crisis detection safety layer.
Identifies high-risk messages requiring immediate intervention.
"""
import re
from typing import Dict, List


class CrisisDetector:
    """Detect crisis indicators in user messages."""
    
    # Crisis keywords and patterns
    CRISIS_KEYWORDS = {
        'suicide': ['suicide', 'kill myself', 'end my life', 'not worth living'],
        'self_harm': ['self harm', 'cut myself', 'hurt myself', 'self-injury'],
        'violence': ['hurt someone', 'harm others', 'attack', 'kill them'],
        'substance_overdose': ['overdose', 'OD', 'too many pills', 'all the pills']
    }
    
    RISK_LEVELS = {
        'low': 0,
        'medium': 1,
        'high': 2,
        'critical': 3
    }
    
    def __init__(self):
        """Initialize crisis detector."""
        self.crisis_patterns = self._compile_patterns()
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for crisis detection."""
        patterns = {}
        for crisis_type, keywords in self.CRISIS_KEYWORDS.items():
            pattern_str = '|'.join(keywords)
            patterns[crisis_type] = re.compile(pattern_str, re.IGNORECASE)
        return patterns
    
    def detect_crisis_keywords(self, text: str) -> Dict[str, bool]:
        """
        Detect crisis keywords in text.
        
        Args:
            text: User message
            
        Returns:
            Dictionary mapping crisis types to boolean detection
        """
        detections = {}
        for crisis_type, pattern in self.crisis_patterns.items():
            detections[crisis_type] = bool(pattern.search(text))
        return detections
    
    def assess_risk_level(self, text: str) -> str:
        """
        Assess overall risk level.
        
        Args:
            text: User message
            
        Returns:
            Risk level: 'low', 'medium', 'high', or 'critical'
        """
        detections = self.detect_crisis_keywords(text)
        
        # Count crisis indicators
        crisis_count = sum(detections.values())
        
        if crisis_count >= 2:
            risk = 'critical'
        elif detections.get('suicide') or detections.get('self_harm'):
            risk = 'high'
        elif detections.get('substance_overdose') or detections.get('violence'):
            risk = 'high'
        elif crisis_count == 1:
            risk = 'medium'
        else:
            risk = 'low'
        
        return risk
    
    def analyze(self, text: str) -> Dict:
        """
        Full crisis analysis.
        
        Args:
            text: User message
            
        Returns:
            Crisis analysis with risk level and recommendations
        """
        detections = self.detect_crisis_keywords(text)
        risk_level = self.assess_risk_level(text)
        
        result = {
            'text': text,
            'crisis_indicators': detections,
            'risk_level': risk_level,
            'risk_score': self.RISK_LEVELS.get(risk_level, 0)
        }
        
        # Add recommendations based on risk level
        recommendations = self._get_recommendations(risk_level)
        result['recommendations'] = recommendations
        
        return result
    
    @staticmethod
    def _get_recommendations(risk_level: str) -> List[str]:
        """
        Get intervention recommendations based on risk level.
        
        Args:
            risk_level: Detected risk level
            
        Returns:
            List of recommendations
        """
        recommendations_map = {
            'low': ['Continue with standard therapy support'],
            'medium': ['Escalate to human therapist', 'Provide crisis resources'],
            'high': ['Immediate human therapist contact', 'Provide suicide hotline: 988'],
            'critical': ['URGENT: Alert emergency services', 'Provide crisis hotline immediately']
        }
        return recommendations_map.get(risk_level, [])


if __name__ == "__main__":
    detector = CrisisDetector()
    
    test_messages = [
        "I'm feeling a bit down today.",
        "I think about hurting myself sometimes.",
        "I've decided to end my life tonight."
    ]
    
    for msg in test_messages:
        result = detector.analyze(msg)
        print(f"Message: {result['text']}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Recommendations: {result['recommendations']}\n")
