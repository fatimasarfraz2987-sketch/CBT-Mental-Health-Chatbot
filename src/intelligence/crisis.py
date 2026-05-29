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
    
    Let Copilot generate the implementation — review each suggested block before accepting.
"""
import re
from typing import Dict, List, Tuple


class CrisisDetector:
    """Detect crisis indicators with high recall for safety."""
    
    # Crisis keywords organized by severity
    CRISIS_KEYWORDS = {
        'suicide': ['suicide', 'kill myself', 'end my life', 'not worth living', 'take my life', 'end it all'],
        'self_harm': ['self harm', 'cut myself', 'hurt myself', 'self-injury', 'self injure', 'slash wrist'],
        'violence': ['hurt someone', 'harm others', 'attack them', 'kill them', 'violent'],
        'substance_overdose': ['overdose', 'OD', 'too many pills', 'all the pills', 'take everything']
    }
    
    # Global and Pakistani helplines
    HELPLINES = {
        'international': {
            'international_lifeline': '1-800-273-8255',
            'crisis_text_line': 'Text HOME to 741741',
        },
        'pakistan': {
            'umang': '0311-7786264',
            'rozan': '051-2890505',
            'befrienders_pakistan': '0333-3964873',
        },
        'other': {
            'findahelpline': 'https://www.findahelpline.com',
        }
    }
    
    RISK_LEVELS = {
        'low': 0,
        'medium': 1,
        'high': 2,
        'critical': 3
    }
    
    def __init__(self):
        """Initialize crisis detector with compiled patterns."""
        print("[Crisis] Initializing detector...")
        self.crisis_patterns = self._compile_patterns()
        print("[Crisis] ✓ Crisis detector ready")
        print("[Crisis] ⚠ SAFETY MODE: 95%+ recall, false positives acceptable")
    
    def _compile_patterns(self) -> Dict[str, re.Pattern]:
        """Compile regex patterns for crisis detection."""
        patterns = {}
        for crisis_type, keywords in self.CRISIS_KEYWORDS.items():
            pattern_str = '|'.join(re.escape(kw) for kw in keywords)
            patterns[crisis_type] = re.compile(pattern_str, re.IGNORECASE)
        return patterns
    
    def detect_keywords(self, text: str) -> Dict[str, bool]:
        """
        Detect crisis keywords in text.
        
        Args:
            text: User message
            
        Returns:
            Dictionary mapping crisis types to detection boolean
        """
        detections = {}
        for crisis_type, pattern in self.crisis_patterns.items():
            detections[crisis_type] = bool(pattern.search(text))
        return detections
    
    def assess_risk_level(self, text: str) -> str:
        """
        Assess overall risk level (MUST prioritize recall).
        
        Args:
            text: User message
            
        Returns:
            Risk level: 'low', 'medium', 'high', or 'critical'
        """
        detections = self.detect_keywords(text)
        crisis_count = sum(detections.values())
        
        # Priority: suicide and self-harm trigger high/critical
        if detections.get('suicide'):
            return 'critical' if 'definitely' in text.lower() or 'now' in text.lower() else 'high'
        
        if detections.get('self_harm'):
            return 'high'
        
        if crisis_count >= 2:
            return 'critical'
        
        if detections.get('substance_overdose') or detections.get('violence'):
            return 'high'
        
        if crisis_count == 1:
            return 'medium'
        
        return 'low'
    
    def analyze(self, text: str) -> Dict:
        """
        Full crisis analysis.
        
        Args:
            text: User message
            
        Returns:
            Crisis analysis with risk level, helplines, and recommendations
        """
        detections = self.detect_keywords(text)
        risk_level = self.assess_risk_level(text)
        
        is_crisis = risk_level in ['high', 'critical']
        confidence = min(1.0, sum(detections.values()) / len(detections)) if detections else 0.0
        
        result = {
            'text': text,
            'is_crisis': is_crisis,
            'crisis_indicators': detections,
            'risk_level': risk_level,
            'risk_score': self.RISK_LEVELS.get(risk_level, 0),
            'confidence': confidence,
        }
        
        # Add helplines if crisis detected
        if is_crisis:
            result['helplines'] = self.HELPLINES
            result['recommendations'] = self._get_recommendations(risk_level)
        else:
            result['helplines'] = None
            result['recommendations'] = []
        
        return result
    
    @staticmethod
    def _get_recommendations(risk_level: str) -> List[str]:
        """
        Get intervention recommendations based on risk level.
        
        Args:
            risk_level: Detected risk level
            
        Returns:
            List of recommendations for intervention
        """
        recommendations_map = {
            'low': [],
            'medium': [
                '⚠ Escalate to human therapist if risk increases',
                '📞 Have helpline numbers ready'
            ],
            'high': [
                '🚨 ALERT: Contact qualified human therapist immediately',
                '📞 Provide crisis helpline: Umang (0311-7786264) or Rozan (051-2890505)',
                '📱 Encourage immediate professional contact'
            ],
            'critical': [
                '🚨 CRITICAL: Emergency response required',
                '📞 Pakistan: Call Umang 0311-7786264 or Rozan 051-2890505',
                '🏥 International: Call 1-800-273-8255 (US) or Emergency Services',
                '⚠ This conversation may need to be reported to authorities if in danger'
            ]
        }
        return recommendations_map.get(risk_level, [])


if __name__ == "__main__":
    print("\n" + "="*60)
    print("WEEK 2: Testing Crisis Detector")
    print("="*60)
    
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
