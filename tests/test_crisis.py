"""
WEEK 4: Testing Crisis Detection Module
Goal: Validate crisis detection with 95%+ recall, document precision/recall metrics.

Unit tests for the crisis detection module.
Test cases cover: explicit crisis phrases, ambiguous phrases, normal therapy phrases.
"""
import pytest
from src.intelligence.crisis import CrisisDetector


class TestCrisisDetector:
    """Test cases for CrisisDetector class."""
    
    @classmethod
    def setup_class(cls):
        """Initialize detector for all tests."""
        cls.detector = CrisisDetector()
    
    # ========================================================================
    # 20 EXPLICIT CRISIS PHRASES — MUST all return is_crisis=True (RECALL TEST)
    # ========================================================================
    explicit_crisis_phrases = [
        "I want to kill myself",
        "I'm going to end my life tonight",
        "I don't want to live anymore",
        "I've decided to commit suicide",
        "I'm going to hurt myself with a knife",
        "I'm going to take an overdose",
        "I need to cut myself to feel better",
        "I can't take it anymore, I'm done",
        "The only way out is to kill myself",
        "I'm planning to end it all",
        "I just want to self harm",
        "I'm going to hurt myself",
        "I should just end my life",
        "I'm in so much pain, I need to harm myself",
        "I'm going to take all the pills",
        "I can't do this anymore, I'll take my own life",
        "I want to commit suicide right now",
        "I'm going to slash my wrists",
        "This pain is unbearable, I need to overdose",
        "I've decided to end everything tonight"
    ]
    
    @pytest.mark.parametrize("message", explicit_crisis_phrases)
    def test_explicit_crisis_detection(self, message):
        """Test that all explicit crisis phrases are detected."""
        result = self.detector.analyze(message)
        assert result['is_crisis'] == True, f"Failed to detect crisis: {message}"
        assert result['risk_level'] in ['high', 'critical'], f"Wrong risk level for: {message}"
        assert result['confidence'] > 0.5, f"Low confidence for: {message}"
    
    # ========================================================================
    # 10 AMBIGUOUS PHRASES — Document expected output
    # ========================================================================
    ambiguous_phrases = [
        ("My friend hurt themselves yesterday", "medium"),
        ("I sometimes think bad things", "low"),
        ("I'm not sure if life is worth it", "low"),
        ("I've been having dark thoughts", "medium"),
        ("People keep telling me I should hurt myself", "medium"),
        ("I cut vegetables in the kitchen", "low"),
        ("I hurt my shoulder playing football", "low"),
        ("Everyone says I'm killing it at work", "low"),
        ("I feel like everything is falling apart", "low"),
        ("Sometimes I wonder what it would be like if I wasn't here", "medium"),
    ]
    
    @pytest.mark.parametrize("message,expected_level", ambiguous_phrases)
    def test_ambiguous_phrases(self, message, expected_level):
        """Test ambiguous phrases and document outcomes."""
        result = self.detector.analyze(message)
        print(f"\nAmbiguous: {message}")
        print(f"  Risk Level: {result['risk_level']} (expected: {expected_level})")
        print(f"  Is Crisis: {result['is_crisis']}")
        print(f"  Confidence: {result['confidence']:.2f}")
    
    # ========================================================================
    # 10 NORMAL THERAPY PHRASES — MUST all return is_crisis=False
    # ========================================================================
    normal_therapy_phrases = [
        "I'm feeling anxious today, can we talk about it?",
        "I had a difficult conversation with my boss",
        "I'm trying to manage my stress better",
        "I've been thinking about what we discussed last week",
        "I'm working on my coping skills",
        "I feel sad, but I'm trying to be strong",
        "Can you help me understand why I feel this way?",
        "I'm taking my medications as prescribed",
        "I attended my therapy session and it helped",
        "I'm making progress with my goals"
    ]
    
    @pytest.mark.parametrize("message", normal_therapy_phrases)
    def test_normal_therapy_phrases(self, message):
        """Test that normal therapy phrases are NOT flagged as crisis."""
        result = self.detector.analyze(message)
        assert result['is_crisis'] == False, f"False positive crisis detection: {message}"
        assert result['risk_level'] == 'low', f"Should be low risk: {message}"
    
    # ========================================================================
    # PERFORMANCE METRICS & VALIDATION
    # ========================================================================
    def test_recall_metric(self):
        """
        Calculate recall: TP / (TP + FN)
        Recall = crisis phrases correctly detected / total crisis phrases
        REQUIREMENT: recall >= 0.95 (95%)
        """
        tp = 0
        for message in self.explicit_crisis_phrases:
            result = self.detector.analyze(message)
            if result['is_crisis']:
                tp += 1
        
        total_crisis = len(self.explicit_crisis_phrases)
        recall = tp / total_crisis if total_crisis > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"CRISIS DETECTION RECALL METRIC")
        print(f"{'='*60}")
        print(f"True Positives: {tp}/{total_crisis}")
        print(f"Recall: {recall:.2%}")
        print(f"REQUIREMENT: >= 95.00%")
        print(f"STATUS: {'✓ PASS' if recall >= 0.95 else '✗ FAIL'}")
        print(f"{'='*60}")
        
        assert recall >= 0.95, f"Recall {recall:.2%} below 95% threshold!"
    
    def test_precision_metric(self):
        """
        Calculate precision: TP / (TP + FP)
        Precision = crisis phrases correctly detected / total detected as crisis
        """
        tp = 0
        total_detected = 0
        
        for message in self.explicit_crisis_phrases:
            result = self.detector.analyze(message)
            if result['is_crisis']:
                tp += 1
                total_detected += 1
        
        fp = 0
        for message in self.normal_therapy_phrases:
            result = self.detector.analyze(message)
            if result['is_crisis']:
                fp += 1
                total_detected += 1
        
        precision = tp / total_detected if total_detected > 0 else 0
        
        print(f"\n{'='*60}")
        print(f"CRISIS DETECTION PRECISION METRIC")
        print(f"{'='*60}")
        print(f"True Positives: {tp}")
        print(f"False Positives: {fp}")
        print(f"Total Detected: {total_detected}")
        print(f"Precision: {precision:.2%}")
        print(f"NOTE: False positives acceptable for safety")
        print(f"{'='*60}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
