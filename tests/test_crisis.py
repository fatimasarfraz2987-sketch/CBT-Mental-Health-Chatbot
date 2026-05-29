"""
Unit tests for crisis detection module.
"""
import unittest
from src.intelligence.crisis import CrisisDetector


class TestCrisisDetector(unittest.TestCase):
    """Test CrisisDetector class."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize detector for all tests."""
        cls.detector = CrisisDetector()
    
    def test_suicide_detection(self):
        """Test detection of suicide indicators."""
        text = "I want to kill myself"
        result = self.detector.analyze(text)
        
        self.assertTrue(result['crisis_indicators']['suicide'])
        self.assertIn(result['risk_level'], ['high', 'critical'])
    
    def test_self_harm_detection(self):
        """Test detection of self-harm indicators."""
        text = "I cut myself to feel better"
        result = self.detector.analyze(text)
        
        self.assertTrue(result['crisis_indicators']['self_harm'])
        self.assertIn(result['risk_level'], ['high', 'critical'])
    
    def test_overdose_detection(self):
        """Test detection of overdose indicators."""
        text = "I took an overdose of pills"
        result = self.detector.analyze(text)
        
        self.assertTrue(result['crisis_indicators']['substance_overdose'])
        self.assertIn(result['risk_level'], ['high', 'critical'])
    
    def test_no_crisis(self):
        """Test normal message without crisis indicators."""
        text = "I'm feeling a bit down today"
        result = self.detector.analyze(text)
        
        self.assertEqual(result['risk_level'], 'low')
        for indicator in result['crisis_indicators'].values():
            self.assertFalse(indicator)
    
    def test_high_risk_recommendations(self):
        """Test recommendations for high-risk messages."""
        text = "I've been thinking about hurting myself"
        result = self.detector.analyze(text)
        
        self.assertIn('recommendations', result)
        self.assertGreater(len(result['recommendations']), 0)


if __name__ == '__main__':
    unittest.main()
