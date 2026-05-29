"""
Unit tests for cognitive distortion detector.
"""
import unittest
from src.intelligence.distortion import CognitiveDdistortionDetector


class TestDistortionDetector(unittest.TestCase):
    """Test CognitiveDdistortionDetector class."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize detector for all tests."""
        cls.detector = CognitiveDdistortionDetector(use_ml=False)
    
    def test_catastrophizing_detection(self):
        """Test detection of catastrophizing."""
        text = "This is a complete disaster, everything is ruined"
        result = self.detector.analyze(text)
        
        self.assertIn('catastrophizing', result['detected_distortions'])
    
    def test_all_or_nothing_detection(self):
        """Test detection of all-or-nothing thinking."""
        text = "I always fail at everything"
        result = self.detector.analyze(text)
        
        self.assertIn('all_or_nothing', result['detected_distortions'])
    
    def test_overgeneralization_detection(self):
        """Test detection of overgeneralization."""
        text = "Everyone always leaves me"
        result = self.detector.analyze(text)
        
        self.assertIn('overgeneralization', result['detected_distortions'])
    
    def test_no_distortions(self):
        """Test when no distortions are present."""
        text = "I'm going to the store today"
        result = self.detector.analyze(text)
        
        self.assertEqual(len(result['detected_distortions']), 0)


if __name__ == '__main__':
    unittest.main()
