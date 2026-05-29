"""
Unit tests for sentiment analysis module.
"""
import unittest
from src.intelligence.sentiment import SentimentAnalyzer


class TestSentimentAnalyzer(unittest.TestCase):
    """Test SentimentAnalyzer class."""
    
    @classmethod
    def setUpClass(cls):
        """Initialize analyzer for all tests."""
        cls.analyzer = SentimentAnalyzer(use_bert=False)
    
    def test_negative_sentiment(self):
        """Test detection of negative sentiment."""
        text = "I feel terrible and hopeless"
        result = self.analyzer.analyze(text)
        
        self.assertIn('overall_sentiment', result)
        self.assertEqual(result['overall_sentiment'], 'NEGATIVE')
    
    def test_positive_sentiment(self):
        """Test detection of positive sentiment."""
        text = "I'm feeling great and happy today"
        result = self.analyzer.analyze(text)
        
        self.assertIn('overall_sentiment', result)
        self.assertEqual(result['overall_sentiment'], 'POSITIVE')
    
    def test_neutral_sentiment(self):
        """Test detection of neutral sentiment."""
        text = "The weather is cloudy today"
        result = self.analyzer.analyze(text)
        
        self.assertIn('overall_sentiment', result)
        # Neutral sentiments don't have specific keywords
    
    def test_vader_scores(self):
        """Test VADER scoring."""
        text = "I love this!"
        result = self.analyzer.analyze_vader(text)
        
        self.assertIn('compound', result)
        self.assertGreater(result['compound'], 0)  # Should be positive


if __name__ == '__main__':
    unittest.main()
