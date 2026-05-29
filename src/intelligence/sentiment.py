"""
Sentiment analysis layer using VADER and BERT.
Detects emotional state of user messages.
"""
from nltk.sentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import torch


class SentimentAnalyzer:
    """Multi-model sentiment analysis for therapy conversations."""
    
    def __init__(self, use_bert=True):
        """
        Initialize sentiment analyzer.
        
        Args:
            use_bert: Use BERT model in addition to VADER
        """
        self.vader = SentimentIntensityAnalyzer()
        self.use_bert = use_bert
        
        if use_bert:
            self.bert_pipeline = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=0 if torch.cuda.is_available() else -1
            )
    
    def analyze_vader(self, text):
        """
        Analyze sentiment using VADER (fast, rule-based).
        
        Args:
            text: User message text
            
        Returns:
            Dictionary with compound, pos, neu, neg scores
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
        Analyze sentiment using BERT (slower, more accurate).
        
        Args:
            text: User message text
            
        Returns:
            BERT sentiment label and confidence
        """
        result = self.bert_pipeline(text[:512])[0]  # BERT limit: 512 tokens
        return {
            'method': 'BERT',
            'label': result['label'],
            'confidence': result['score']
        }
    
    def analyze(self, text):
        """
        Full sentiment analysis using both methods.
        
        Args:
            text: User message
            
        Returns:
            Combined sentiment analysis results
        """
        vader_result = self.analyze_vader(text)
        
        result = {
            'text': text,
            'vader': vader_result
        }
        
        if self.use_bert:
            bert_result = self.analyze_bert(text)
            result['bert'] = bert_result
            
            # Determine final sentiment
            if vader_result['compound'] < -0.5 or bert_result['label'] == 'NEGATIVE':
                result['overall_sentiment'] = 'NEGATIVE'
            elif vader_result['compound'] > 0.5 or bert_result['label'] == 'POSITIVE':
                result['overall_sentiment'] = 'POSITIVE'
            else:
                result['overall_sentiment'] = 'NEUTRAL'
        else:
            if vader_result['compound'] < -0.1:
                result['overall_sentiment'] = 'NEGATIVE'
            elif vader_result['compound'] > 0.1:
                result['overall_sentiment'] = 'POSITIVE'
            else:
                result['overall_sentiment'] = 'NEUTRAL'
        
        return result


if __name__ == "__main__":
    analyzer = SentimentAnalyzer(use_bert=True)
    
    test_message = "I feel terrible today and don't know how to cope"
    result = analyzer.analyze(test_message)
    
    print(f"Message: {result['text']}")
    print(f"Overall Sentiment: {result['overall_sentiment']}")
    print(f"VADER: {result['vader']}")
    if 'bert' in result:
        print(f"BERT: {result['bert']}")
