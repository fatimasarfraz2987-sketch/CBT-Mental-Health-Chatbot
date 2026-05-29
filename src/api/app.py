"""
Flask/FastAPI backend for CBT mental health chatbot.
Handles chat requests, sentiment analysis, and safety checks.
"""
from flask import Flask, request, jsonify
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import modules (update paths as needed when modules are ready)
# from src.model.inference import DialogueInference
# from src.intelligence.sentiment import SentimentAnalyzer
# from src.intelligence.distortion import CognitiveDdistortionDetector
# from src.intelligence.crisis import CrisisDetector

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

# Initialize models (when ready)
# inference = DialogueInference()
# sentiment_analyzer = SentimentAnalyzer()
# distortion_detector = CognitiveDdistortionDetector()
# crisis_detector = CrisisDetector()


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'ok', 'message': 'CBT Chatbot API is running'}), 200


@app.route('/chat', methods=['POST'])
def chat():
    """
    Main chat endpoint.
    Receives user message and returns therapy response with analysis.
    """
    data = request.json
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400
    
    try:
        # Placeholder response (implement with actual models)
        response = {
            'user_message': user_message,
            'therapist_response': 'Thank you for sharing that. Can you tell me more?',
            'sentiment': {'overall_sentiment': 'NEUTRAL'},
            'risk_level': 'low',
            'distortions': []
        }
        
        return jsonify(response), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/analyze', methods=['POST'])
def analyze():
    """
    Analysis-only endpoint for debugging.
    Returns sentiment, distortions, and crisis indicators without generating response.
    """
    data = request.json
    user_message = data.get('message', '').strip()
    
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400
    
    try:
        analysis = {
            'user_message': user_message,
            'sentiment': {'overall_sentiment': 'NEUTRAL'},
            'distortions': [],
            'risk_indicators': {},
            'risk_level': 'low'
        }
        
        return jsonify(analysis), 200
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors."""
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors."""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    debug = os.getenv('DEBUG', 'False') == 'True'
    app.run(host='0.0.0.0', port=5000, debug=debug)
