"""
Main entry point for the CBT Mental Health Chatbot.
Handles startup of backend API and project initialization.
"""
import os
import sys
import argparse
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / 'src'))


def start_backend():
    """Start Flask backend server."""
    print("Starting Backend API...")
    try:
        from src.api.app import app
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"Error starting backend: {e}")
        return False
    return True


def check_requirements():
    """Check if required packages are installed."""
    print("Checking requirements...")
    try:
        import torch
        import transformers
        import flask
        import nltk
        print("✓ All required packages are installed")
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("Run: pip install -r requirements.txt")
        return False


def setup_nltk_data():
    """Download required NLTK data."""
    print("Setting up NLTK data...")
    try:
        import nltk
        nltk.download('vader_lexicon', quiet=True)
        nltk.download('punkt', quiet=True)
        print("✓ NLTK data downloaded")
        return True
    except Exception as e:
        print(f"✗ Error downloading NLTK data: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='CBT Mental Health Chatbot')
    parser.add_argument(
        '--backend',
        action='store_true',
        help='Start only the backend API server'
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check requirements without starting'
    )
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("CBT Mental Health Chatbot")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        return 1
    
    # Setup NLTK data
    if not setup_nltk_data():
        print("Warning: NLTK data setup failed, continuing anyway...")
    
    if args.check:
        print("\n✓ All checks passed!")
        return 0
    
    if args.backend:
        print("\nStarting backend only...")
        start_backend()
    else:
        print("\nStarting full application...")
        print("\nBackend: http://localhost:5000")
        print("Frontend: Open frontend/index.html in your browser")
        start_backend()


if __name__ == '__main__':
    sys.exit(main())
