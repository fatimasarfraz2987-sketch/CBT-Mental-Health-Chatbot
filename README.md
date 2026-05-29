# CBT Mental Health Chatbot 🧠

An AI-powered Cognitive Behavioral Therapy (CBT) chatbot that provides mental health support through evidence-based therapeutic techniques.

## Features

- **AI Therapy Responses**: Fine-tuned DialoGPT generates empathetic, therapy-informed responses
- **Sentiment Analysis**: VADER + BERT models detect emotional state in real-time
- **Cognitive Distortion Detection**: Identifies thinking patterns like catastrophizing and all-or-nothing thinking
- **Crisis Detection**: Flags high-risk messages and provides emergency resources
- **Interactive Web UI**: Modern chat interface with real-time analysis panel

## Project Structure

```
codemind/
├── data/
│   ├── raw/                  # Downloaded datasets (never modified)
│   │   ├── counsel_chat/
│   │   ├── empathetic_dialogues/
│   │   └── daic_woz/
│   └── processed/            # Cleaned, tokenized training data
│
├── models/
│   ├── base/                 # Downloaded base models
│   └── fine_tuned/           # Your trained checkpoints
│
├── src/
│   ├── data_processing/      # Data cleaning & tokenization
│   ├── model/                # Training & inference
│   ├── intelligence/         # Sentiment, distortion, crisis detection
│   └── api/                  # Flask backend
│
├── frontend/                 # Chat UI (HTML/CSS/JS)
├── notebooks/                # Google Colab notebooks
├── tests/                    # Unit tests
├── .env                      # Environment variables (not committed)
├── .gitignore
├── requirements.txt
└── run.py                    # Main entry point
```

## Installation

### 1. Clone Repository
```bash
git clone https://github.com/fatimasarfraz2987-sketch/CBT-Mental-Health-Chatbot.git
cd CBT-Mental-Health-Chatbot
```

### 2. Create Virtual Environment
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables
Create `.env` file (DO NOT COMMIT):
```
HUGGINGFACE_TOKEN=your_token_here
FLASK_SECRET_KEY=your_secret_key
MODEL_PATH=models/fine_tuned/checkpoint-final
DATABASE_URL=sqlite:///codemind.db
DEBUG=True
```

## Usage

### Start Backend
```bash
python run.py --backend
```

### Run Full Application
```bash
python run.py
```

### Check Requirements
```bash
python run.py --check
```

### Run Tests
```bash
python -m pytest tests/
```

## Data Preparation

1. **Download Datasets**:
   ```bash
   python src/data_processing/clean_datasets.py
   ```

2. **Tokenize Data**:
   ```bash
   python src/data_processing/tokenize.py
   ```

## Model Training

```bash
python src/model/train.py \
    --model-name "microsoft/DialoGPT-small" \
    --data-path "data/processed/cbt_pairs.json" \
    --num-epochs 3 \
    --batch-size 8
```

## API Endpoints

### Health Check
```
GET /health
```

### Chat
```
POST /chat
Content-Type: application/json

{
    "message": "I'm feeling anxious"
}

Response:
{
    "user_message": "I'm feeling anxious",
    "therapist_response": "...",
    "sentiment": {
        "overall_sentiment": "NEGATIVE"
    },
    "risk_level": "medium",
    "distortions": ["catastrophizing"]
}
```

### Analysis Only
```
POST /analyze
Content-Type: application/json

{
    "message": "I'm feeling anxious"
}
```

## Technologies

- **Models**: DialoGPT, Flan-T5, BERT
- **Backend**: Flask/FastAPI
- **Frontend**: HTML, CSS, JavaScript
- **Sentiment**: VADER, Transformers
- **Crisis Detection**: Pattern matching + BERT
- **Data**: Hugging Face Datasets

## Safety & Ethics

⚠️ **Important**: This chatbot is **NOT a replacement for professional mental health care**.

- Crisis indicators trigger emergency resource recommendations
- High-risk messages alert to human intervention
- All user data should be handled with HIPAA compliance
- Regular testing for harmful outputs

## Emergency Resources

- **988**: Suicide & Crisis Lifeline (US)
- **Crisis Text Line**: Text HOME to 741741
- **International**: findahelpline.com

## Development

### Run Linter
```bash
flake8 src/
```

### Format Code
```bash
black src/
```

### Type Checking
```bash
mypy src/
```

## Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature/your-feature`
5. Submit pull request

## License

MIT License - see LICENSE file for details

## Author

**Fatima Sarfraz**
- GitHub: [@fatimasarfraz2987-sketch](https://github.com/fatimasarfraz2987-sketch)

## Disclaimer

This project is for educational purposes. Always consult qualified mental health professionals for serious psychological concerns.

---

**Last Updated**: May 2026
