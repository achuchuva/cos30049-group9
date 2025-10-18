# Spam Detection API Backend

FastAPI backend for spam detection using Logistic Regression. Provides REST endpoints for text classification with batch processing support.

## Features

- REST API with FastAPI
- Single and batch text prediction
- Health monitoring endpoints  
- Auto-generated API documentation
- CORS enabled for frontend integration

## Installation

### Prerequisites
- Python 3.8+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# Navigate to backend directory
cd backend

# Install dependencies using uv
uv sync
```

## Running the Application

```bash
# Development mode with auto-reload
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### Access the API
- **API Root**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check and status |
| POST | `/api/v1/predict` | Single text prediction |
| POST | `/api/v1/predict/batch` | Batch predictions |
| GET | `/api/v1/model/info` | Get model information |
| PUT | `/api/v1/model/reload` | Reload the model |

### Example Usage

**Single Prediction:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Win money now! Click here!"}'
```

**Batch Prediction:**
```bash
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Hello there", "WIN CASH NOW!!!"]}'
```

**Interactive Documentation:**
Visit http://localhost:8000/docs to test endpoints directly in your browser.

## Project Structure

```
backend/
├── app/
│   ├── main.py               # FastAPI application & routes  
│   ├── models.py             # Pydantic request/response models
│   ├── predictor.py          # ML prediction logic
│   └── config.py             # Configuration management
├── models/
│   └── logreg.joblib          # Trained model
├── artifacts/
│   ├── spam_vectorizer.joblib # TF-IDF vectorizer
│   └── spam_scaler.joblib     # Feature scaler
└── pyproject.toml             # Project dependencies
```
