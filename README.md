# COS30049 Group 9 Spam Detection

Full-stack website: FastAPI backend provides spam/ham predictions; React frontend displays results.

## Install

Clone repo then:

```bash
# Backend deps
cd backend
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Frontend deps
cd ../spamapp
npm install
```

## Run Backend
```bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```
Visit: http://localhost:8000/docs

## Run Frontend
```bash
cd spamapp
npm start
```
App: http://localhost:3000

## Model Artifacts
`backend/models/logreg.joblib` and `backend/artifacts/spam_vectorizer.joblib`, `spam_scaler.joblib` provided as trained models. `ai4cyber` folder contains datasets and training code for different models. Assignment 2 contains more details.

## Batch Demo
Use the UI batch demo button or:
```bash
curl -X POST http://localhost:8000/api/v1/predict/batch \
  -H 'Content-Type: application/json' \
  -d '{"texts":["Hello","WIN FREE CASH","Meeting tomorrow"]}'
```
