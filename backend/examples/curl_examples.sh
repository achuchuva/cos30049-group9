"""
cURL examples for testing the Spam Detection API.
"""

# Health Check
curl -X GET "http://localhost:8000/health"

# Get Model Info
curl -X GET "http://localhost:8000/api/v1/model/info"

# Single Prediction - Spam Example
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You won $1000. Click here NOW!"}'

# Single Prediction - Ham Example
curl -X POST "http://localhost:8000/api/v1/predict" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello! Hope you are having a great day."}'

# Batch Prediction
curl -X POST "http://localhost:8000/api/v1/predict/batch" \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Hello, how are you?",
      "WIN FREE CASH NOW!!!",
      "Meeting at 3pm tomorrow"
    ]
  }'

# Reload Model
curl -X PUT "http://localhost:8000/api/v1/model/reload"

# Clear Cache
curl -X DELETE "http://localhost:8000/api/v1/cache/clear"
