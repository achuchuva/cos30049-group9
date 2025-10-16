"""
Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime


class PredictionRequest(BaseModel):
    """Request model for single text prediction."""
    text: str = Field(
        ...,
        description="Text to classify as spam or ham",
        min_length=1,
        max_length=10000,
        example="Congratulations! You've won a $1000 gift card. Click here to claim now!"
    )
    
    @validator('text')
    def text_must_not_be_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Text cannot be empty or only whitespace')
        return v


class PredictionResponse(BaseModel):
    """Response model for single prediction."""
    text: str = Field(..., description="Input text (truncated if too long)")
    prediction: str = Field(..., description="Classification label: 'spam' or 'ham'")
    confidence: float = Field(..., description="Confidence score for the prediction", ge=0.0, le=1.0)
    is_spam: bool = Field(..., description="Boolean flag indicating if text is spam")
    spam_probability: float = Field(..., description="Probability of text being spam", ge=0.0, le=1.0)
    ham_probability: float = Field(..., description="Probability of text being ham", ge=0.0, le=1.0)
    timestamp: str = Field(..., description="Timestamp of prediction")
    model_name: str = Field(..., description="Name of the model used")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Congratulations! You've won a $1000 gift card...",
                "prediction": "spam",
                "confidence": 0.95,
                "is_spam": True,
                "spam_probability": 0.95,
                "ham_probability": 0.05,
                "timestamp": "2025-10-16T10:30:00.000000",
                "model_name": "Logistic Regression"
            }
        }


class BatchPredictionRequest(BaseModel):
    """Request model for batch predictions."""
    texts: List[str] = Field(
        ...,
        description="List of texts to classify",
        min_items=1,
        max_items=100,
        example=[
            "Hello, how are you?",
            "WIN FREE CASH NOW!!!",
            "Meeting scheduled for 3pm tomorrow"
        ]
    )


class SingleBatchPrediction(BaseModel):
    """Single prediction within a batch response."""
    text: str
    prediction: str
    confidence: float
    is_spam: bool
    spam_probability: float
    ham_probability: float
    timestamp: str
    model_name: str
    error: Optional[str] = None


class BatchPredictionResponse(BaseModel):
    """Response model for batch predictions."""
    predictions: List[dict] = Field(..., description="List of predictions")
    total: int = Field(..., description="Total number of predictions")
    timestamp: str = Field(..., description="Timestamp of batch prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "predictions": [
                    {
                        "text": "Hello, how are you?",
                        "prediction": "ham",
                        "confidence": 0.92,
                        "is_spam": False,
                        "spam_probability": 0.08,
                        "ham_probability": 0.92,
                        "timestamp": "2025-10-16T10:30:00.000000",
                        "model_name": "Logistic Regression"
                    }
                ],
                "total": 1,
                "timestamp": "2025-10-16T10:30:00.000000"
            }
        }


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str = Field(..., description="Service status: 'healthy' or 'unhealthy'")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    timestamp: str = Field(..., description="Current timestamp")
    version: str = Field(..., description="API version")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "model_loaded": True,
                "timestamp": "2025-10-16T10:30:00.000000",
                "version": "1.0.0"
            }
        }


class ModelInfo(BaseModel):
    """Response model for model information."""
    model_name: str = Field(..., description="Name of the model")
    model_type: str = Field(..., description="Type of model")
    features: dict = Field(..., description="Feature information")
    loaded_at: str = Field(..., description="Timestamp when model was loaded")
    
    class Config:
        json_schema_extra = {
            "example": {
                "model_name": "Logistic Regression",
                "model_type": "LogisticRegression",
                "features": {
                    "text_features": 5000,
                    "numerical_features": 5,
                    "total_features": 5005
                },
                "loaded_at": "2025-10-16T10:00:00.000000"
            }
        }


class ErrorResponse(BaseModel):
    """Response model for errors."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    timestamp: str = Field(..., description="Timestamp of error")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValidationError",
                "message": "Text cannot be empty",
                "timestamp": "2025-10-16T10:30:00.000000"
            }
        }
