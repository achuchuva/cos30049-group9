"""
Pydantic models for request/response validation.
"""
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
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


class TextFeatures(BaseModel):
    """Text feature extraction results."""
    char_count: int = Field(..., description="Number of characters in text")
    word_count: int = Field(..., description="Number of words in text")
    suspicious_word_count: int = Field(..., description="Number of suspicious/spam words")
    url_count: int = Field(..., description="Number of URLs in text")
    url_digit_count: int = Field(..., description="Number of digits in URLs")


class PredictionResponse(BaseModel):
    """Response model for single prediction."""
    text: str = Field(..., description="Input text (truncated if too long)")
    prediction: str = Field(..., description="Classification label: 'spam' or 'ham'")
    confidence: float = Field(..., description="Confidence score for the prediction", ge=0.0, le=1.0)
    is_spam: bool = Field(..., description="Boolean flag indicating if text is spam")
    spam_probability: float = Field(..., description="Probability of text being spam", ge=0.0, le=1.0)
    ham_probability: float = Field(..., description="Probability of text being ham", ge=0.0, le=1.0)
    features: TextFeatures = Field(..., description="Extracted text features")
    timestamp: str = Field(..., description="Timestamp of prediction")
    model_name: str = Field(..., description="Name of the model used")
    prediction_id: Optional[int] = Field(None, description="Database ID of saved prediction")
    
    class Config:
        json_schema_extra = {
            "example": {
                "text": "Congratulations! You've won a $1000 gift card...",
                "prediction": "spam",
                "confidence": 0.95,
                "is_spam": True,
                "spam_probability": 0.95,
                "ham_probability": 0.05,
                "features": {
                    "char_count": 150,
                    "word_count": 25,
                    "suspicious_word_count": 3,
                    "url_count": 1,
                    "url_digit_count": 5
                },
                "timestamp": "2025-10-16T10:30:00.000000",
                "model_name": "Logistic Regression",
                "prediction_id": 123
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


class FileUploadResponse(BaseModel):
    """Response model for file upload predictions."""
    filename: str = Field(..., description="Name of uploaded file")
    file_size_bytes: int = Field(..., description="Size of file in bytes")
    extracted_text_length: int = Field(..., description="Length of extracted text")
    prediction_result: PredictionResponse = Field(..., description="Prediction result")


class HistoryResponse(BaseModel):
    """Response model for prediction history."""
    predictions: List[Dict[str, Any]] = Field(..., description="List of predictions")
    total: int = Field(..., description="Total count of predictions")
    limit: int = Field(..., description="Limit used in query")
    offset: int = Field(..., description="Offset used in query")


class StatsResponse(BaseModel):
    """Response model for statistics."""
    total_predictions: int
    spam_count: int
    ham_count: int
    spam_rate: float
    avg_confidence: float
    feature_averages: Dict[str, float]
    time_series: List[Dict[str, Any]]
    confidence_distribution: List[Dict[str, Any]]
    feature_distribution: List[Dict[str, Any]]

