"""
Configuration settings for the FastAPI application.
"""
from pydantic_settings import BaseSettings
from typing import List
from pathlib import Path


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # API Settings
    API_TITLE: str = "Spam Detection API"
    API_VERSION: str = "1.0.0"
    API_DESCRIPTION: str = "AI-powered spam detection service"
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    
    # Model Paths (relative to backend directory)
    MODEL_PATH: str = "models/logreg.joblib"
    VECTORIZER_PATH: str = "artifacts/spam_vectorizer.joblib"
    SCALER_PATH: str = "artifacts/spam_scaler.joblib"
    
    # CORS Settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://localhost:8000",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:8000",
    ]
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Validation Limits
    MAX_TEXT_LENGTH: int = 10000
    MAX_BATCH_SIZE: int = 100
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
