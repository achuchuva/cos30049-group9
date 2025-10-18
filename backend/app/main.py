"""
FastAPI Backend for Spam Detection AI Model
Provides RESTful API endpoints for spam classification with comprehensive error handling.
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
from datetime import datetime

from .models import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ModelInfo
)
from .predictor import SpamPredictor
from .config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global predictor instance
predictor: SpamPredictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for loading/unloading ML models."""
    global predictor
    try:
        logger.info("Loading spam detection model...")
        predictor = SpamPredictor(
            model_path=settings.MODEL_PATH,
            vectorizer_path=settings.VECTORIZER_PATH,
            scaler_path=settings.SCALER_PATH
        )
        logger.info("Model loaded successfully!")
        yield
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    finally:
        logger.info("Shutting down application...")
        predictor = None


# Initialize FastAPI app
app = FastAPI(
    title="Spam Detection API",
    description="AI-powered spam detection service using Logistic Regression",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unexpected errors."""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred. Please try again later.",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.get("/", response_model=dict)
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Spam Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict": "/api/v1/predict",
            "batch_predict": "/api/v1/predict/batch",
            "model_info": "/api/v1/model/info",
            "docs": "/docs"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify API and model availability.
    
    Returns:
        HealthResponse: Current health status of the service
    """
    try:
        model_loaded = predictor is not None and predictor.is_loaded()
        
        return HealthResponse(
            status="healthy" if model_loaded else "unhealthy",
            model_loaded=model_loaded,
            timestamp=datetime.utcnow().isoformat(),
            version="1.0.0"
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Service is unavailable"
        )


@app.get("/api/v1/model/info", response_model=ModelInfo)
async def get_model_info():
    """
    Get information about the loaded model.
    
    Returns:
        ModelInfo: Details about the current model
    """
    try:
        if predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded"
            )
        
        info = predictor.get_model_info()
        return ModelInfo(**info)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve model information: {str(e)}"
        )


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict_spam(request: PredictionRequest):
    """
    Predict whether a single text is spam or ham.
    
    Args:
        request: PredictionRequest containing the text to classify
        
    Returns:
        PredictionResponse: Classification result with confidence score
        
    Raises:
        HTTPException: If prediction fails or model is not loaded
    """
    try:
        if predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please contact the administrator."
            )
        
        # Validate input
        if not request.text or not request.text.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text cannot be empty"
            )
        
        if len(request.text) > 10000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Text is too long. Maximum length is 10,000 characters."
            )
        
        # Make prediction
        result = predictor.predict(request.text)
        
        return PredictionResponse(
            text=request.text[:100] + "..." if len(request.text) > 100 else request.text,
            prediction=result["label"],
            confidence=result["confidence"],
            is_spam=result["is_spam"],
            spam_probability=result["spam_probability"],
            ham_probability=result["ham_probability"],
            timestamp=datetime.utcnow().isoformat(),
            model_name=result.get("model_name", "Logistic Regression")
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/api/v1/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict spam for multiple texts in a single request.
    
    Args:
        request: BatchPredictionRequest containing multiple texts
        
    Returns:
        BatchPredictionResponse: List of predictions for each text
        
    Raises:
        HTTPException: If prediction fails or model is not loaded
    """
    try:
        if predictor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model not loaded. Please contact the administrator."
            )
        
        # Validate input
        if not request.texts or len(request.texts) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No texts provided"
            )
        
        if len(request.texts) > 100:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Too many texts. Maximum batch size is 100."
            )
        
        # Make predictions
        predictions = []
        for idx, text in enumerate(request.texts):
            if not text or not text.strip():
                predictions.append({
                    "text": "",
                    "prediction": "error",
                    "confidence": 0.0,
                    "is_spam": False,
                    "spam_probability": 0.0,
                    "ham_probability": 0.0,
                    "error": "Empty text",
                    "timestamp": datetime.utcnow().isoformat(),
                    "model_name": "Logistic Regression"
                })
                continue
            
            try:
                result = predictor.predict(text)
                predictions.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "prediction": result["label"],
                    "confidence": result["confidence"],
                    "is_spam": result["is_spam"],
                    "spam_probability": result["spam_probability"],
                    "ham_probability": result["ham_probability"],
                    "timestamp": datetime.utcnow().isoformat(),
                    "model_name": result.get("model_name", "Logistic Regression")
                })
            except Exception as e:
                logger.error(f"Error predicting text {idx}: {e}")
                predictions.append({
                    "text": text[:100] + "..." if len(text) > 100 else text,
                    "prediction": "error",
                    "confidence": 0.0,
                    "is_spam": False,
                    "spam_probability": 0.0,
                    "ham_probability": 0.0,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat(),
                    "model_name": "Logistic Regression"
                })
        
        return BatchPredictionResponse(
            predictions=predictions,
            total=len(predictions),
            timestamp=datetime.utcnow().isoformat()
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )


@app.put("/api/v1/model/reload")
async def reload_model():
    """
    Reload the machine learning model.
    Useful for updating the model without restarting the server.
    
    Returns:
        dict: Status message
        
    Raises:
        HTTPException: If model reload fails
    """
    global predictor
    try:
        logger.info("Reloading model...")
        predictor = SpamPredictor(
            model_path=settings.MODEL_PATH,
            vectorizer_path=settings.VECTORIZER_PATH,
            scaler_path=settings.SCALER_PATH
        )
        logger.info("Model reloaded successfully!")
        
        return {
            "message": "Model reloaded successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model reload failed: {str(e)}"
        )


@app.delete("/api/v1/cache/clear")
async def clear_cache():
    """
    Clear any internal caches (placeholder for future caching implementation).
    
    Returns:
        dict: Status message
    """
    try:
        # This is a placeholder for future caching implementation
        logger.info("Cache clear requested")
        
        return {
            "message": "Cache cleared successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cache clear failed: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
