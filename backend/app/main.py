"""
FastAPI Backend for Spam Detection AI Model
Provides RESTful API endpoints for spam classification with comprehensive error handling.
"""
from fastapi import FastAPI, HTTPException, status, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from contextlib import asynccontextmanager
import logging
from datetime import datetime
import csv
import io
import json

from .models import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ModelInfo,
    FileUploadResponse,
    HistoryResponse,
    StatsResponse,
    TextFeatures,
    EmailDetectionsResponse,
    EmailMonitorStats
)
from .predictor import SpamPredictor
from .config import settings
from .exceptions import (
    ModelNotLoadedError,
    EmptyTextError,
    TextTooLongError
)
from .file_parser import extract_text_from_file
from . import database
from .email_monitor import EmailMonitor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global predictor and email monitor instances
predictor: SpamPredictor = None
email_monitor: EmailMonitor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for loading/unloading ML models and initializing database."""
    global predictor, email_monitor
    try:
        logger.info("Initializing database...")
        database.init_database()
        logger.info("Database initialized successfully!")
        
        logger.info("Loading spam detection model...")
        predictor = SpamPredictor(
            model_path=settings.MODEL_PATH,
            vectorizer_path=settings.VECTORIZER_PATH,
            scaler_path=settings.SCALER_PATH
        )
        logger.info("Model loaded successfully!")
        
        logger.info("Initializing email monitor...")
        email_monitor = EmailMonitor(predictor)
        await email_monitor.start()
        logger.info("Email monitor initialized!")
        
        yield
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    finally:
        logger.info("Shutting down application...")
        if email_monitor:
            await email_monitor.stop()
        predictor = None
        email_monitor = None


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
            "predict_file": "/api/v1/predict/file",
            "history": "/api/v1/history",
            "stats": "/api/v1/stats",
            "export": "/api/v1/export/{format}",
            "model_info": "/api/v1/model/info",
            "email_stats": "/api/v1/email/stats",
            "email_predictions": "/api/v1/email/predictions",
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
        PredictionResponse: Classification result with confidence score and features
        
    Raises:
        HTTPException: If prediction fails or model is not loaded
    """
    try:
        if predictor is None:
            raise ModelNotLoadedError()
        
        if not request.text or not request.text.strip():
            raise EmptyTextError()
        
        if len(request.text) > 10000:
            raise TextTooLongError(len(request.text))
        
        result = predictor.predict(request.text)
        numerical_features = predictor.extract_numerical_features(request.text)
        
        timestamp = datetime.utcnow().isoformat()
        text_preview = request.text[:200] + "..." if len(request.text) > 200 else request.text
        
        prediction_id = database.save_prediction(
            text_preview=text_preview,
            prediction=result["label"],
            confidence=result["confidence"],
            is_spam=result["is_spam"],
            spam_probability=result["spam_probability"],
            ham_probability=result["ham_probability"],
            features=numerical_features,
            source_type="text",
            timestamp=timestamp
        )
        
        return PredictionResponse(
            text=text_preview,
            prediction=result["label"],
            confidence=result["confidence"],
            is_spam=result["is_spam"],
            spam_probability=result["spam_probability"],
            ham_probability=result["ham_probability"],
            features=TextFeatures(**numerical_features),
            timestamp=timestamp,
            model_name=result.get("model_name", "Logistic Regression"),
            prediction_id=prediction_id
        )
    
    except (ModelNotLoadedError, EmptyTextError, TextTooLongError):
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )


@app.post("/api/v1/predict/file", response_model=FileUploadResponse)
async def predict_file(file: UploadFile = File(...)):
    """
    Predict spam for text extracted from uploaded file.
    
    Args:
        file: Uploaded file (TXT, PDF, or DOCX)
        
    Returns:
        FileUploadResponse: File info and prediction result
        
    Raises:
        HTTPException: If file processing or prediction fails
    """
    try:
        if predictor is None:
            raise ModelNotLoadedError()
        
        content = await file.read()
        
        extracted_text = extract_text_from_file(content, file.filename)
        
        if not extracted_text or not extracted_text.strip():
            raise EmptyTextError()
        
        if len(extracted_text) > 50000:
            extracted_text = extracted_text[:50000]
        
        result = predictor.predict(extracted_text)
        numerical_features = predictor.extract_numerical_features(extracted_text)
        
        timestamp = datetime.utcnow().isoformat()
        text_preview = extracted_text[:200] + "..." if len(extracted_text) > 200 else extracted_text
        
        prediction_id = database.save_prediction(
            text_preview=text_preview,
            prediction=result["label"],
            confidence=result["confidence"],
            is_spam=result["is_spam"],
            spam_probability=result["spam_probability"],
            ham_probability=result["ham_probability"],
            features=numerical_features,
            source_type="file",
            filename=file.filename,
            timestamp=timestamp
        )
        
        prediction_response = PredictionResponse(
            text=text_preview,
            prediction=result["label"],
            confidence=result["confidence"],
            is_spam=result["is_spam"],
            spam_probability=result["spam_probability"],
            ham_probability=result["ham_probability"],
            features=TextFeatures(**numerical_features),
            timestamp=timestamp,
            model_name=result.get("model_name", "Logistic Regression"),
            prediction_id=prediction_id
        )
        
        return FileUploadResponse(
            filename=file.filename,
            file_size_bytes=len(content),
            extracted_text_length=len(extracted_text),
            prediction_result=prediction_response
        )
    
    except (ModelNotLoadedError, EmptyTextError):
        raise
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File prediction error: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File prediction failed: {str(e)}"
        )


@app.get("/api/v1/history", response_model=HistoryResponse)
async def get_history(limit: int = 50, offset: int = 0):
    """
    Get prediction history.
    
    Args:
        limit: Maximum number of predictions to return (default 50)
        offset: Number of predictions to skip (default 0)
        
    Returns:
        HistoryResponse: List of predictions with pagination info
    """
    try:
        if limit < 1 or limit > 200:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Limit must be between 1 and 200"
            )
        
        if offset < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Offset must be non-negative"
            )
        
        predictions = database.get_predictions(limit=limit, offset=offset)
        
        return HistoryResponse(
            predictions=predictions,
            total=len(predictions),
            limit=limit,
            offset=offset
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve history: {str(e)}"
        )


@app.delete("/api/v1/history/{prediction_id}")
async def delete_history_item(prediction_id: int):
    """
    Delete a prediction from history.
    
    Args:
        prediction_id: ID of the prediction to delete
        
    Returns:
        dict: Status message
    """
    try:
        deleted = database.delete_prediction(prediction_id)
        
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Prediction with ID {prediction_id} not found"
            )
        
        return {
            "message": f"Prediction {prediction_id} deleted successfully",
            "timestamp": datetime.utcnow().isoformat()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete prediction {prediction_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete prediction: {str(e)}"
        )


@app.get("/api/v1/stats", response_model=StatsResponse)
async def get_stats():
    """
    Get aggregated statistics and data for visualizations.
    
    Returns:
        StatsResponse: Statistics including time-series, distributions, and aggregates
    """
    try:
        stats = database.get_statistics()
        return StatsResponse(**stats)
    
    except Exception as e:
        logger.error(f"Failed to retrieve statistics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve statistics: {str(e)}"
        )


@app.get("/api/v1/export/{format}")
async def export_predictions(format: str):
    """
    Export all predictions in specified format.
    
    Args:
        format: Export format ('csv' or 'json')
        
    Returns:
        StreamingResponse: File download with predictions
    """
    try:
        if format not in ["csv", "json"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Format must be 'csv' or 'json'"
            )
        
        predictions = database.get_all_predictions_for_export()
        
        if format == "csv":
            output = io.StringIO()
            if predictions:
                fieldnames = predictions[0].keys()
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(predictions)
            
            output.seek(0)
            return StreamingResponse(
                iter([output.getvalue()]),
                media_type="text/csv",
                headers={"Content-Disposition": "attachment; filename=predictions.csv"}
            )
        
        else:
            output = json.dumps(predictions, indent=2)
            return StreamingResponse(
                iter([output]),
                media_type="application/json",
                headers={"Content-Disposition": "attachment; filename=predictions.json"}
            )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export predictions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export predictions: {str(e)}"
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


@app.get("/api/v1/email/detections", response_model=EmailDetectionsResponse)
async def get_email_detections(limit: int = 50):
    """
    Get recent email spam detections from email monitoring.
    
    Args:
        limit: Maximum number of detections to return (default 50)
        
    Returns:
        EmailDetectionsResponse: List of email detections
    """
    try:
        if email_monitor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Email monitor not initialized"
            )
        
        detections = email_monitor.get_recent_detections(limit=limit)
        
        return EmailDetectionsResponse(
            detections=detections,
            total=len(detections)
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve email detections: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve email detections: {str(e)}"
        )


@app.get("/api/v1/email/predictions", response_model=HistoryResponse)
async def get_email_predictions_history(limit: int = 50, offset: int = 0):
    """
    Get email prediction history.
    
    Args:
        limit: Maximum number of predictions to return (default 50)
        offset: Number of predictions to skip (default 0)
        
    Returns:
        HistoryResponse: List of email predictions with pagination info
    """
    try:
        if limit < 1 or limit > 200:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Limit must be between 1 and 200"
            )
        
        if offset < 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Offset must be non-negative"
            )
        
        predictions = database.get_email_predictions(limit=limit, offset=offset)
        
        return HistoryResponse(
            predictions=predictions,
            total=len(predictions),
            limit=limit,
            offset=offset
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve email prediction history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve email prediction history: {str(e)}"
        )


@app.get("/api/v1/email/stats", response_model=EmailMonitorStats)
async def get_email_stats():
    """
    Get email monitoring statistics.
    
    Returns:
        EmailMonitorStats: Email monitoring statistics
    """
    try:
        if email_monitor is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Email monitor not initialized"
            )
        
        stats = email_monitor.get_stats()
        return EmailMonitorStats(**stats)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to retrieve email stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve email stats: {str(e)}"
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
