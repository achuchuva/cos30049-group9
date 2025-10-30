"""
Database operations for prediction history storage.
"""
import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "predictions.db"


@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_database() -> None:
    """Initialize the database with required tables."""
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    text_preview TEXT NOT NULL,
                    prediction TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    is_spam INTEGER NOT NULL,
                    spam_probability REAL NOT NULL,
                    ham_probability REAL NOT NULL,
                    char_count INTEGER NOT NULL,
                    word_count INTEGER NOT NULL,
                    suspicious_word_count INTEGER NOT NULL,
                    url_count INTEGER NOT NULL,
                    url_digit_count INTEGER NOT NULL,
                    source_type TEXT NOT NULL,
                    filename TEXT,
                    timestamp TEXT NOT NULL,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_timestamp ON predictions(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_prediction ON predictions(prediction)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON predictions(created_at)
            """)
            logger.info(f"Database initialized at {DB_PATH}")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise


def save_prediction(
    text_preview: str,
    prediction: str,
    confidence: float,
    is_spam: bool,
    spam_probability: float,
    ham_probability: float,
    features: Dict[str, int],
    source_type: str = "text",
    filename: Optional[str] = None,
    timestamp: Optional[str] = None
) -> int:
    """
    Save a prediction to the database.
    
    Args:
        text_preview: Preview of the input text (truncated)
        prediction: Classification label ('spam' or 'ham')
        confidence: Confidence score
        is_spam: Boolean flag
        spam_probability: Probability of spam
        ham_probability: Probability of ham
        features: Dictionary with text features
        source_type: Source type ('text' or 'file')
        filename: Original filename if source is file
        timestamp: ISO timestamp
        
    Returns:
        ID of the inserted record
    """
    try:
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat()
        
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO predictions (
                    text_preview, prediction, confidence, is_spam,
                    spam_probability, ham_probability,
                    char_count, word_count, suspicious_word_count,
                    url_count, url_digit_count,
                    source_type, filename, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                text_preview,
                prediction,
                confidence,
                1 if is_spam else 0,
                spam_probability,
                ham_probability,
                features.get("char_count", 0),
                features.get("word_count", 0),
                features.get("suspicious_word_count", 0),
                features.get("url_count", 0),
                features.get("url_digit_count", 0),
                source_type,
                filename,
                timestamp
            ))
            return cursor.lastrowid
    except Exception as e:
        logger.error(f"Failed to save prediction: {e}")
        raise


def get_predictions(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    """
    Retrieve predictions from the database.
    
    Args:
        limit: Maximum number of records to retrieve
        offset: Number of records to skip
        
    Returns:
        List of prediction dictionaries
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT * FROM predictions
                ORDER BY created_at DESC
                LIMIT ? OFFSET ?
            """, (limit, offset))
            
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to retrieve predictions: {e}")
        raise


def get_prediction_by_id(prediction_id: int) -> Optional[Dict[str, Any]]:
    """
    Retrieve a single prediction by ID.
    
    Args:
        prediction_id: ID of the prediction
        
    Returns:
        Prediction dictionary or None if not found
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions WHERE id = ?", (prediction_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
    except Exception as e:
        logger.error(f"Failed to retrieve prediction {prediction_id}: {e}")
        raise


def delete_prediction(prediction_id: int) -> bool:
    """
    Delete a prediction from the database.
    
    Args:
        prediction_id: ID of the prediction to delete
        
    Returns:
        True if deleted, False if not found
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM predictions WHERE id = ?", (prediction_id,))
            return cursor.rowcount > 0
    except Exception as e:
        logger.error(f"Failed to delete prediction {prediction_id}: {e}")
        raise


def get_statistics() -> Dict[str, Any]:
    """
    Get aggregated statistics from predictions.
    
    Returns:
        Dictionary with statistics
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("SELECT COUNT(*) as total FROM predictions")
            total = cursor.fetchone()["total"]
            
            cursor.execute("SELECT COUNT(*) as spam_count FROM predictions WHERE is_spam = 1")
            spam_count = cursor.fetchone()["spam_count"]
            
            cursor.execute("SELECT AVG(confidence) as avg_confidence FROM predictions")
            avg_confidence = cursor.fetchone()["avg_confidence"] or 0.0
            
            cursor.execute("""
                SELECT 
                    AVG(char_count) as avg_char_count,
                    AVG(word_count) as avg_word_count,
                    AVG(suspicious_word_count) as avg_suspicious_words,
                    AVG(url_count) as avg_url_count
                FROM predictions
            """)
            feature_avgs = cursor.fetchone()
            
            cursor.execute("""
                SELECT 
                    strftime('%Y-%m-%d %H:00:00', created_at) as hour,
                    COUNT(*) as count,
                    SUM(CASE WHEN is_spam = 1 THEN 1 ELSE 0 END) as spam_count
                FROM predictions
                GROUP BY hour
                ORDER BY hour DESC
                LIMIT 24
            """)
            time_series = cursor.fetchall()
            
            cursor.execute("""
                SELECT confidence, prediction
                FROM predictions
                ORDER BY created_at DESC
                LIMIT 100
            """)
            confidence_dist = cursor.fetchall()
            
            cursor.execute("""
                SELECT 
                    char_count, word_count, suspicious_word_count, 
                    url_count, is_spam
                FROM predictions
                ORDER BY created_at DESC
                LIMIT 100
            """)
            feature_data = cursor.fetchall()
            
            return {
                "total_predictions": total,
                "spam_count": spam_count,
                "ham_count": total - spam_count,
                "spam_rate": spam_count / total if total > 0 else 0,
                "avg_confidence": round(avg_confidence, 4),
                "feature_averages": {
                    "avg_char_count": round(feature_avgs["avg_char_count"] or 0, 2),
                    "avg_word_count": round(feature_avgs["avg_word_count"] or 0, 2),
                    "avg_suspicious_words": round(feature_avgs["avg_suspicious_words"] or 0, 2),
                    "avg_url_count": round(feature_avgs["avg_url_count"] or 0, 2),
                },
                "time_series": [
                    {
                        "hour": row["hour"],
                        "total": row["count"],
                        "spam_count": row["spam_count"],
                        "spam_rate": row["spam_count"] / row["count"] if row["count"] > 0 else 0
                    }
                    for row in time_series
                ],
                "confidence_distribution": [
                    {"confidence": row["confidence"], "prediction": row["prediction"]}
                    for row in confidence_dist
                ],
                "feature_distribution": [
                    {
                        "char_count": row["char_count"],
                        "word_count": row["word_count"],
                        "suspicious_word_count": row["suspicious_word_count"],
                        "url_count": row["url_count"],
                        "is_spam": bool(row["is_spam"])
                    }
                    for row in feature_data
                ]
            }
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        raise


def get_all_predictions_for_export() -> List[Dict[str, Any]]:
    """
    Get all predictions for export.
    
    Returns:
        List of all prediction dictionaries
    """
    try:
        with get_db_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions ORDER BY created_at DESC")
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Failed to export predictions: {e}")
        raise
