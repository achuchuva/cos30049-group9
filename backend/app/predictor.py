"""
Spam predictor module for making predictions using the trained model.
"""
import re
import string
from pathlib import Path
from typing import Dict, Any
from datetime import datetime

import joblib
import pandas as pd
from scipy.sparse import hstack


# Constants from original data_processing module
NUMERICAL_COLS = [
    "char_count",
    "word_count",
    "suspicious_word_count",
    "url_count",
    "url_digit_count",
]

SUSPICIOUS_WORDS = [
    'free', 'win', 'winner', 'cash', 'prize', 'urgent', 'apply now', 'buy',
    'subscribe', 'click', 'limited time', 'offer', 'money', 'credit', 'loan',
    'investment', 'pharmacy', 'viagra', 'sex', 'hot', 'deal', 'now',
    'guaranteed', 'congratulations', 'won', 'claim', 'unlimited', 'certified',
    'extra', 'income', 'earn', 'per', 'week', 'work', 'from', 'home', 'opportunity',
    'exclusive', 'amazing', 'selected', 'special', 'promotion', 'bonus',
    'not', 'spam', 'unsubscribe', 'opt-out', 'dear', 'friend', '$'
]

_punct_trans = str.maketrans("", "", string.punctuation)


class SpamPredictor:
    """Handles spam prediction with preprocessing and postprocessing."""
    
    def __init__(self, model_path: str, vectorizer_path: str, scaler_path: str):
        """
        Initialize the predictor with model and preprocessing artifacts.
        
        Args:
            model_path: Path to the trained model file
            vectorizer_path: Path to the TF-IDF vectorizer
            scaler_path: Path to the feature scaler
        """
        self.model_path = Path(model_path)
        self.vectorizer_path = Path(vectorizer_path)
        self.scaler_path = Path(scaler_path)
        
        self.model = None
        self.vectorizer = None
        self.scaler = None
        self.loaded_at = None
        
        self._load_artifacts()
    
    def _load_artifacts(self):
        """Load model and preprocessing artifacts."""
        try:
            # Load vectorizer
            if not self.vectorizer_path.exists():
                raise FileNotFoundError(f"Vectorizer not found at {self.vectorizer_path}")
            self.vectorizer = joblib.load(self.vectorizer_path)
            
            # Load scaler
            if not self.scaler_path.exists():
                raise FileNotFoundError(f"Scaler not found at {self.scaler_path}")
            self.scaler = joblib.load(self.scaler_path)
            
            # Load model
            if not self.model_path.exists():
                raise FileNotFoundError(f"Model not found at {self.model_path}")
            self.model = joblib.load(self.model_path)
            
            self.loaded_at = datetime.utcnow().isoformat()
            
        except Exception as e:
            raise RuntimeError(f"Failed to load model artifacts: {e}")
    
    def is_loaded(self) -> bool:
        """Check if all artifacts are loaded."""
        return all([
            self.model is not None,
            self.vectorizer is not None,
            self.scaler is not None
        ])
    
    @staticmethod
    def clean_text(text: str) -> str:
        """
        Clean and preprocess text.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if not isinstance(text, str):
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove email headers like 'subject:' at start
        text = re.sub(r"^subject:\s*", "", text)
        
        # Remove punctuation
        text = text.translate(_punct_trans)
        
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        
        return text
    
    @staticmethod
    def extract_numerical_features(text: str) -> Dict[str, int]:
        """
        Extract numerical features from text.
        
        Args:
            text: Input text string
            
        Returns:
            Dictionary of numerical features
        """
        return {
            "char_count": len(text),
            "word_count": len(text.split()),
            "suspicious_word_count": sum(1 for word in SUSPICIOUS_WORDS if word in text.lower()),
            "url_count": len(re.findall(r"http[s]?://", text)),
            "url_digit_count": sum(c.isdigit() for c in "".join(re.findall(r"http[s]?://\S+", text))),
        }
    
    def preprocess(self, text: str) -> Any:
        """
        Preprocess text for prediction.
        
        Args:
            text: Raw text string
            
        Returns:
            Combined feature matrix ready for prediction
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Extract numerical features
        numerical_features = self.extract_numerical_features(cleaned_text)
        numerical_df = pd.DataFrame([numerical_features], columns=NUMERICAL_COLS)
        
        # Transform text with vectorizer
        text_vector = self.vectorizer.transform([cleaned_text])
        
        # Scale numerical features
        numerical_vector = self.scaler.transform(numerical_df.values)
        
        # Combine features
        combined_features = hstack([text_vector, numerical_vector])
        
        return combined_features
    
    def postprocess(self, prediction: int, probabilities: list) -> Dict[str, Any]:
        """
        Postprocess model output into a structured response.
        
        Args:
            prediction: Binary prediction (0 or 1)
            probabilities: Class probabilities [ham_prob, spam_prob]
            
        Returns:
            Dictionary with formatted prediction results
        """
        label = "spam" if prediction == 1 else "ham"
        is_spam = prediction == 1
        
        # Get probabilities
        ham_prob = float(probabilities[0])
        spam_prob = float(probabilities[1])
        confidence = spam_prob if is_spam else ham_prob
        
        return {
            "label": label,
            "is_spam": is_spam,
            "confidence": round(confidence, 4),
            "spam_probability": round(spam_prob, 4),
            "ham_probability": round(ham_prob, 4)
        }
    
    def predict(self, text: str) -> Dict[str, Any]:
        """
        Predict whether text is spam or ham.
        
        Args:
            text: Input text to classify
            
        Returns:
            Dictionary containing prediction results
            
        Raises:
            ValueError: If text is empty or invalid
            RuntimeError: If model is not loaded
        """
        if not self.is_loaded():
            raise RuntimeError("Model artifacts not loaded")
        
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
        
        try:
            # Preprocess
            features = self.preprocess(text)
            
            # Predict
            prediction = self.model.predict(features)[0]
            
            # Get probabilities if available
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(features)[0]
            else:
                # Fallback if model doesn't support probabilities
                probabilities = [1.0 - prediction, prediction]
            
            # Postprocess
            result = self.postprocess(prediction, probabilities)
            result["model_name"] = "Logistic Regression"
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        if not self.is_loaded():
            raise RuntimeError("Model artifacts not loaded")
        
        return {
            "model_name": "Logistic Regression",
            "model_type": type(self.model).__name__,
            "features": {
                "text_features": len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, 'vocabulary_') else 0,
                "numerical_features": len(NUMERICAL_COLS),
                "total_features": (len(self.vectorizer.vocabulary_) if hasattr(self.vectorizer, 'vocabulary_') else 0) + len(NUMERICAL_COLS)
            },
            "loaded_at": self.loaded_at
        }
