from __future__ import annotations
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

DEFAULT_RANDOM_STATE = 42
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

TEXT_COL = "text"
TARGET_COL = "spam"
NUMERICAL_COLS = [
    "char_count",
    "word_count",
    "suspicious_word_count",
    "url_count",
    "url_digit_count",
]

_punct_trans = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    """
    Clean text: lowercase, subject: email headers, remove punctuation, and extra whitespace.
    Args:
        text: raw text string
    Returns:
        cleaned text string
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    # remove email headers like 'subject:' at start
    text = re.sub(r"^subject:\s*", "", text)
    # remove punctuation
    text = text.translate(_punct_trans)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_spam_dataset(csv_path: str | Path) -> pd.DataFrame:
    """
    Load and clean dataset from CSV file.
    Expects columns: 'text', 'spam', and numerical feature columns.
    Args:
        csv_path: path to CSV file
    Returns:
        pandas dataframe
    """
    df = pd.read_csv(csv_path)
    # basic validation
    expected_cols = {TEXT_COL, TARGET_COL}
    if not expected_cols.issubset(df.columns):
        raise ValueError(f"Dataset must contain columns {expected_cols}, found {set(df.columns)}")
    
    # Handle missing values
    df[TEXT_COL] = df[TEXT_COL].fillna("")
    for col in NUMERICAL_COLS:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # drop duplicates
    df = df.drop_duplicates(subset=[TEXT_COL])
    # drop NA spam column rows
    df = df.dropna(subset=[TARGET_COL])
    # clean text
    df[TEXT_COL] = df[TEXT_COL].astype(str).apply(clean_text)
    return df


@dataclass
class ProcessedData:
    """Dataclass to hold processed data and artifacts."""
    X_train: any
    X_test: any
    y_train: pd.Series
    y_test: pd.Series
    vectorizer: TfidfVectorizer
    scaler: MinMaxScaler


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = DEFAULT_RANDOM_STATE,
    max_features: int = 5000,
    ngram_range: Tuple[int, int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.95,
) -> ProcessedData:
    """
    Preprocess the dataset: split into train/test, vectorize text, scale numerical features, and combine them.
    Args:
        df: input dataframe with 'text', 'spam' and numerical columns
        test_size: proportion of test set (0,1)
        random_state: random seed for reproducibility
        max_features: maximum number of features for TF-IDF
        ngram_range: n-gram range for TF-IDF
        min_df: minimum document frequency for terms
        max_df: maximum document frequency for terms (proportion)
    Returns:
        ProcessedData dataclass with train/test splits, vectorizer, and scaler
    """
    X_text = df[TEXT_COL].values
    X_numerical = df[NUMERICAL_COLS].values
    y = df[TARGET_COL].astype(int)

    X_train_text, X_test_text, X_train_num, X_test_num, y_train, y_test = train_test_split(
        X_text, X_numerical, y, test_size=test_size, stratify=y, random_state=random_state
    )

    # Vectorize text data
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        sublinear_tf=True,
        min_df=min_df,
        max_df=max_df,
    )
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    # Scale numerical data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_num)
    X_test_scaled = scaler.transform(X_test_num)

    # Combine text and numerical features
    X_train = hstack([X_train_vec, X_train_scaled])
    X_test = hstack([X_test_vec, X_test_scaled])

    return ProcessedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        vectorizer=vectorizer,
        scaler=scaler,
    )


def save_artifacts(processed: ProcessedData, prefix: str = "spam") -> None:
    """
    Save vectorizer, scaler, and datasets to disk using joblib.
    joblib provides efficient serialization for large numpy arrays.
    Args:
        processed: ProcessedData dataclass
        prefix: prefix for saved filenames
    """
    joblib.dump(processed.vectorizer, ARTIFACTS_DIR / f"{prefix}_vectorizer.joblib")
    joblib.dump(processed.scaler, ARTIFACTS_DIR / f"{prefix}_scaler.joblib")
    joblib.dump((processed.X_train, processed.y_train), ARTIFACTS_DIR / f"{prefix}_train.joblib")
    joblib.dump((processed.X_test, processed.y_test), ARTIFACTS_DIR / f"{prefix}_test.joblib")


def load_artifacts(prefix: str = "spam") -> ProcessedData:
    """
    Load vectorizer, scaler, and datasets from disk.
    Args:
        prefix: prefix for saved filenames
    Returns:
        ProcessedData dataclass
    """
    vectorizer: TfidfVectorizer = joblib.load(ARTIFACTS_DIR / f"{prefix}_vectorizer.joblib")
    scaler: MinMaxScaler = joblib.load(ARTIFACTS_DIR / f"{prefix}_scaler.joblib")
    X_train, y_train = joblib.load(ARTIFACTS_DIR / f"{prefix}_train.joblib")
    X_test, y_test = joblib.load(ARTIFACTS_DIR / f"{prefix}_test.joblib")
    return ProcessedData(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        vectorizer=vectorizer,
        scaler=scaler,
    )

def preprocess_data(csv_path: str = "data/spam_featured.csv"):
    df = load_spam_dataset(csv_path)
    processed = preprocess(df)
    save_artifacts(processed)
    print("Artifacts saved in ./artifacts")

if __name__ == "__main__":  # simple manual run
    preprocess_data()
