from __future__ import annotations
import re
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

DEFAULT_RANDOM_STATE = 42
ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)

TEXT_COL = "text"
TARGET_COL = "spam"

_punct_trans = str.maketrans("", "", string.punctuation)


def clean_text(text: str) -> str:
    """
    Clean text: lowercase, remove punctuation, digits, extra whitespace.
    Args:
        text: raw text string
    Returns:
        cleaned text string
    """

    text = text.lower()
    # remove email headers like 'subject:' at start
    text = re.sub(r"^subject:\s*", "", text)
    # remove urls
    text = re.sub(r"https?://\S+", " URL ", text)
    # remove digits
    text = re.sub(r"\d+", " ", text)
    # remove punctuation
    text = text.translate(_punct_trans)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_spam_dataset(csv_path: str | Path) -> pd.DataFrame:
    """
    Load and clean dataset from CSV file.
    Expects columns: 'text', 'spam'.
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
    # drop duplicates
    df = df.drop_duplicates(subset=[TEXT_COL])
    # drop NA spam column rows
    df = df.dropna(subset=[TEXT_COL, TARGET_COL])
    # clean text
    df[TEXT_COL] = df[TEXT_COL].astype(str).apply(clean_text)
    return df


@dataclass
class ProcessedData:
    X_train: any
    X_test: any
    y_train: pd.Series
    y_test: pd.Series
    vectorizer: TfidfVectorizer


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
    Preprocess the dataset: split into train/test, vectorize text.
    Args:
        df: input dataframe with 'text' and 'spam' columns
        test_size: proportion of test set (0,1)
        random_state: random seed for reproducibility
        max_features: maximum number of features for TF-IDF
        ngram_range: n-gram range for TF-IDF
        min_df: minimum document frequency for terms
        max_df: maximum document frequency for terms (proportion)
    Returns:
        ProcessedData dataclass with train/test splits and vectorizer
    """
    X = df[TEXT_COL].values
    y = df[TARGET_COL].astype(int)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        sublinear_tf=True,
        min_df=min_df,
        max_df=max_df,
    )
    X_train = vectorizer.fit_transform(X_train_text)
    X_test = vectorizer.transform(X_test_text)

    return ProcessedData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, vectorizer=vectorizer)


def save_artifacts(processed: ProcessedData, prefix: str = "spam") -> None:
    """
    Save vectorizer and datasets to disk using joblib.
    joblib provides efficient serialization for large numpy arrays.
    Args:
        processed: ProcessedData dataclass
        prefix: prefix for saved filenames
    """
    joblib.dump(processed.vectorizer, ARTIFACTS_DIR / f"{prefix}_vectorizer.joblib")
    joblib.dump((processed.X_train, processed.y_train), ARTIFACTS_DIR / f"{prefix}_train.joblib")
    joblib.dump((processed.X_test, processed.y_test), ARTIFACTS_DIR / f"{prefix}_test.joblib")


def load_artifacts(prefix: str = "spam") -> ProcessedData:
    """
    Load vectorizer and datasets from disk.
    Args:
        prefix: prefix for saved filenames
    Returns:
        ProcessedData dataclass
    """
    vectorizer: TfidfVectorizer = joblib.load(ARTIFACTS_DIR / f"{prefix}_vectorizer.joblib")
    X_train, y_train = joblib.load(ARTIFACTS_DIR / f"{prefix}_train.joblib")
    X_test, y_test = joblib.load(ARTIFACTS_DIR / f"{prefix}_test.joblib")
    return ProcessedData(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, vectorizer=vectorizer)

def preprocess_data(csv_path: str = "data/spam.csv"):
    df = load_spam_dataset(csv_path)
    processed = preprocess(df)
    save_artifacts(processed)
    print("Artifacts saved in ./artifacts")

if __name__ == "__main__":  # simple manual run
    preprocess_data()
