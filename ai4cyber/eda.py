"""
Exploratory Data Analysis (EDA) utilities.
Generates figures saved under reports/data_figures.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from data_processing import load_spam_dataset, TEXT_COL, TARGET_COL

FIG_DIR = Path("reports/data_figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def class_distribution(df: pd.DataFrame, target_col: str = TARGET_COL):
    """
    Plot and save class distribution bar chart.
    Args:
        df: pandas dataframe with target_col
    """
    counts = df[target_col].value_counts().sort_index()
    plt.figure(figsize=(4,4))
    counts.plot(kind="bar", color=["steelblue", "salmon"])
    plt.title("Class Distribution (0=Ham,1=Spam)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_distribution.png", dpi=150)
    plt.close()


def message_length_distribution(df: pd.DataFrame, text_col: str = TEXT_COL):
    """
    Plot and save message length distribution histogram.
    Args:
        df: pandas dataframe with text_col
    """
    lengths = df[text_col].str.split().apply(len)
    plt.figure(figsize=(6,4))
    plt.hist(lengths, bins=50, color="slateblue", alpha=0.7)
    plt.title("Message Length Distribution (tokens)")
    plt.xlabel("Length")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "length_distribution.png", dpi=150)
    plt.close()


def top_ngrams(df: pd.DataFrame, n: int = 20, ngram_range: Tuple[int,int]=(1,2), text_col: str = TEXT_COL):
    """
    Plot and save top n-grams bar chart.
    Args:
        df: pandas dataframe with text_col
        n: number of top n-grams to display
        ngram_range: n-gram range for CountVectorizer
    """
    vec = CountVectorizer(stop_words="english", ngram_range=ngram_range, max_features=5000)
    X = vec.fit_transform(df[text_col])
    freqs = X.sum(axis=0).A1
    vocab = vec.get_feature_names_out()
    pairs = list(zip(vocab, freqs))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:n]
    words, counts = zip(*top)
    plt.figure(figsize=(8,4))
    plt.barh(words[::-1], counts[::-1], color="teal")
    plt.title("Top n-grams")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "top_ngrams.png", dpi=150)
    plt.close()


def run_eda(csv_path: str = "data/emails.csv"):
    df = load_spam_dataset(csv_path)
    class_distribution(df)
    message_length_distribution(df)
    top_ngrams(df)
    print(f"EDA figures saved to {FIG_DIR}")


if __name__ == "__main__":
    run_eda()
