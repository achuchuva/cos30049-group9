"""
Exploratory Data Analysis (EDA) utilities.
Generates figures saved under reports/data_figures.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

from data_processing import load_spam_dataset, TEXT_COL, TARGET_COL, NUMERICAL_COLS

# List of suspicious words often found in spam
SUSPICIOUS_WORDS = [
    'free', 'win', 'winner', 'cash', 'prize', 'urgent', 'apply now', 'buy',
    'subscribe', 'click', 'limited time', 'offer', 'money', 'credit', 'loan',
    'investment', 'pharmacy', 'viagra', 'sex', 'hot', 'deal', 'now',
    'guaranteed', 'congratulations', 'won', 'claim', 'unlimited', 'certified',
    'extra', 'income', 'earn', 'per', 'week', 'work', 'from', 'home', 'opportunity',
    'exclusive', 'amazing', 'selected', 'special', 'promotion', 'bonus',
    'not', 'spam', 'unsubscribe', 'opt-out', 'dear', 'friend', '$'
]

FIG_DIR = Path("reports/data_figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


def class_distribution(df: pd.DataFrame, target_col: str = TARGET_COL):
    """
    Plot and save class distribution bar chart.
    Args:
        df: pandas dataframe with target_col
    """
    counts = df[target_col].value_counts().sort_index()
    plt.figure(figsize=(4, 4))
    counts.plot(kind="bar", color=["steelblue", "salmon"])
    plt.title("Class Distribution (0=Ham,1=Spam)")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "class_distribution.png", dpi=150)
    plt.close()


def word_count_boxplot(df: pd.DataFrame, col: str = "word_count"):
    """
    Plot and save word count boxplot, removing outliers.
    Args:
        df: pandas dataframe with word_count column
    """
    # Remove top 1% of outliers for better visualization
    q = df[col].quantile(0.99)
    df_filtered = df[df[col] < q]
    plt.figure(figsize=(6, 4))
    sns.boxplot(x=df_filtered[col], color="skyblue")
    plt.title("Message Word Count Distribution")
    plt.xlabel("Word Count")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "word_count_boxplot.png", dpi=150)
    plt.close()


def top_ngrams(
    df: pd.DataFrame,
    n: int = 20,
    ngram_range: Tuple[int, int] = (1, 2),
    text_col: str = TEXT_COL,
):
    """
    Plot and save top n-grams bar chart.
    Args:
        df: pandas dataframe with text_col
        n: number of top n-grams to display
        ngram_range: n-gram range for CountVectorizer
    """
    vec = CountVectorizer(
        stop_words="english", ngram_range=ngram_range, max_features=5000
    )
    X = vec.fit_transform(df[text_col])
    freqs = X.sum(axis=0).A1
    vocab = vec.get_feature_names_out()
    pairs = list(zip(vocab, freqs))
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:n]
    words, counts = zip(*top)
    plt.figure(figsize=(8, 4))
    plt.barh(words[::-1], counts[::-1], color="teal")
    plt.title("Top n-grams")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "top_ngrams.png", dpi=150)
    plt.close()


def suspicious_wordcloud(df: pd.DataFrame, text_col: str = TEXT_COL):
    """
    Generate and save a word cloud for the most common suspicious words in all emails.
    """
    all_text = " ".join(df[text_col].str.lower())
    word_counts = {word: all_text.count(word) for word in SUSPICIOUS_WORDS}
    word_counts = {word: count for word, count in word_counts.items() if count > 0}

    if not word_counts:
        print("No suspicious words found to generate a word cloud.")
        return

    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="Reds",
    ).generate_from_frequencies(word_counts)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title("Most Frequent Suspicious Words in All Emails")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "suspicious_wordcloud.png", dpi=150)
    plt.close()


def correlation_heatmap(df: pd.DataFrame):
    """
    Plot and save a heatmap of feature correlations.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        df[NUMERICAL_COLS + [TARGET_COL]].corr(),
        annot=True,
        fmt=".2f",
        cmap="viridis",
    )
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "correlation_heatmap.png", dpi=150)
    plt.close()


def suspicious_word_distribution(df: pd.DataFrame, target_col: str = TARGET_COL):
    """
    Plot distribution of suspicious word count for spam vs ham.
    """
    plt.figure(figsize=(8, 5))
    sns.violinplot(
        x=target_col, y="suspicious_word_count", data=df, palette=["lightblue", "salmon"]
    )
    plt.title("Distribution of Suspicious Word Count")
    plt.xlabel("Class (0=Ham, 1=Spam)")
    plt.ylabel("Suspicious Word Count")
    plt.xticks([0, 1], ["Ham", "Spam"])

    # Limit y-axis to the 99th percentile to exclude extreme outliers
    ylim_max = df["suspicious_word_count"].quantile(0.99)
    plt.ylim(0, ylim_max)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "suspicious_word_distribution.png", dpi=150)
    plt.close()


def run_eda(csv_path: str = "data/spam_featured.csv"):
    df = load_spam_dataset(csv_path)
    class_distribution(df)
    word_count_boxplot(df)
    top_ngrams(df)
    suspicious_wordcloud(df)
    correlation_heatmap(df)
    suspicious_word_distribution(df)
    print(f"EDA figures saved to {FIG_DIR}")


if __name__ == "__main__":
    run_eda()
