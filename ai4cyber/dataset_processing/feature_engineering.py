import pandas as pd
import re
from urllib.parse import urlparse
from data_processing import SUSPICIOUS_WORDS

def count_suspicious_words(text):
    """Counts the number of suspicious words in a text."""
    if not isinstance(text, str):
        return 0
    count = 0
    for word in SUSPICIOUS_WORDS:
        count += text.lower().count(word)
    return count

def count_urls(text):
    """Counts the number of URLs in a text."""
    if not isinstance(text, str):
        return 0
    # Simple regex to find URLs
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    return len(urls)

def count_digits_in_urls(text):
    """Counts the number of digits in the domain names of URLs in a text."""
    if not isinstance(text, str):
        return 0
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    digit_count = 0
    for url in urls:
        try:
            hostname = urlparse(url).hostname
            if hostname:
                digit_count += sum(c.isdigit() for c in hostname)
        except Exception:
            continue
    return digit_count

def feature_engineering(input_path='data/spam.csv', output_path='data/spam_featured.csv'):
    """
    Loads spam data, engineers new features, and saves the enhanced dataset.
    """
    try:
        df = pd.read_csv(input_path, encoding='latin1')
    except FileNotFoundError:
        print(f"Error: The file {input_path} was not found.")
        return
    
    # Number of characters
    df['char_count'] = df['text'].str.len().fillna(0)

    # Number of words (length)
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)

    # Number of suspicious words
    df['suspicious_word_count'] = df['text'].apply(count_suspicious_words)

    # Number of URLs
    df['url_count'] = df['text'].apply(count_urls)

    # Number of digits in URLs
    df['url_digit_count'] = df['text'].apply(count_digits_in_urls)

    print(df.head())

    df.to_csv(output_path, index=False)

if __name__ == '__main__':
    feature_engineering()
