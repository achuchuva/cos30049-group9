import pandas as pd
import re
from urllib.parse import urlparse

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
    # A simple regex to find URLs
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

    # Ensure the 'text' column exists
    if 'v2' in df.columns and 'v1' in df.columns:
        df = df.rename(columns={'v2': 'text', 'v1': 'label'})
        df = df[['text', 'label']]
    elif 'text' not in df.columns:
        print("Error: The CSV must have a 'text' column.")
        return
    
    # 1. Number of characters
    df['char_count'] = df['text'].str.len().fillna(0)

    # 2. Number of words (length)
    df['word_count'] = df['text'].apply(lambda x: len(str(x).split()) if pd.notna(x) else 0)

    # 3. Number of suspicious words
    df['suspicious_word_count'] = df['text'].apply(count_suspicious_words)

    # 4. Number of URLs
    df['url_count'] = df['text'].apply(count_urls)

    # 5. Number of digits in URLs
    df['url_digit_count'] = df['text'].apply(count_digits_in_urls)

    print("Feature engineering complete.")
    print(df.head())

    # Save the new dataframe
    print(f"Saving featured data to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done.")

if __name__ == '__main__':
    feature_engineering()
