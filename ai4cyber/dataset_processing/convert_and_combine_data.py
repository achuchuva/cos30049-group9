"""
Script to convert texts.csv to match emails.csv format and combine both datasets.
"""

import pandas as pd
import os

def convert_texts_csv():
    """Convert texts.csv to match emails.csv format"""

    # Read texts.csv
    texts_df = pd.read_csv('data/texts.csv', encoding='latin-1')
    
    # The first two columns contain the label and text
    # Rename columns to match what we need
    texts_df = texts_df.iloc[:, :2]  # Keep only first two columns
    texts_df.columns = ['label', 'text']
    
    # Convert labels: 'ham' -> 0, 'spam' -> 1
    label_map = {'ham': 0, 'spam': 1}
    texts_df['spam'] = texts_df['label'].map(label_map)
    
    # Keep only text and spam columns, reorder to match emails.csv
    texts_df = texts_df[['text', 'spam']]
    
    print(f"Converted {len(texts_df)} text messages")
    print(f"Label distribution:\n{texts_df['spam'].value_counts()}")
    
    return texts_df

def load_emails_csv():
    """Load emails.csv"""
    emails_df = pd.read_csv('data/emails.csv')
    
    return emails_df

def combine_datasets(texts_df, emails_df):
    """Combine both datasets"""
    
    # Concatenate the dataframes
    combined_df = pd.concat([texts_df, emails_df], ignore_index=True)
    
    # Remove any duplicates
    combined_df = combined_df.drop_duplicates(subset=['text'], keep='first')
    
    # Shuffle the dataset
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return combined_df

def main():
    """Convert and combine datasets"""
    
    # Convert texts.csv
    texts_df = convert_texts_csv()
    
    # Load emails.csv
    emails_df = load_emails_csv()
    
    # Combine datasets
    combined_df = combine_datasets(texts_df, emails_df)
    
    # Backup original texts.csv
    if os.path.exists('data/texts.csv'):
        os.rename('data/texts.csv', 'data/texts_original.csv')
    
    # Save converted texts.csv
    texts_df.to_csv('data/texts.csv', index=False, quoting=1)
    
    # Save combined dataset
    combined_df.to_csv('data/combined_spam_data.csv', index=False, quoting=1)

if __name__ == "__main__":
    main()
