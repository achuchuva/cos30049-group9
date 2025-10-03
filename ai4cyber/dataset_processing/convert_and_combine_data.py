"""
Script to convert texts.csv to match emails.csv format and combine both datasets.
This will help reduce overfitting by providing more training data.
"""

import pandas as pd
import os

def convert_texts_csv():
    """Convert texts.csv to match emails.csv format"""
    print("Reading texts.csv...")
    # Read texts.csv with proper column names
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
    print("\nReading emails.csv...")
    emails_df = pd.read_csv('data/emails.csv')
    
    print(f"Loaded {len(emails_df)} emails")
    print(f"Label distribution:\n{emails_df['spam'].value_counts()}")
    
    return emails_df

def combine_datasets(texts_df, emails_df):
    """Combine both datasets"""
    print("\nCombining datasets...")
    
    # Concatenate the dataframes
    combined_df = pd.concat([texts_df, emails_df], ignore_index=True)
    
    # Remove any duplicate messages
    print(f"Total messages before deduplication: {len(combined_df)}")
    combined_df = combined_df.drop_duplicates(subset=['text'], keep='first')
    print(f"Total messages after deduplication: {len(combined_df)}")
    
    # Shuffle the dataset
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nFinal label distribution:\n{combined_df['spam'].value_counts()}")
    print(f"Spam percentage: {combined_df['spam'].mean()*100:.2f}%")
    
    return combined_df

def main():
    """Main function to convert and combine datasets"""
    print("="*60)
    print("Converting texts.csv to match emails.csv format")
    print("="*60)
    
    # Convert texts.csv
    texts_df = convert_texts_csv()
    
    # Load emails.csv
    emails_df = load_emails_csv()
    
    # Combine datasets
    combined_df = combine_datasets(texts_df, emails_df)
    
    # Backup original texts.csv
    if os.path.exists('data/texts.csv'):
        print("Backing up original texts.csv to texts_original.csv...")
        os.rename('data/texts.csv', 'data/texts_original.csv')
    
    # Save converted texts.csv in the same format as emails.csv
    texts_df.to_csv('data/texts.csv', index=False, quoting=1)
    
    # Save combined dataset
    combined_df.to_csv('data/combined_spam_data.csv', index=False, quoting=1)
    print(f"Saved combined_spam_data.csv ({len(combined_df)} rows)")

if __name__ == "__main__":
    main()
