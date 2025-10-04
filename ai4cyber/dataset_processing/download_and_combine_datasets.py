"""
Script to download datasets from Hugging Face and combine them with local data.
"""
import os
import pandas as pd
from datasets import load_dataset

def download_and_prepare_hf_datasets():
    """
    Downloads and prepares datasets from Hugging Face.
    - Deysi/spam-detection-dataset
    - FredZhang7/all-scam-spam
    """

    # Deysi/spam-detection-dataset
    deysi_ds = load_dataset("Deysi/spam-detection-dataset", split="train")
    deysi_df = deysi_ds.to_pandas()
    # It has 'text' and 'label' (spam and not_spam)
    deysi_df = deysi_df.rename(columns={"label": "spam"})
    deysi_df['spam'] = deysi_df['spam'].map({'spam': 1, 'not_spam': 0})
    print(f"Loaded {len(deysi_df)} messages from Deysi/spam-detection-dataset.")

    # FredZhang7/all-scam-spam
    fred_ds = load_dataset("FredZhang7/all-scam-spam", split="train")
    fred_df = fred_ds.to_pandas()
    # It has 'text' and 'is_spam' columns
    fred_df = fred_df.rename(columns={"is_spam": "spam"})
    print(f"Loaded {len(fred_df)} messages from FredZhang7/all-scam-spam.")

    # Combine the Hugging Face datasets
    hf_df = pd.concat([deysi_df, fred_df], ignore_index=True)
    return hf_df

def main():
    """Download, combine, and save the datasets."""

    # Data directory exists
    os.makedirs("data", exist_ok=True)

    # Get Hugging Face datasets
    hf_df = download_and_prepare_hf_datasets()

    # Get local datasets
    local_df = pd.read_csv('data/combined_spam_data.csv')

    # Combine datasets
    combined_df = pd.concat([hf_df, local_df], ignore_index=True)

    # Drop rows where text is missing
    combined_df.dropna(subset=['text'], inplace=True)
    combined_df['text'] = combined_df['text'].astype(str)
    # Drop duplicates
    combined_df.drop_duplicates(subset=['text'], keep='first', inplace=True)

    # Shuffle the dataset
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Save the final dataset
    output_path = 'data/spam.csv'
    combined_df.to_csv(output_path, index=False, quoting=1)

    print(f"Total rows: {len(combined_df)}")


if __name__ == "__main__":
    main()
