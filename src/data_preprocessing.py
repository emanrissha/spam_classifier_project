import os
import pandas as pd

def load_raw_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df):
    df = df.dropna()
    return df

def save_processed_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    raw_data_path = os.path.join("data", "raw", "spam_data.csv")
    processed_data_path = os.path.join("data", "processed", "spam_data_processed.csv")

    print("Loading raw data...")
    df_raw = load_raw_data(raw_data_path)

    print("Preprocessing data...")
    df_processed = preprocess_data(df_raw)

    print("Saving processed data...")
    save_processed_data(df_processed, processed_data_path)

    print("Data preprocessing completed.")
