import os
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.model import build_model

def load_data(processed_data_path):
    """
    Load the CSV file containing processed text data with labels.
    """
    return pd.read_csv(processed_data_path)

def vectorize_text(texts):
    """
    Convert list of texts into padded sequences of integers.
    Uses a tokenizer limited to 1000 words with OOV token.
    """
    tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=100, padding='post')
    return padded, tokenizer

def train(processed_data_path, model_save_path, tokenizer_save_path):
    """
    Train the spam classifier model.
    - Loads data
    - Prepares tokenizer and sequences
    - Splits into train and validation sets
    - Builds and trains the model with early stopping
    - Saves model and tokenizer
    """
    df = load_data(processed_data_path)

    # Check if labels are strings ('ham', 'spam') or numeric (0,1)
    if df['label'].dtype == object:
        # Convert string labels to numeric 0 (ham) and 1 (spam)
        df['label'] = df['label'].astype(str).str.lower().map({'ham': 0, 'spam': 1})
    else:
        # Already numeric labels, just make sure they're ints
        df['label'] = df['label'].astype(int)

    # Warn if any labels couldn't be mapped
    if df['label'].isnull().any():
        print("❗ Warning: Some labels couldn't be mapped. Check your CSV data:")
        print(df[df['label'].isnull()])

    X_text = df['text'].astype(str).tolist()
    y = df['label'].values

    # Vectorize text data (tokenize and pad sequences)
    X, tokenizer = vectorize_text(X_text)

    # Split into training and validation datasets (80% train, 20% val)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Build the neural network model
    model = build_model(input_shape=(X.shape[1],))

    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )

    # Train the model
    model.fit(
        X_train,
        y_train,
        epochs=20,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=[early_stop]
    )

    # Make sure directories exist for saving model and tokenizer
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    os.makedirs(os.path.dirname(tokenizer_save_path), exist_ok=True)

    # Save the trained model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

    # Save the tokenizer for use in prediction
    with open(tokenizer_save_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {tokenizer_save_path}")

    return model, tokenizer

if __name__ == "__main__":
    processed_data_path = os.path.join("data", "processed", "spam_data_processed.csv")
    model_save_path = os.path.join("outputs", "models", "spam_classifier.keras")
    tokenizer_save_path = os.path.join("outputs", "models", "tokenizer.pickle")

    train(processed_data_path, model_save_path, tokenizer_save_path)
