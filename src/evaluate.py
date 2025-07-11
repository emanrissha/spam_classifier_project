import os
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate(model_path, tokenizer_path, processed_data_path):
    """
    Evaluate the saved model on the entire processed dataset.
    Prints classification report and saves confusion matrix plot.

    Parameters:
    - model_path: Path to the saved Keras model (.keras or .h5)
    - tokenizer_path: Path to the saved tokenizer pickle file
    - processed_data_path: CSV file path containing the dataset with 'text' and 'label' columns
    """

    # Load dataset
    df = pd.read_csv(processed_data_path)
    X_text = df['text'].astype(str).tolist()
    y_true = df['label'].values

    # Load tokenizer from disk
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # Vectorize the texts using saved tokenizer
    sequences = tokenizer.texts_to_sequences(X_text)
    X = pad_sequences(sequences, maxlen=100, padding='post')

    # Load trained model
    model = load_model(model_path)

    # Predict probabilities for each text sample
    y_pred_probs = model.predict(X)

    # Convert predicted probabilities to binary classes (0 or 1)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()

    # Print classification metrics to console
    print("Classification Report:")
    print(classification_report(y_true, y_pred))

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Plot confusion matrix heatmap using seaborn
    plt.figure(figsize=(6, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=['Ham', 'Spam'],
        yticklabels=['Ham', 'Spam']
    )
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Ensure output directory exists
    os.makedirs(os.path.join("outputs", "plots"), exist_ok=True)

    # Save the confusion matrix plot to a file
    plt.savefig(os.path.join("outputs", "plots", "confusion_matrix.png"))
    plt.tight_layout()

    # Show the plot
    plt.show()

if __name__ == "__main__":
    processed_data_path = os.path.join("data", "processed", "spam_data_processed.csv")
    model_path = os.path.join("outputs", "models", "spam_classifier.keras")
    tokenizer_path = os.path.join("outputs", "models", "tokenizer.pickle")

    evaluate(model_path, tokenizer_path, processed_data_path)
