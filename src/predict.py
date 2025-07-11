import os
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

def predict_message(model_path, tokenizer_path, message):
    """
    Predict if a single input message is spam or ham.
    Loads the saved model and tokenizer to preprocess and predict.
    """
    # Load the saved tokenizer from disk for text preprocessing
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)

    # Convert the input message to sequence of integers
    message_seq = tokenizer.texts_to_sequences([message])
    # Pad the sequence so it matches the model input length
    padded_message = pad_sequences(message_seq, maxlen=100, padding='post')

    # Load the trained Keras model from file
    model = load_model(model_path)

    # Predict the probability that the message is spam
    prediction = model.predict(padded_message)[0][0]

    # Decide label based on threshold 0.5
    label = "SPAM 🚨" if prediction > 0.5 else "HAM ✅"

    # Print results with confidence score
    print(f"\nInput message: {message}")
    print(f"Prediction: {label} (confidence: {prediction:.2f})")

if __name__ == "__main__":
    # Paths to saved model and tokenizer
    model_path = os.path.join("outputs", "models", "spam_classifier.keras")
    tokenizer_path = os.path.join("outputs", "models", "tokenizer.pickle")

    # Get input from user
    user_input = input("Enter a message to classify (Spam or Ham):\n> ")

    # Run prediction
    predict_message(model_path, tokenizer_path, user_input)
