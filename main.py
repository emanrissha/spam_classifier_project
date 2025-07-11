import os
from src.train import train
from src.evaluate import evaluate
from src.predict import predict_message

def main():
    # Define paths for data, model, and tokenizer
    processed_data_path = os.path.join("data", "processed", "spam_data_processed.csv")
    model_save_path = os.path.join("outputs", "models", "spam_classifier.keras")
    tokenizer_save_path = os.path.join("outputs", "models", "tokenizer.pickle")

    print("Select an option:")
    print("1. Train Model")
    print("2. Evaluate Model")
    print("3. Predict Message")
    choice = input("Enter choice (1/2/3): ")

    if choice == '1':
        # Train the model and save tokenizer
        train(processed_data_path, model_save_path, tokenizer_save_path)
    elif choice == '2':
        # Evaluate the saved model using model, tokenizer and data paths
        evaluate(model_save_path, tokenizer_save_path, processed_data_path)
    elif choice == '3':
        # Predict single message label (spam/ham)
        message = input("Enter a message to classify (Spam or Ham):\n> ")
        predict_message(model_save_path, tokenizer_save_path, message)
    else:
        print("Invalid choice. Exiting.")

if __name__ == "__main__":
    main()
