# Spam Classifier Project

## Overview
This project implements a simple spam detection system using deep learning with TensorFlow and Keras. It classifies SMS messages as **Spam** or **Ham** (not spam).

The system includes:
- Data preprocessing and tokenization
- A feed-forward neural network model
- Training with early stopping
- Model evaluation with classification report and confusion matrix
- Message prediction with saved model and tokenizer
- Command-line interface to train, evaluate, or predict

---

## Project Structure

spam_classifier_project/
│
├── data/
│ └── processed/
│ └── spam_data_processed.csv # Preprocessed dataset (CSV)
│
├── outputs/
│ ├── models/
│ │ ├── spam_classifier.keras # Saved trained model
│ │ └── tokenizer.pickle # Saved tokenizer
│ └── plots/
│ └── confusion_matrix.png # Confusion matrix plot
│
├── src/
│ ├── train.py # Training script
│ ├── evaluate.py # Evaluation script
│ ├── predict.py # Single message prediction script
│ ├── model.py # Model architecture definition
│
├── main.py # CLI for running training/evaluation/prediction
├── requirements.txt # Project dependencies
└── README.md # This file


---

## Setup Instructions

1. Clone or download the project.

2. Create and activate a Python virtual environment:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
Install dependencies:


pip install -r requirements.txt
How to Use
Run the main script:


python main.py
Choose from the menu:

1. Train Model: Train the neural network on the dataset.

2. Evaluate Model: Evaluate saved model performance and view confusion matrix.

3. Predict Message: Input your own message and get spam/ham prediction.

Dataset Format
The CSV file spam_data_processed.csv should contain two columns:

text: SMS message text (string)

label: 0 for Ham (not spam), 1 for Spam

Example:

label	text
0	"Ok lar... Joking wif u oni..."
1	"Free entry in 2 a wkly comp to win FA Cup..."

Model Details
Simple feed-forward neural network with two hidden layers (16 and 8 neurons)

ReLU activation for hidden layers

Sigmoid output for binary classification

Trained with Adam optimizer and binary crossentropy loss

Early stopping used to prevent overfitting

Notes & Tips
Ensure your data labels are numeric 0/1, not string labels like 'ham'/'spam'.

The tokenizer is limited to the top 1000 words and uses padding/truncation to length 100.

Evaluation outputs precision, recall, f1-score, and shows a confusion matrix heatmap.

You can reuse the saved model and tokenizer in your own applications.

Contact & Contribution
Feel free to open issues or submit pull requests for improvements!

License
MIT License

