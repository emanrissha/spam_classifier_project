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

```plaintext
spam_classifier_project/
│
├── data/
│   └── processed/
│       └── spam_data_processed.csv         # Preprocessed SMS dataset (CSV)
│
├── outputs/
│   ├── models/
│   │   ├── spam_classifier.keras           # Saved trained model (Keras format)
│   │   └── tokenizer.pickle                 # Saved tokenizer object (pickle format)
│   └── plots/
│       └── confusion_matrix.png             # Confusion matrix plot from evaluation
│
├── src/
│   ├── train.py                            # Script to train the model
│   ├── evaluate.py                         # Script to evaluate the saved model
│   ├── predict.py                          # Script to predict spam/ham for input messages
│   ├── model.py                            # Defines the neural network architecture
│
├── main.py                                # Main CLI entry point to run training, evaluation, or prediction
├── requirements.txt                       # Python dependencies
└── README.md                             # Project documentation (this file)
```

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
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```
---

## How to Use
Run the main script:

```bash
python main.py
```

You will be prompted with a menu:

```mathematica
Select an option:
1. Train Model
2. Evaluate Model
3. Predict Message
Enter choice (1/2/3):
```


```markdown
### What happens on each choice:

#### 1. Train Model

- Loads the dataset from `data/processed/spam_data_processed.csv`.
- Converts SMS text messages into padded sequences using a tokenizer.
- Builds the neural network model (defined in `src/model.py`).
- Trains the model with early stopping to prevent overfitting.
- Saves the trained model to `outputs/models/spam_classifier.keras`.
- Saves the tokenizer object to `outputs/models/tokenizer.pickle`.

#### 2. Evaluate Model

- Loads the saved model from `outputs/models/spam_classifier.keras`.
- Loads the tokenizer from `outputs/models/tokenizer.pickle`.
- Reads the entire dataset again (`data/processed/spam_data_processed.csv`).
- Vectorizes the text using the loaded tokenizer.
- Predicts labels on the dataset.
- Prints a classification report with precision, recall, f1-score.
- Generates and displays a confusion matrix plot.
- Saves the confusion matrix plot as `outputs/plots/confusion_matrix.png`.

#### 3. Predict Message

- Loads the saved model and tokenizer (same as above).
- Prompts the user to enter a custom SMS message.
- Converts the input message into model input format.
- Predicts if the message is **Spam** or **Ham** with confidence score.
- Prints the prediction result.

```

## Dataset Format
The CSV file spam_data_processed.csv should contain two columns:

text: SMS message text (string)

label: 0 for Ham (not spam), 1 for Spam

Example:

label	text
0	"Ok lar... Joking wif u oni..."
1	"Free entry in 2 a wkly comp to win FA Cup..."

## Model Details
Simple feed-forward neural network with two hidden layers (16 and 8 neurons)

ReLU activation for hidden layers

Sigmoid output for binary classification

Trained with Adam optimizer and binary crossentropy loss

Early stopping used to prevent overfitting

## Notes & Tips
Ensure your data labels are numeric 0/1, not string labels like 'ham'/'spam'.

The tokenizer is limited to the top 1000 words and uses padding/truncation to length 100.

Evaluation outputs precision, recall, f1-score, and shows a confusion matrix heatmap.

You can reuse the saved model and tokenizer in your own applications.

## Contact & Contribution
Feel free to open issues or submit pull requests for improvements!

## License
This project is licensed under the MIT License.



