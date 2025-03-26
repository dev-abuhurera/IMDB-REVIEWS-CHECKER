Report on Building the IMDB Reviews Checker Model

1. Introduction

This report outlines the complete process of developing an IMDB Reviews Checker Model to classify movie reviews as positive or negative. It includes dataset acquisition, preprocessing, model training, evaluation, and deployment using Dropbox for dataset storage and GitHub for version control.

2. Project Setup

a) Prerequisites

Before starting, install the required dependencies:

pip install -r requirements.txt

Ensure you have Python 3.x, Git, and a virtual environment (optional) set up.

b) Project Directory Structure

IMDB-REVIEWS-CHECKER/
│── model.py
│── preprocess.py
│── train.py
│── evaluate.py
│── config.py
│── requirements.txt
│── README.md
│── performance_plots/
│── model_artifacts/

3. Dataset Integration

a) Using Dropbox for Dataset Storage

We store the IMDB Dataset in Dropbox and use the shared link to download it dynamically.

Configuration (config.py):

class Config:
    DATASET_URL = "https://www.dropbox.com/scl/fi/e0htuwzj1yfdy4srn1mxd/IMDB-Dataset.csv?rlkey=iw3mf3xn16kqj81kg3ozrqsqn&st=8vgtki29&dl=1"
    DATASET_PATH = "IMDB_Dataset.csv"

b) Downloading the Dataset

Modify train.py to download the dataset:

import requests
from config import Config

def download_dataset():
    response = requests.get(Config.DATASET_URL)
    with open(Config.DATASET_PATH, 'wb') as file:
        file.write(response.content)

Run:

python train.py

4. Data Preprocessing

a) Steps in preprocess.py

Text Cleaning: Remove special characters, numbers, and stopwords.

Tokenization & Vectorization: Convert text into numerical format using TF-IDF.

Handling N-Grams: Use unigrams, bigrams, and trigrams.

Example Code:

from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_text(texts):
    vectorizer = TfidfVectorizer(max_features=15000, ngram_range=(1,3))
    return vectorizer.fit_transform(texts)

5. Model Development

a) Training (train.py)

Using Logistic Regression as the baseline classifier:

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from config import Config

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model, X_test, y_test

b) Saving the Model

import joblib
joblib.dump(model, 'model_artifacts/imdb_model.pkl')

6. Model Evaluation (evaluate.py)

from sklearn.metrics import accuracy_score, classification_report

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

Run:

python evaluate.py

7. Version Control with GitHub

a) Push Changes to GitHub

git init
git add .
git commit -m "Initial Commit"
git remote add origin https://github.com/Abuhurera-coder/IMDB-REVIEWS-CHECKER.git
git push -u origin main

b) Handling Errors

If you get an error while pushing:

git pull origin main --rebase
git push origin main

8. Running the Model

a) Load and Predict

model = joblib.load('model_artifacts/imdb_model.pkl')
new_review = preprocess_text(["The movie was fantastic!"])
print(model.predict(new_review))

Run:

python predict.py

9. Conclusion

This report provides a structured approach to developing a sentiment analysis model for IMDB reviews using machine learning. We utilized Dropbox for dataset storage and GitHub for version control. The model can be further improved with deep learning techniques (e.g., LSTMs, Transformers).

