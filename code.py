import pandas as pd
import re
import string
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Configuration
class Config:
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    MAX_FEATURES = 15000
    NGRAM_RANGE = (1, 3)
    MIN_WORD_LENGTH = 3
    MODEL_DIR = "model_artifacts"
    PLOT_DIR = "performance_plots"

# Enhanced stop words
CUSTOM_STOP_WORDS = ENGLISH_STOP_WORDS.union({
    'movie', 'film', 'one', 'make', 'get', 'even', 'would', 'like',
    'character', 'story', 'plot', 'scene', 'watch'
})

def load_dataset():
    """Load and validate dataset"""
    dataset_filename = "IMDB Dataset.csv"
    search_paths = [
        os.path.join(os.getcwd(), dataset_filename),
        os.path.join(os.path.dirname(__file__), dataset_filename),
        os.path.join(os.path.expanduser("~"), "Downloads", dataset_filename),
        os.path.join("data", dataset_filename)
    ]
    
    for path in search_paths:
        if os.path.exists(path):
            print(f"Found dataset at: {path}")
            df = pd.read_csv(path)
            if not all(col in df.columns for col in ['review', 'sentiment']):
                raise ValueError("Dataset missing required columns")
            print(f"Loaded {len(df)} reviews")
            return df
    
    raise FileNotFoundError("Dataset not found in expected locations")

def enhanced_preprocess(text):
    """Thorough text cleaning"""
    text = str(text).lower().strip()
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
    text = re.sub(r'[^\w\s]', '', text)  # Remove special chars
    
    # Handle contractions
    contractions = {
        "won't": "will not", "can't": "cannot", "n't": " not",
        "'re": " are", "'s": " is", "'d": " would", "'ll": " will"
    }
    for pat, repl in contractions.items():
        text = re.sub(pat, repl, text)
    
    words = text.split()
    words = [w for w in words 
             if (w not in CUSTOM_STOP_WORDS and 
                 len(w) >= Config.MIN_WORD_LENGTH)]
    return ' '.join(words)

def interactive_predict(vectorizer, model):
    """Interactive prediction loop"""
    print("\n" + "="*50)
    print("MOVIE REVIEW SENTIMENT ANALYZER")
    print("Enter a movie review and press Enter to get prediction")
    print("Type 'quit' or 'exit' to end the session")
    print("="*50)
    
    while True:
        review = input("\nEnter your movie review: ")
        
        if review.lower() in ['quit', 'exit']:
            print("\nExiting... Thank you for using the sentiment analyzer!")
            break
            
        try:
            # Preprocess and predict
            clean_text = enhanced_preprocess(review)
            features = vectorizer.transform([clean_text])
            pred = model.predict(features)[0]
            proba = model.predict_proba(features)[0]
            
            # Display results
            print("\n" + "-"*30)
            print("PREDICTION RESULT:")
            print(f"Sentiment: {'POSITIVE' if pred == 1 else 'NEGATIVE'}")
            print(f"Confidence: {max(proba)*100:.1f}%")
            print(f"Positive: {proba[1]*100:.1f}% | Negative: {proba[0]*100:.1f}%")
            print("-"*30)
            
        except Exception as e:
            print(f"Error processing your review: {e}")

def main():
    # Setup directories
    os.makedirs(Config.MODEL_DIR, exist_ok=True)
    os.makedirs(Config.PLOT_DIR, exist_ok=True)
    
    # Load and preprocess data
    print("\nLoading and preprocessing data...")
    df = load_dataset()
    tqdm.pandas(desc="Processing reviews")
    df['clean_text'] = df['review'].progress_apply(enhanced_preprocess)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        df['clean_text'],
        df['sentiment'],
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=df['sentiment']
    )

    # Vectorization
    vectorizer = TfidfVectorizer(
        max_features=Config.MAX_FEATURES,
        ngram_range=Config.NGRAM_RANGE,
        stop_words=list(CUSTOM_STOP_WORDS),
        min_df=5,
        max_df=0.85
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Model training
    model = LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        random_state=Config.RANDOM_STATE,
        C=0.9,
        solver='liblinear'
    )
    model.fit(X_train_vec, y_train)

    # Evaluation
    y_pred = model.predict(X_test_vec)
    print("\n=== Model Performance ===")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

    # Save artifacts
    version = "v1"
    artifact_path = os.path.join(Config.MODEL_DIR, version)
    os.makedirs(artifact_path, exist_ok=True)
    joblib.dump(vectorizer, os.path.join(artifact_path, "vectorizer.joblib"))
    joblib.dump(model, os.path.join(artifact_path, "model.joblib"))
    
    # Start interactive prediction
    interactive_predict(vectorizer, model)

if __name__ == "__main__":
    main()