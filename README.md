# 🎬 IMDB Reviews Analyzer - Machine Learning Sentiment Classification

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.2+-orange?logo=scikit-learn" alt="Scikit-Learn"/>
  <img src="https://img.shields.io/badge/NLP-Processing-ff69b4" alt="NLP"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
</div>

## 📌 Project Overview

A machine learning pipeline that analyzes IMDB movie reviews to classify sentiment as positive or negative. The system includes:

- Automated dataset fetching from Dropbox
- Text preprocessing and feature extraction
- Model training and evaluation
- Ready-to-use prediction functionality

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

git clone https://github.com/Abuhurera-coder/IMDB-REVIEWS-CHECKER.git
cd IMDB-REVIEWS-CHECKER
pip install -r requirements.txt
Basic Usage
bash
# Download and preprocess data
python download_dataset.py
python preprocess.py

# Train and evaluate model
python train_model.py
python evaluate.py

# Make predictions
python predict.py "The movie was fantastic!"


<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/240304586-218f11fa-42f4-4af1-9e0a-a2a8e8a7f4e9.gif" width="100%" height="2px"/>
</div>

🏗️ Project Structure
text
IMDB-REVIEWS-CHECKER/
├── main.py                 # Main training/evaluation script
├── config.py               # Configuration settings
├── download_dataset.py     # Dataset fetcher
├── preprocess.py           # Text cleaning
├── train_model.py          # Model training
├── evaluate.py             # Performance evaluation
├── predict.py              # Prediction interface
├── requirements.txt        # Dependencies
├── model_artifacts/        # Saved models
└── performance_plots/      # Evaluation metrics


<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/240304586-218f11fa-42f4-4af1-9e0a-a2a8e8a7f4e9.gif" width="100%" height="2px"/>
</div>

🔧 Key Components
1. Data Processing
   
Text cleaning (special characters, stopwords)
Tokenization and normalization
TF-IDF vectorization

3. Machine Learning
   
Default classifier: Logistic Regression
Alternative: Random Forest
Hyperparameters configurable in train_model.py

3. Evaluation Metrics
   
Accuracy score
Precision/Recall
Confusion matrix
Classification report


<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/240304586-218f11fa-42f4-4af1-9e0a-a2a8e8a7f4e9.gif" width="100%" height="2px"/>
</div>


💻 Usage Examples
Training with Custom Parameters
python train_model.py --model lr --max_iter 1000
Evaluating Specific Metrics
python evaluate.py --metrics precision recall f1

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/240304586-218f11fa-42f4-4af1-9e0a-a2a8e8a7f4e9.gif" width="100%" height="2px"/>
</div>

🤝 Contributing
We welcome contributions! Please:

Fork the repository

Create a feature branch

Commit your changes

Push to the branch

Open a Pull Request

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/240304586-218f11fa-42f4-4af1-9e0a-a2a8e8a7f4e9.gif" width="100%" height="2px"/>
</div>

📜 License
MIT License - see LICENSE for details.

<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/240304586-218f11fa-42f4-4af1-9e0a-a2a8e8a7f4e9.gif" width="100%" height="2px"/>
</div>

✉️ Contact
For questions or support:
Email: abuhurerarchani@gmail.com
