# 🎬 IMDB Reviews Analyzer - Machine Learning Sentiment Classification

![Project Banner](https://via.placeholder.com/1200x400/0D1117/7d40ff?text=IMDB+REVIEWS+ANALYZER) <!-- Replace with actual banner -->

<div align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/Scikit--Learn-1.2+-orange?logo=scikit-learn" alt="Scikit-Learn"/>
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License"/>
  <img src="https://img.shields.io/badge/Contributions-Welcome-brightgreen" alt="Contributions"/>
</div>

<!-- Gradient Separator -->
<div align="center">
  <img src="https://user-images.githubusercontent.com/74038190/240304586-218f11fa-42f4-4af1-9e0a-a2a8e8a7f4e9.gif" width="100%" height="2px"/>
</div>

## 🌟 Key Features

<div style="columns: 2; column-gap: 20px;">
  
✔ **Automated Dataset Fetching** from Dropbox  
✔ **Advanced Text Preprocessing** (Tokenization, Stopword Removal)  
✔ **TF-IDF Vectorization** for feature extraction  
✔ **Sentiment Classification** (Positive/Negative)  
✔ **Model Persistence** with joblib serialization  
✔ **Performance Visualization** (Accuracy, Confusion Matrix)  
✔ **Ready-to-Use Prediction** for new reviews  

</div>

<!-- Wave Separator -->
<div align="center">
  <img src="https://capsule-render.vercel.app/api?type=wave&color=7d40ff&height=30&section=divider"/>
</div>

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

# Clone repository
git clone https://github.com/Abuhurera-coder/IMDB-REVIEWS-CHECKER.git
cd IMDB-REVIEWS-CHECKER

# Install dependencies
pip install -r requirements.txt
Usage Pipeline
bash
# 1. Download dataset
python download_dataset.py

# 2. Preprocess data
python preprocess.py

# 3. Train model
python train_model.py

# 4. Evaluate model
python evaluate.py

# 5. Make predictions
python predict.py "This movie exceeded all expectations!"
<!-- Dashed Separator --><div align="center"> <hr style="border: 1px dashed #7d40ff; width: 80%; margin: 25px 0;"> </div>
🏗️ Project Structure
text
IMDB-REVIEWS-CHECKER/
├── main.py                 # Main training/evaluation script
├── config.py               # Configuration settings
├── download_dataset.py     # Dataset fetcher from Dropbox
├── preprocess.py           # Text cleaning and preparation
├── train_model.py          # Model training pipeline
├── evaluate.py             # Performance evaluation
├── predict.py              # Sentiment prediction
├── requirements.txt        # Dependency list
├── model_artifacts/        # Saved models
│   ├── classifier.pkl
│   └── vectorizer.pkl
└── performance_plots/      # Evaluation visuals
    ├── accuracy.png
    └── confusion_matrix.png
<!-- Double Line Separator --><div align="center"> <hr style="border-top: 1px solid #7d40ff; border-bottom: 1px solid #7d40ff; height: 4px; width: 70%; margin: 25px 0; background: transparent;"> </div>
📊 Model Performance
Metric	Score
Accuracy	89.2%
Precision (Positive)	88.5%
Recall (Negative)	90.1%
F1-Score	89.3%
https://via.placeholder.com/400/0D1117/7d40ff?text=Confusion+Matrix+Example <!-- Replace with actual plot -->

<!-- Dots Separator --><div align="center"> <span style="color: #7d40ff; font-size: 24px;">• • •</span> </div>
🔧 Customization
Using Your Own Dataset
Upload your CSV to Dropbox

Update config.py:

python
DATASET_URL = "your-dropbox-link?dl=1"  # Must end with dl=1
Changing Model Parameters
Edit train_model.py:

python
# Example: Switch to Logistic Regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
<!-- Section Separator --><div align="center"> <img src="https://capsule-render.vercel.app/api?type=rect&color=7d40ff&height=2&section=footer&width=100%"/> </div>
🤝 How to Contribute
Fork the repository

Create your feature branch (git checkout -b feature/AmazingFeature)

Commit your changes (git commit -m 'Add some feature')

Push to the branch (git push origin feature/AmazingFeature)

Open a Pull Request

📜 License
Distributed under the MIT License. See LICENSE for more information.

✉️ Contact
Muhammad Abuhurera
📧 abuhurerarchani@gmail.com
🔗 GitHub Profile
