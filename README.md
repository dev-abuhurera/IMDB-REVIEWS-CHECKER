IMDB Reviews Checker

This project is designed to analyze movie reviews using machine learning. It fetches the IMDB dataset from Dropbox, processes it, and applies natural language processing techniques to classify reviews as positive or negative.

📌 Features

Fetches dataset from Dropbox automatically.

Preprocesses text data (removes stopwords, tokenizes, etc.).

Implements machine learning models for sentiment analysis.

Saves trained models for future use.

Provides performance plots for model evaluation.

🛠️ Setup & Installation

Follow these steps to set up and run the project:

1️⃣ Clone the Repository

git clone https://github.com/Abuhurera-coder/IMDB-REVIEWS-CHECKER.git
cd IMDB-REVIEWS-CHECKER

2️⃣ Install Dependencies

Make sure you have Python installed. Then, run:

pip install -r requirements.txt

This installs necessary packages like Pandas, NumPy, Scikit-learn, etc.

3️⃣ Fetch IMDB Dataset from Dropbox

We are using Dropbox to store and retrieve the dataset. The dataset is specified in config.py:

DATASET_URL = "https://www.dropbox.com/scl/fi/e0htuwzj1yfdy4srn1mxd/IMDB-Dataset.csv?rlkey=iw3mf3xn16kqj81kg3ozrqsqn&st=8vgtki29&dl=1"

To download the dataset, run:

python download_dataset.py

This script fetches the dataset and saves it as IMDB Dataset.csv in your project folder.

📂 Project Structure

📂 IMDB-REVIEWS-CHECKER/
├── 📄 main.py                 # Main script to train and evaluate the model
├── 📄 config.py               # Configuration settings
├── 📄 download_dataset.py     # Script to download dataset from Dropbox
├── 📄 preprocess.py           # Text preprocessing functions
├── 📄 train_model.py          # Model training script
├── 📄 evaluate.py             # Model evaluation script
├── 📂 model_artifacts/        # Saved models and vectorizers
├── 📂 performance_plots/      # Plots for model evaluation
├── 📄 requirements.txt        # Required dependencies
├── 📄 README.md               # Documentation
└── 📄 .gitignore              # Ignore unnecessary files

📝 Usage Instructions

4️⃣ Preprocess the Data

Before training, we need to clean the dataset. Run:

python preprocess.py

This removes unwanted characters, tokenizes text, and prepares data for model training.

5️⃣ Train the Model

Train the sentiment analysis model:

python train_model.py

This will:

Extract features from the text (TF-IDF vectorization)

Train a machine learning model

Save the trained model to model_artifacts/

6️⃣ Evaluate the Model

Once trained, test the model:

python evaluate.py

This script:

Loads the trained model

Evaluates it on test data

Generates accuracy and confusion matrix plots in performance_plots/

7️⃣ Predict New Reviews

To classify a new review, use:

python predict.py "This movie was fantastic!"

💾 Dropbox Integration

This project uses Dropbox to store the dataset. You can replace the dataset URL in config.py with your own Dropbox link:

DATASET_URL = "your-dropbox-file-link"

Ensure that:

The link ends with dl=1 to allow direct downloads.

You update download_dataset.py accordingly.

📊 Performance Evaluation

Model accuracy and confusion matrix will be saved in performance_plots/.

You can modify evaluate.py to test different evaluation metrics.

🚀 Next Steps

Improve text preprocessing.

Experiment with different ML models.

Deploy the model using Flask or FastAPI.

🛠️ Troubleshooting

Error: ModuleNotFoundError: No module named 'pandas'Solution: Install dependencies using pip install -r requirements.txt.

Error: Git push failed due to rejected updatesSolution: Run git pull origin main --rebase before pushing.

👨‍💻 Contributing

Want to improve this project? Feel free to fork and submit a pull request!

📜 License

This project is open-source under the MIT License.

🔗 Connect with Me

GitHub: Abuhurera-coder

If you face any issue feel free to contact me.
