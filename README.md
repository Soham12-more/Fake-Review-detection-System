[![Python](https://img.shields.io/badge/Python-3.8-orange)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-2.0.2-blue)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.2.3-blue)](https://pandas.pydata.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.5.2-yellow)](https://scikit-learn.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18-yellow)](https://www.tensorflow.org/)
[![MLflow](https://img.shields.io/badge/MLflow-2.20.3-lightblue)](https://mlflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7.6-green)](https://xgboost.readthedocs.io/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.43.0-red)](https://streamlit.io/)


# Fake Review Detection Project

Welcome to the comprehensive documentation for our Fake Review Detection project. This project is designed to distinguish between computer-generated (fake) reviews and original (human-written) reviews by leveraging a variety of natural language processing (NLP) techniques, traditional machine learning models, and deep learning methods. In this document, we detail the project background, data processing, feature engineering, model experimentation, experiment tracking using MLflow, and finally, the deployment of our application.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Data Collection & Preprocessing](#data-collection--preprocessing)
3. [Feature Engineering & Embeddings](#feature-engineering--embeddings)
4. [Modeling Approaches](#modeling-approaches)
   - [Traditional Machine Learning Models](#traditional-machine-learning-models)
   - [Deep Learning Models](#deep-learning-models)
5. [Experiment Tracking with MLflow](#experiment-tracking-with-mlflow)
6. [Running the App](#running-the-app)
7. [Docker Image](#docker-image)
8. [Conclusion](#conclusion)

---

## Project Overview

Fake reviews are a growing problem in the online ecosystem, impacting consumer trust and business reputations. Our project aims to automatically detect and flag these computer-generated reviews by analyzing textual content. To achieve this, we have employed various techniques including:

- **Text Preprocessing:** Cleaning and normalizing text data.
- **Feature Engineering:** Extracting useful metrics such as lexical diversity, sentiment scores, and syntactic patterns.
- **Embeddings:** Generating different embeddings using methods such as TF-IDF, Count Vectorization, and precomputed embeddings from models like BERT and GloVe.
- **Model Training:** Experimenting with a range of traditional machine learning models (Logistic Regression, Random Forest, SVC) and a more complex deep learning model using a two-layer LSTM.
- **Experiment Tracking:** Logging every experiment with detailed metrics, hyperparameters, and artifacts using MLflow.

---

## Data Collection & Preprocessing

Our data consists of review texts along with corresponding labels indicating whether a review is computer-generated or original. The preprocessing pipeline includes:

- **Text Cleaning:** Removing unwanted characters, punctuation, and noise.
- **Normalization:** Converting text to lower case, tokenizing sentences and words, and applying techniques such as stemming, lemmatization, and stop word removal.
- **Feature Extraction:** Calculating metrics like lexical diversity, average word length, sentiment polarity, subjectivity, Flesch Reading Ease, sentence length, and various part-of-speech counts.

Processed data files are stored in our `../Data/Feature-Engineered/` folder.

---

## Feature Engineering & Embeddings

To capture the nuances of textual data, we explored several embedding techniques:

- **TF-IDF Embeddings:** Transforming text into weighted term-frequency representations.
- **Count Vectorization:** Creating basic term-frequency vectors.
- **Precomputed Embeddings:** Using models such as BERT and GloVe to generate embeddings, which are stored as CSV files in the `../../embeddings/` folder.

These methods provide diverse representations of the data, enabling our models to learn both syntactic and semantic patterns.

You can view all our datasets created throught out the process here:  
[Drive link for all the datasets used](https://drive.google.com/file/d/1DNBx44dBOd0kvqR-lWxq74RI5_hsP6pN/view?usp=drive_link)

---

## Modeling Approaches

### Traditional Machine Learning Models

We experimented with several scikit-learn models:

- **Logistic Regression**
- **Random Forest Classifier**
- **Support Vector Classifier (SVC)**

For each model, we performed hyperparameter tuning using GridSearchCV with feasible parameter grids. Experiment results, including confusion matrices and metrics (accuracy, precision, recall, F1 score), are logged in MLflow.

### Deep Learning Models

To capture complex patterns, we built a deep learning model using TensorFlow Keras:

- **Two-Layer LSTM:** The model includes an embedding layer, two LSTM layers, and dense layers with dropout for regularization.
- **Text Tokenization & Padding:** We convert raw text into sequences using Keras’ Tokenizer and pad them to a uniform length.
- **Evaluation:** Model performance is evaluated on standard metrics and confusion matrices are logged.

---

## Experiment Tracking with MLflow

Our experiments are fully tracked using MLflow. For every run, we log:

- **Parameters:** File names, model types, hyperparameters, and embedding types.
- **Metrics:** Accuracy, precision, recall, and F1 score.
- **Artifacts:** Confusion matrices (as PNG images) and model artifacts.
- **Datasets:** Using the `mlflow.data` API, our dataset information is logged and appears under the MLflow UI's "Datasets" tab (for MLflow ≥ 2.4). When unavailable, the CSV files are logged as artifacts.

You can view all our experiments on Dagshub through MLflow here:  
[View MLflow Experiments on Dagshub](https://dagshub.com/<your-dagshub-username>/<your-repo>.mlflow/)

A progress log (`progress_log.csv`) is maintained to ensure experiments are not re-run unnecessarily.

---

## Production Pipeline

The web app combines two models: a scikit-learn pipeline (below) and a deep learning BiLSTM (see [Deep Learning Model and Ensemble](#deep-learning-model-and-ensemble)). The pipeline is trained by `src/train.py` as a single object, so the app computes features exactly the way the model was trained on them. Feature code lives in `src/features.py`.

**Signals used by the model:**
- **Word patterns:** TF-IDF over word unigrams and bigrams (top 50,000).
- **Sentiment:** TextBlob polarity and subjectivity, plus the mismatch between sentiment and the star rating.
- **Repetition:** repeated phrases within a review and lexical diversity.
- **Text similarity:** cosine similarity to the closest training review and the number of near-duplicate training reviews.
- **Style:** word and sentence length, readability, punctuation, uppercase and exclamation usage.

**Results on a held-out test set** (8,084 reviews, 20% stratified split; the regularization strength is chosen on a separate validation split):

| Features | Test accuracy |
|----------|---------------|
| TF-IDF only | 94.87% |
| TF-IDF + stats | 95.00% |
| TF-IDF + similarity | 94.61% |
| All features (refit on full training split) | **95.20%** (precision 94.69%, recall 95.77%, F1 95.23%) |
| Stats + similarity only (no words) | 86.06% |

Full metrics are written to `Reports/metrics.json` and `Reports/confusion_matrix_pipeline.png`.

### Deep Learning Model and Ensemble

`src/train_dl.py` trains a **bidirectional LSTM** on the same train/validation/test split. It reads the raw review text: a built-in tokenizer (20,000-word vocabulary, first 200 words of each review) feeds a 64-dimensional embedding, a BiLSTM layer with 64 units per direction, and a dense layer. Training stops early when validation accuracy stops improving (3 epochs; best epoch kept). The app averages the probabilities of the two models.

| Model | Test accuracy | Precision | Recall | F1 |
|-------|---------------|-----------|--------|----|
| Logistic regression pipeline | 95.20% | 94.69% | 95.77% | 95.23% |
| BiLSTM (deep learning) | 95.47% | 96.07% | 94.83% | 95.44% |
| **Ensemble (deployed)** | **96.57%** | **96.68%** | **96.46%** | **96.57%** |

The training curve is saved to `Reports/dl_training_curve.png`.

**Explainability:** the logistic regression model is linear, so for every prediction the app shows how much each word, phrase and signal pushed the decision toward "Original" or "Computer-generated", and lists the most similar training reviews.

**Retrain** (Python 3.10–3.12; TensorFlow does not import on Python 3.9.0):
```bash
cd src
python train.py      # about 10 minutes; writes src/model_pipeline.joblib
python train_dl.py   # about 8 minutes on CPU; writes src/dl_model.keras and the ensemble metrics
```

---

## Running the App

The Streamlit web app takes a review and its star rating, predicts whether the review is computer-generated or original, and explains which words and signals drove the prediction.

Use Python 3.10–3.12 (TensorFlow does not import on Python 3.9.0):
```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash); on macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run src/app.py        # opens at http://localhost:8501
```

**Whole datasets:** the app's **Batch (CSV file)** tab takes a CSV with one review per row. Choose the text column, and optionally a star-rating column and a true-label column (CG/OR or 0/1). It shows how many reviews were flagged, accuracy if labels are given, and lets you download every prediction as a CSV. From the command line:
```bash
cd src
python predict.py reviews.csv --text-column text_ --rating-column rating --label-column label -o predictions.csv
```
Both run at roughly 100 reviews per second on a laptop CPU (the full 40,432-review dataset takes about 4 minutes). Scoring the project's own dataset gives higher accuracy than the held-out figures, because most of it was used for training.

---

## Docker Image

We provide a Docker image for easy deployment of the project. The image includes all necessary code and dependencies.

**To download and run the Docker image:**

1. **Pull the Docker Image:**  
   ```bash
   docker pull <your-dockerhub-username>/fake_review_detection:latest

2. **Run the Docker Container:**
   ```bash
   docker run -p 8501:8501 <your-dockerhub-username>/fake_review_detection:latest

Your application will then be accessible at http://localhost:8501.

For more details, refer to our Docker Hub repository:
Docker Hub: [<your-dockerhub-username>/fake_review_detection](https://hub.docker.com/repository/docker/<your-dockerhub-username>/fake_review_detection/general)

## Conclusion

Our project integrates advanced NLP techniques, comprehensive feature engineering, rigorous experimentation, and robust MLflow-based tracking to build a reproducible system for fake review detection. This documentation provides an end-to-end overview of our process, from data preprocessing and model training to deployment. We encourage you to explore the code repository and MLflow experiment dashboard for more details.

*Thank you for your interest in our project!*

## Contributors

- [Soham More](https://github.com/Soham12-more)
- [Jay Joshi](https://github.com/anonjj)
