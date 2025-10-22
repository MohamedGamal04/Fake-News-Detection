# Fake News Detection

Binary fake/real news classifier built with the "Fake and Real News" dataset. Includes data loading, robust text preprocessing, TF‑IDF feature extraction, model selection (Logistic Regression, SVC) with hyperparameter search, evaluation (confusion matrix, ROC), and visualizations (feature importance, word clouds). Notebook-oriented and reproducible.

## Features
- Data cleaning and lemmatization using NLTK
- TF‑IDF feature extraction
- Model selection and hyperparameter tuning (Logistic Regression, SVC)
- Feed‑forward neural network with TF‑IDF input
- Word clouds and feature importance visualization
- Classification reports and evaluation utilities

## Requirements (recommended)
- Python 3.8+  
- pandas, numpy, scikit-learn  
- nltk, num2words, contractions  
- plotly, wordcloud

## Installation (Windows PowerShell)
Create & activate virtual environment:
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```
Install packages:
```powershell
pip install -r requirements.txt
```
If no requirements.txt:
```powershell
pip install pandas numpy scikit-learn nltk num2words contractions plotly wordcloud
```
Run NLTK downloads in the notebook (punkt, stopwords, wordnet, averaged_perceptron_tagger).

## Usage
1. Open `Fake News Detection.ipynb` in VS Code or Jupyter and run cells sequentially.  
2. Steps in the notebook: download dataset, preprocess text, extract TF‑IDF features, run GridSearchCV, evaluate models, and visualize results.  
3. Save vectorizer and best model artifacts for inference.

## Acknowledgments
- The dataset used in this project is sourced from [Kaggle]([https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset]).
