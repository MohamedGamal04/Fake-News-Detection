# Fake News Detection

Binary fake/real news classifier built with the "Fake and Real News" dataset. Includes data loading, robust text preprocessing, TF‑IDF feature extraction, model selection (Logistic Regression, SVC) with hyperparameter search, evaluation (confusion matrix, ROC), and visualizations (feature importance, word clouds). Notebook-oriented and reproducible.

Live demo : https://fake-news-detection-chelkxp852f4npgq7hjytm.streamlit.app/
## Features

- Text preprocessing including contraction expansion, stopword removal, lemmatization, and number conversion
- TF-IDF feature extraction from reviews
- Logistic Regression classification for sentiment prediction
- Exploratory data analysis and visualization
- Batch prediction support
- Streamlit UI demo for interactive sentiment analysis

## Project Structure

- `model.pkl` `vectorizer.pkl` - model and vectorizer files
- `Fake News Detection.ipynb` - Jupyter Notebook containing preprocessing, training, and evaluation pipeline
- `app.py` - Streamlit application script for live sentiment analysis
- `requirements.txt` - Required Python packages
- `README.md` - Project overview and usage instructions

## Installation

1. Clone the repository:
``` powershell
git clone https://github.com/MohamedGamal04/Fake-News-Detection.git
cd Fake-News-Detection
```
2. Create a Python environment and activate it (recommended):
``` powershell
conda create -n classification-env python=3.8
conda activate classification-env
```
3. Install dependencies:
``` powershell
pip install -r requirements.txt
```
4. Download necessary NLTK data resources:
``` python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
```

## Usage

- Train the model and evaluate using the Jupyter Notebook.
- To use the Streamlit app (once created and set up), run:
``` terminal
streamlit run app.py
```

- The Streamlit UI allows you to enter reviews manually or upload CSV files for batch prediction.

## Dataset

This project uses the [Fake and Real News dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset).
