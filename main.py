"""
================================================================================
PROJECT: Amazon Insights Pitch Battle - Solution Pipeline
TEAM: [Insert Team Name]
CLIENT: Amazon
DATASET: Amazon Fine Food Reviews
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
from collections import Counter

# Download necessary NLTK data (uncomment on first run)
# nltk.download('punkt')
# nltk.download('stopwords')

# ==============================================================================
# 1. DATA LOADING & PREPROCESSING
# ==============================================================================

def load_data(file_path):
    """
    Loads the Amazon Fine Food Reviews dataset.
    BUSINESS VALUE: Ensures we are working with clean, reliable data.
    """
    try:
        df = pd.read_csv(file_path)
        print(f"[SUCCESS] Loaded {len(df)} records.")
        return df
    except Exception as e:
        print(f"[ERROR] Could not load data: {e}")
        return None

def preprocess_text(text):
    """Cleans review text for NLP analysis."""
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        tokens = [w for w in tokens if not w in stop_words]
        return " ".join(tokens)
    return ""

def prepare_dataset(df):
    """
    Feature Engineering step.
    - Creates 'Polarity' target for Classification.
    - Calculates 'Helpfulness Ratio' for Regression features (VECTORIZED).
    - Cleans text for NLP (Sampled for memory efficiency).
    """
    data = df.copy()
    
    # 1. Classification Target: Polarity
    def map_polarity(score):
        if score <= 2: return 'Negative'
        elif score == 3: return 'Neutral'
        else: return 'Positive'
    
    data['Polarity'] = data['Score'].apply(map_polarity)
    
    # 2. Regression Features - VECTORIZED
    data['HelpfulnessRatio'] = np.where(
        data['HelpfulnessDenominator'] == 0,
        0,
        data['HelpfulnessNumerator'] / data['HelpfulnessDenominator']
    )
    data['TextLength'] = data['Text'].astype(str).apply(len)
    
    # Handle missing Summary columns gracefully
    if 'Summary' in data.columns:
        data['SummaryLength'] = data['Summary'].astype(str).apply(len)
    else:
        data['SummaryLength'] = 0
    
    # 3. NLP Cleaning - SAMPLING FOR SPEED
    print("[INFO] Cleaning text for NLP (sampling for speed)...")
    sample_size = min(50000, len(data))
    data_sample = data.sample(sample_size, random_state=42)
    data_sample['CleanText'] = data_sample['Text'].apply(preprocess_text)
    
    return data_sample

# ==============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ==============================================================================

def perform_eda(df, save_dir='eda_plots'):
    """Generates key visualizations to present to Amazon Stakeholders."""
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # Use a safe style available in newer Matplotlib versions
    plt.style.use('ggplot') 
    
    # Plot 1: Distribution of Product Ratings
    plt.figure(figsize=(10, 6))
    sns.countplot(x='Score', data=df, palette='viridis')
    plt.title('Distribution of Product Ratings (1-5)')
    plt.savefig(f'{save_dir}/rating_distribution.png')
    plt.close()
    
    # Plot 2: Sentiment Polarity Balance
    plt.figure(figsize=(8, 6))
    sns.countplot(x='Polarity', data=df, order=['Positive', 'Neutral', 'Negative'], palette='coolwarm')
    plt.title('Review Sentiment Polarity')
    plt.savefig(f'{save_dir}/sentiment_polarity.png')
    plt.close()
    
    # Plot 3: Text Length vs Rating
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='Score', y='TextLength', data=df)
    plt.title('Review Text Length by Rating Score')
    plt.savefig(f'{save_dir}/text_length_vs_rating.png')
    plt.close()
    
    print(f"[SUCCESS] EDA plots saved to '{save_dir}' folder.")

# ==============================================================================
# 3. MODELING (REGRESSION + CLASSIFICATION)
# ==============================================================================

def train_models(df):
    """
    Trains predictive models required for the pitch.
    """
    models = {}
    
    # --- REGRESSION MODEL ---
    reg_features = ['HelpfulnessRatio', 'TextLength', 'SummaryLength']
    X_reg = df[reg_features].fillna(0)
    y_reg = df['Score']
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_reg, y_reg, test_size=0.2, random_state=42)
    
    reg_model = RandomForestRegressor(n_estimators=50, random_state=42) # Reduced estimators for speed
    reg_model.fit(X_train_r, y_train_r)
    reg_pred = reg_model.predict(X_test_r)
    reg_rmse = np.sqrt(mean_squared_error(y_test_r, reg_pred))
    
    models['regression'] = reg_model
    print(f"[MODEL] Regression RMSE: {reg_rmse:.2f}")
    
    # --- CLASSIFICATION MODEL ---
    tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
    X_clf = tfidf.fit_transform(df['CleanText'])
    y_clf = df['Polarity']
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_clf, y_clf, test_size=0.2, random_state=42)
    
    clf_model = LogisticRegression(max_iter=1000)
    clf_model.fit(X_train_c, y_train_c)
    clf_pred = clf_model.predict(X_test_c)
    
    models['classification'] = clf_model
    models['tfidf_vectorizer'] = tfidf
    print(f"[MODEL] Classification Accuracy: {accuracy_score(y_test_c, clf_pred):.2f}")
    
    return models

# ==============================================================================
# 4. NLP INSIGHTS (Themes & Keywords)
# ==============================================================================

def extract_top_keywords(df, top_n=10):
    """Identifies frequent words in Positive vs Negative reviews."""
    pos_text = " ".join(df[df['Polarity'] == 'Positive']['CleanText'])
    neg_text = " ".join(df[df['Polarity'] == 'Negative']['CleanText'])
    
    pos_words = Counter(pos_text.split()).most_common(top_n)
    neg_words = Counter(neg_text.split()).most_common(top_n)
    
    return pos_words, neg_words

# ==============================================================================
# 5. MAIN EXECUTION PIPELINE
# ==============================================================================

if __name__ == "__main__":
    DATASET_PATH = 'Reviews.csv' 
    MODEL_SAVE_PATH = 'models.pkl'
    
    print("=== AMAZON INSIGHTS PIPELINE STARTED ===")
    df = load_data(DATASET_PATH)
    
    if df is not None:
        df = prepare_dataset(df)
        perform_eda(df)
        
        
        models = train_models(df)
        
        pos_keywords, neg_keywords = extract_top_keywords(df)
        print("[INSIGHT] Top Positive Keywords:", pos_keywords[:5])
        print("[INSIGHT] Top Negative Keywords:", neg_keywords[:5])
        
        with open(MODEL_SAVE_PATH, 'wb') as f:
            pickle.dump(models, f)
        print(f"[SUCCESS] Models saved to {MODEL_SAVE_PATH}")
        
        # Save Cleaned Data for Streamlit App
        df.to_csv('cleaned_reviews.csv', index=False)
        print("=== PIPELINE COMPLETE. READY FOR STREAMLIT APP ===")
    else:
        print("=== PIPELINE ABORTED DUE TO DATA ERROR ===")
        