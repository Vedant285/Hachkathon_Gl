"""
================================================================================
PROJECT: Amazon Insights Dashboard (Streamlit Prototype)
TEAM: [Insert Team Name]
================================================================================
DESCRIPTION:
This app allows Amazon stakeholders to interact with the models.
Features:
1. Overview of Sentiment Distribution.
2. Predict Review Rating (Regression) based on metadata.
3. Classify Review Sentiment (Classification) based on review text.
4. View Key Insights (Keywords/Themes).
================================================================================
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import re
from PIL import Image

# Page Config
st.set_page_config(page_title="Amazon Quality Insights", layout="wide")

# Title & Header
st.title("📦 Amazon Fine Food Reviews - Quality Insights Dashboard")
st.markdown("""
**Consultant Pitch:** This dashboard helps Amazon optimize product retention, 
detect seller quality issues early, and automate sentiment monitoring using predictive analytics.
""")

# Sidebar for Navigation
st.sidebar.header("Navigation")
option = st.sidebar.selectbox(
    "Select View",
    ["Overview & EDA", "Predict Rating (Regression)", "Classify Sentiment (NLP)", "NLP Insights"]
)

# Load Models and Data (Cached for performance)
@st.cache_resource
def load_models():
    try:
        with open('models.pkl', 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

@st.cache_data
def load_data():
    try:
        return pd.read_csv('cleaned_reviews.csv')
    except:
        return pd.read_csv('Reviews.csv')  # Fallback

models = load_models()
df = load_data()

# ==============================================================================
# HELPER: Preprocess text for classification (matches training pipeline)
# ==============================================================================
def preprocess_text_for_prediction(text):
    """Cleans input text to match training preprocessing."""
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        # Simple tokenization (no NLTK dependency for app speed)
        tokens = text.split()
        # Remove common stopwords (small set for speed)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
        tokens = [w for w in tokens if w not in stop_words and len(w) > 2]
        return " ".join(tokens)
    return ""

# ==============================================================================
# VIEW 1: OVERVIEW & EDA
# ==============================================================================
if option == "Overview & EDA":
    st.header("1. Consumer Sentiment Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rating Distribution")
        fig1, ax1 = plt.subplots()
        df['Score'].value_counts().sort_index().plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title("Product Ratings (1-5)")
        st.pyplot(fig1)
        
    with col2:
        st.subheader("Sentiment Polarity")
        if 'Polarity' not in df.columns:
            df['Polarity'] = df['Score'].apply(
                lambda x: 'Negative' if x<=2 else ('Neutral' if x==3 else 'Positive')
            )
        fig2, ax2 = plt.subplots()
        df['Polarity'].value_counts().plot(kind='pie', ax=ax2, autopct='%1.1f%%')
        ax2.set_title("Positive vs Negative Reviews")
        st.pyplot(fig2)

    st.info("**Business Insight:** Majority of reviews are positive. Focus on the 'Negative' segment for quality improvement.")

# ==============================================================================
# VIEW 2: PREDICT RATING (REGRESSION)
# ==============================================================================
elif option == "Predict Rating (Regression)":
    st.header("2. Predict Product Rating")
    st.markdown("Enter review metadata to predict the likely star rating (1-5).")
    
    if models and 'regression' in models:
        col1, col2 = st.columns(2)
        with col1:
            help_num = st.number_input("Helpfulness Numerator", min_value=0, value=5)
            help_den = st.number_input("Helpfulness Denominator", min_value=1, value=10)
            text_len = st.number_input("Review Text Length (chars)", min_value=0, value=100)
            sum_len = st.number_input("Summary Length (chars)", min_value=0, value=20)
            
        with col2:
            st.write("### Model Prediction")
            if st.button("Predict Rating"):
                help_ratio = help_num / help_den if help_den > 0 else 0
                input_data = pd.DataFrame([[help_ratio, text_len, sum_len]], 
                                          columns=['HelpfulnessRatio', 'TextLength', 'SummaryLength'])
                
                prediction = models['regression'].predict(input_data)[0]
                st.metric("Predicted Score", f"{prediction:.2f} / 5.0")
                
                if prediction < 3:
                    st.error("⚠️ Risk: Low rating predicted. Investigate product quality.")
                else:
                    st.success("✅ Opportunity: High rating predicted. Good product fit.")
    else:
        st.warning("Models not found. Please run solution.py first.")

# ==============================================================================
# VIEW 3: CLASSIFY SENTIMENT (CLASSIFICATION + NLP) ✨ NEW ✨
# ==============================================================================
elif option == "Classify Sentiment (NLP)":
    st.header("3. Classify Review Sentiment")
    st.markdown("""
    **Business Use:** Automatically flag incoming reviews for seller monitoring.
    - 🟢 Positive: Promote product, feature in recommendations
    - 🟡 Neutral: Monitor for trends
    - 🔴 Negative: Alert seller, trigger quality review
    """)
    
    if models and 'classification' in models and 'tfidf_vectorizer' in models:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            review_text = st.text_area(
                "Enter Review Text:",
                height=150,
                placeholder="e.g., 'This coffee tastes amazing and arrived fresh!'"
            )
            
        with col2:
            st.write("### Sentiment Prediction")
            if st.button("Classify Sentiment"):
                if review_text.strip():
                    # Preprocess input to match training
                    clean_text = preprocess_text_for_prediction(review_text)
                    
                    # Vectorize
                    tfidf = models['tfidf_vectorizer']
                    X_input = tfidf.transform([clean_text])
                    
                    # Predict
                    clf = models['classification']
                    prediction = clf.predict(X_input)[0]
                    proba = clf.predict_proba(X_input)[0]
                    
                    # Display result
                    st.metric("Predicted Polarity", prediction)
                    
                    # Confidence bars
                    st.write("**Confidence:**")
                    classes = clf.classes_
                    for cls, prob in zip(classes, proba):
                        st.progress(int(prob * 100))
                        st.caption(f"{cls}: {prob:.1%}")
                    
                    # Business action
                    st.write("---")
                    if prediction == 'Negative':
                        st.error("🚨 Action: Flag for seller review + quality check")
                    elif prediction == 'Neutral':
                        st.warning("👀 Action: Monitor for emerging trends")
                    else:
                        st.success("✅ Action: Feature in recommendations + promote")
                else:
                    st.warning("Please enter a review text to classify.")
    else:
        st.warning("Classification model not found. Please run solution.py first.")

# ==============================================================================
# VIEW 4: NLP INSIGHTS (Themes & Keywords)
# ==============================================================================
elif option == "NLP Insights":
    st.header("4. Topic & Keyword Analysis")
    st.markdown("What words drive Positive vs Negative sentiment?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🟢 Positive Drivers")
        pos_df = df[df['Score'] >= 4] if 'Score' in df.columns else df[df['Polarity'] == 'Positive']
        st.write("Top words: *fresh, delicious, quality, love, perfect*")
        st.dataframe(pos_df[['Summary', 'Score']].head(5) if 'Summary' in df.columns else pos_df[['Text', 'Score']].head(5))
        
    with col2:
        st.subheader("🔴 Negative Drivers")
        neg_df = df[df['Score'] <= 2] if 'Score' in df.columns else df[df['Polarity'] == 'Negative']
        st.write("Top words: *expired, spoiled, tasteless, damaged, late*")
        st.dataframe(neg_df[['Summary', 'Score']].head(5) if 'Summary' in df.columns else neg_df[['Text', 'Score']].head(5))
        
    st.markdown("""
    **Recommendation for Amazon:** 
    Use these keywords to auto-flag incoming reviews. 
    If 'expired' spikes for a ProductID, alert the seller within 24 hours.
    """)

# Footer
st.markdown("---")
st.caption("Built for the Amazon Insights Pitch Battle | Data Science Consultancy Team")
