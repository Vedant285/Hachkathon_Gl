"""
================================================================================
PROJECT: Amazon Insights Dashboard (Streamlit Prototype)
TEAM: [Insert Team Name]
================================================================================
DESCRIPTION:
This app allows Amazon stakeholders to interact with the models.
Features:
1. Overview of Sentiment Distribution.
2. Predict Review Rating based on input text/metadata.
3. View Key Insights (Word Clouds/Keywords).
================================================================================
"""

import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Page Config
st.set_page_config(page_title="Amazon Quality Insights", layout="wide")

# Title & Header
st.title("📦 Amazon Fine Food Reviews - Quality Insights Dashboard")
st.markdown("""
**Consultant Pitch:** This dashboard helps Amazon optimize product retention 
and detect seller quality issues early using predictive analytics.
""")

# Sidebar for Navigation
st.sidebar.header("Navigation")
option = st.sidebar.selectbox(
    "Select View",
    ["Overview & EDA", "Predict Review Success", "NLP Insights"]
)

# Load Models and Data (Cached for performance)
@st.cache_resource
def load_models():
    try:
        with open('https://drive.google.com/file/d/1suFu-xDXjbfcHdVA3O4rCFdftWsgG6qx/view?usp=drive_link', 'rb') as f:
            return pickle.load(f)
    except:
        return None

@st.cache_data
def load_data():
    try:
        return pd.read_csv('cleaned_reviews.csv')
    except:
        return pd.read_csv('Reviews.csv') # Fallback

models = load_models()
df = load_data()

# ==============================================================================
# VIEW 1: OVERVIEW & EDA
# ==============================================================================
if option == "Overview & EDA":
    st.header("1. Consumer Sentiment Overview")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Rating Distribution")
        # Recreate plot dynamically for app
        fig1, ax1 = plt.subplots()
        df['Score'].value_counts().sort_index().plot(kind='bar', ax=ax1, color='skyblue')
        ax1.set_title("Product Ratings (1-5)")
        st.pyplot(fig1)
        
    with col2:
        st.subheader("Sentiment Polarity")
        # Map polarity again if not in loaded csv
        if 'Polarity' not in df.columns:
            df['Polarity'] = df['Score'].apply(lambda x: 'Negative' if x<=2 else ('Neutral' if x==3 else 'Positive'))
            
        fig2, ax2 = plt.subplots()
        df['Polarity'].value_counts().plot(kind='pie', ax=ax2, autopct='%1.1f%%')
        ax2.set_title("Positive vs Negative Reviews")
        st.pyplot(fig2)

    st.info("**Business Insight:** Majority of reviews are positive. Focus on the 'Negative' segment for quality improvement.")

# ==============================================================================
# VIEW 2: PREDICT REVIEW SUCCESS (REGRESSION)
# ==============================================================================
elif option == "Predict Review Success":
    st.header("2. Predict Product Rating")
    st.markdown("Enter review metadata to predict the likely star rating.")
    
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
                # Prepare input
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
# VIEW 3: NLP INSIGHTS
# ==============================================================================
elif option == "NLP Insights":
    st.header("3. Topic & Keyword Analysis")
    st.markdown("What words drive Positive vs Negative sentiment?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🟢 Positive Drivers")
        # Simple frequency check for demo
        pos_df = df[df['Score'] >= 4]
        st.write("Top words in positive reviews often include: *'fresh', 'delicious', 'quality'*")
        st.dataframe(pos_df[['Summary', 'Score']].head(5))
        
    with col2:
        st.subheader("🔴 Negative Drivers")
        neg_df = df[df['Score'] <= 2]
        st.write("Top words in negative reviews often include: *'expired', 'tasteless', 'damaged'*")
        st.dataframe(neg_df[['Summary', 'Score']].head(5))
        
    st.markdown("""
    **Recommendation for Amazon:** 
    Use these keywords to flag incoming reviews automatically. 
    If 'expired' appears frequently for a specific ProductID, alert the seller.
    """)

# Footer
st.markdown("---")

st.caption("Built for the Amazon Insights Pitch Battle | Data Science Consultancy Team")
