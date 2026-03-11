import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="Amazon Insights | Pitch Battle", layout="wide")

# --- CUSTOM CSS FOR "AMAZON" FEEL ---
st.markdown("""
    <style>
    .main { background-color: #f3f3f3; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 5px solid #ff9900; }
    h1, h2, h3 { color: #232f3e; font-family: 'Amazon Ember', sans-serif; }
    .stButton>button { background-color: #ff9900; color: white; border-radius: 5px; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# --- LOAD ASSETS ---
@st.cache_resource
def load_models():
    reg_model = joblib.load('models/regression_pipeline.pkl')
    cls_model = joblib.load('models/classification_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    stats = joblib.load('models/summary_stats.pkl')
    return reg_model, cls_model, vectorizer, stats

try:
    reg_pipe, cls_model, tfidf, stats = load_models()
except:
    st.error("Models not found. Please run the training script first!")
    st.stop()

# --- SIDEBAR NAVIGATION ---
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/a/a9/Amazon_logo.svg", width=150)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Executive Summary", "Deep EDA", "Sentiment Predictor", "Business Impact"])

# --- PAGE 1: EXECUTIVE SUMMARY ---
if page == "Executive Summary":
    st.title("🚀 Amazon Fine Food Insights")
    st.markdown("### Powering Product Quality through AI-Driven Analytics")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Reviews Analyzed", f"{stats['total_reviews']:,}")
    col2.metric("Avg. Product Rating", f"{stats['avg_score']:.2f}/5")
    col3.metric("Model R² (Rating)", f"{stats['r2_score']:.2f}")
    col4.metric("Sentiment Accuracy", f"{stats['classification_accuracy']:.1%}")

    st.info("**Consultant Note:** Our NLP pipeline identifies early warning signs in reviews before they impact the bottom line.")
    
    # Showcase WordCloud
    st.subheader("Frequent Customer Themes")
    st.image('plots/wordcloud.png', use_container_width=True, caption="Top Keywords in Fine Food Reviews")

# --- PAGE 2: DEEP EDA ---
elif page == "Deep EDA":
    st.title("📊 Distribution & Trends")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Rating Distribution")
        st.image('plots/score_distribution.png')
        st.write("Insight: A high density of 5-star reviews suggests a 'positivity bias' in organic food consumers.")
    
    with col_b:
        st.subheader("Helpfulness vs. Sentiment")
        st.write("Our analysis shows that 'Neutral' reviews (3-stars) are often perceived as more helpful by other shoppers, as they provide balanced pros/cons.")
        # Add a placeholder for a dynamic plot if needed

# --- PAGE 3: SENTIMENT PREDICTOR ---
elif page == "Sentiment Predictor":
    st.title("🔮 Real-Time Review Intelligence")
    st.markdown("Test our models below. Enter a mock review to see how our AI classifies it.")

    user_review = st.text_area("User Review Text:", placeholder="The organic honey was great, but the packaging arrived broken...")
    
    col1, col2 = st.columns(2)
    
    if user_review:
        # Regression: Predict Score
        # Formatting input for the pipeline (needs to match training features)
        input_df = pd.DataFrame({
            'Cleaned_Text': [user_review.lower()],
            'HelpfulnessRatio': [0.5], # Default for simulation
            'Text_Word_Count': [len(user_review.split())]
        })
        
        predicted_score = reg_pipe.predict(input_df)[0]
        predicted_score = np.clip(predicted_score, 1, 5)

        # Classification: Predict Sentiment
        vec_text = tfidf.transform([user_review.lower()])
        sentiment = cls_model.predict(vec_text)[0]

        with col1:
            st.metric("Predicted Star Rating", f"{predicted_score:.1f} ★")
        
        with col2:
            color = "green" if sentiment == "Positive" else "red" if sentiment == "Negative" else "orange"
            st.markdown(f"### Predicted Sentiment: <span style='color:{color}'>{sentiment}</span>", unsafe_allow_html=True)

# --- PAGE 4: BUSINESS IMPACT ---
elif page == "Business Impact":
    st.title("💼 Strategic Recommendations")
    
    st.subheader("1. Early Issue Detection")
    st.write("By monitoring the NLP keyword clouds for words like 'broken', 'stale', or 'expired', Amazon can flag sellers for quality audits automatically.")
    
    st.subheader("2. Seller Quality Score")
    st.write("We propose a new 'Trust Metric' based on the ratio of Helpfulness to Predicted Sentiment to identify authentic reviews vs. bots.")
    
    st.success("Our solution provides a scalable way to maintain the 'Gold Standard' of Amazon product quality.")
