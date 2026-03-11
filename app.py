import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
import gdown # Make sure this is in your requirements.txt!

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
    file_name = 'models.pkl'
    
    # 1. Download from Google Drive if not present locally
    if not os.path.exists(file_name):
        with st.spinner("Downloading AI Models from Google Drive..."):
            try:
                # Extracted the exact File ID from your provided URL
                file_id = '1suFu-xDXjbfcHdVA3O4rCFdftWsgG6qx'
                url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(url, file_name, quiet=False)
            except Exception as e:
                st.error(f"Error downloading model: {e}")
                return None
                
    # 2. Open the locally downloaded file
    try:
        with open(file_name, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_data():
    file_name = 'cleaned_reviews.csv'
    
    # 1. Download from Google Drive if not present locally
    if not os.path.exists(file_name):
        with st.spinner("Downloading Dataset from Google Drive..."):
            try:
                # Extracted the File ID from your CSV link
                file_id = '1XaQgTM_giihM5TVXW7cQ9Qa71gFxEcRT'
                url = f'https://drive.google.com/uc?id={file_id}'
                gdown.download(url, file_name, quiet=False)
            except Exception as e:
                st.error(f"Error downloading data: {e}")
                # Fallback empty dataframe so the app doesn't completely crash
                return pd.DataFrame(columns=['Score', 'Polarity', 'Summary', 'Text']) 
                
    # 2. Read the locally downloaded CSV
    try:
        return pd.read_csv(file_name)
    except Exception as e:
        st.warning(f"Data loading failed: {e}. Using fallback.")
        return pd.DataFrame(columns=['Score', 'Polarity', 'Summary', 'Text'])
        
models = load_models()
df = load_data()
