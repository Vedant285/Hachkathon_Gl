import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score, classification_report, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud

# Create directories for assets (Required for submission organization)
os.makedirs('models', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# ==========================================
# 1. DATA LOADING & PREPROCESSING
# ==========================================
print("Loading Data...")
df = pd.read_csv('Reviews.csv') 

# --- Data Cleaning for NLP ---
def clean_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove special chars
        text = re.sub(r'\s+', ' ', text).strip() # Remove extra spaces
        return text
    return ""

df['Cleaned_Text'] = df['Text'].apply(clean_text)
df['Text_Word_Count'] = df['Cleaned_Text'].apply(lambda x: len(x.split()))
df['HelpfulnessRatio'] = df['HelpfulnessNumerator'] / (df['HelpfulnessDenominator'] + 1)

# Define Sentiment Classes for Classification (Target for Classification Model)
def get_sentiment(score):
    if score <= 2: return 'Negative'
    elif score == 3: return 'Neutral'
    else: return 'Positive'

df['Sentiment'] = df['Score'].apply(get_sentiment)

# ==========================================
# 2. EDA (Exploratory Data Analysis)
# ==========================================
print("Generating EDA Plots...")

# Plot 1: Score Distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Score', data=df, palette='viridis')
plt.title('Distribution of Product Ratings')
plt.savefig('plots/score_distribution.png')
plt.close()

# Plot 2: WordCloud for NLP Insight
text_data = " ".join(review for review in df['Cleaned_Text'].astype(str))
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text_data)
wordcloud.to_file('plots/wordcloud.png')

# ==========================================
# 3. REGRESSION MODEL (Predict Rating)
# ==========================================
print("Training Regression Pipeline...")

# Features: Text (NLP) + Metadata (Helpfulness, Word Count)
X = df[['Cleaned_Text', 'HelpfulnessRatio', 'Text_Word_Count']]
y = df['Score']

# TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SCALING & VECTORIZATION (Your Snippet Integrated)
preprocessor = ColumnTransformer(
    transformers=[
        ('text', TfidfVectorizer(max_features=2800, ngram_range=(1,2), stop_words='english'), 'Cleaned_Text'), 
        ('num', StandardScaler(), ['HelpfulnessRatio', 'Text_Word_Count']) 
    ]
)

# PIPELINE
regress = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

regress.fit(X_train, y_train)
predictions = regress.predict(X_test)

# Clip predictions to valid rating range (1-5)
predictions = np.clip(predictions, 1, 5)

# Metrics
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Regression Mean Absolute Error: {mae:.2f}")
print(f"Regression R2 Score: {r2:.2f}") 

# Save Regression Pipeline
joblib.dump(regress, 'models/regression_pipeline.pkl')

# ==========================================
# 4. CLASSIFICATION MODEL (Sentiment Polarity)
# ==========================================
print("Training Classification Model...")
# Using TF-IDF on Text for NLP-based classification
tfidf_cls = TfidfVectorizer(max_features=5000, stop_words='english')
X_text_cls = tfidf_cls.fit_transform(df['Cleaned_Text'])
y_cls = df['Sentiment']

X_train_cls, X_test_cls, y_train_cls, y_test_cls = train_test_split(X_text_cls, y_cls, test_size=0.2, random_state=42)

cls_model = LogisticRegression(max_iter=1000)
cls_model.fit(X_train_cls, y_train_cls)

y_pred_cls = cls_model.predict(X_test_cls)
accuracy = accuracy_score(y_test_cls, y_pred_cls)

print(f"Classification Accuracy: {accuracy:.4f}")
print(classification_report(y_test_cls, y_pred_cls))

# Save Classification Model & Vectorizer
joblib.dump(cls_model, 'models/classification_model.pkl')
joblib.dump(tfidf_cls, 'models/tfidf_vectorizer.pkl')

# ==========================================
# 5. SAVE SUMMARY STATS FOR STREAMLIT
# ==========================================
summary_stats = {
    'total_reviews': len(df),
    'avg_score': df['Score'].mean(),
    'r2_score': r2,
    'classification_accuracy': accuracy
}
joblib.dump(summary_stats, 'models/summary_stats.pkl')

print("Analysis Complete. Files saved to 'models' and 'plots' folders.")
