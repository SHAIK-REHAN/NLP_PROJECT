# =====================================
# STREAMLIT SENTIMENT ANALYSIS APP
# =====================================

import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.utils import resample


# =====================================
# DOWNLOAD NLTK DATA
# =====================================

nltk.download('vader_lexicon')
nltk.download('stopwords')


# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)


# =====================================
# CUSTOM CSS
# =====================================

st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

.title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    color: #1f2937;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    color: #4b5563;
    margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)


# =====================================
# TITLE SECTION
# =====================================

st.markdown(
    '<div class="title">📊 Sentiment Analysis Dashboard</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Naive Bayes + TF-IDF based NLP Sentiment Classifier</div>',
    unsafe_allow_html=True
)


# =====================================
# LOAD DATASET
# =====================================

@st.cache_data
def load_data():

    df = pd.read_csv(
        "Reviews.csv.zip",
        compression='zip'
    )

    df = df[['Text', 'Score']]

    df.dropna(inplace=True)

    # Reduce dataset size
    df = df.sample(20000, random_state=42)

    return df


# =====================================
# TEXT PREPROCESSING
# =====================================

stop_words = set(stopwords.words('english'))

def clean_text(text):

    text = text.lower()

    text = re.sub(r'[^a-zA-Z]', ' ', text)

    words = text.split()

    words = [w for w in words if w not in stop_words]

    return " ".join(words)


# =====================================
# LABEL TRANSFORMATION
# =====================================

def label_sentiment(score):

    if score < 3:
        return "Negative"

    elif score == 3:
        return "Neutral"

    else:
        return "Positive"


# =====================================
# TRAIN MODEL
# =====================================

@st.cache_resource
def train_model():

    df = load_data()

    # Clean Text
    df['clean_text'] = df['Text'].apply(clean_text)

    # =====================================
    # VADER SENTIMENT SCORE
    # =====================================

    vader = SentimentIntensityAnalyzer()

    df['vader_score'] = df['clean_text'].apply(
        lambda x: vader.polarity_scores(x)['compound']
    )

    # =====================================
    # LABELING
    # =====================================

    df['sentiment'] = df['Score'].apply(label_sentiment)

    # =====================================
    # HANDLE CLASS IMBALANCE
    # =====================================

    df_majority = df[df.sentiment == "Positive"]

    df_negative = df[df.sentiment == "Negative"]

    df_neutral = df[df.sentiment == "Neutral"]

    # Upsample Negative
    df_negative_up = resample(
        df_negative,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )

    # Upsample Neutral
    df_neutral_up = resample(
        df_neutral,
        replace=True,
        n_samples=len(df_majority),
        random_state=42
    )

    # Combine Dataset
    df_balanced = pd.concat([
        df_majority,
        df_negative_up,
        df_neutral_up
    ])

    # =====================================
    # TF-IDF VECTORIZATION
    # =====================================

    vectorizer = TfidfVectorizer(
        max_features=8000,
        ngram_range=(1, 2),
        min_df=5
    )

    X = vectorizer.fit_transform(df_balanced['clean_text'])

    y = df_balanced['sentiment']

    # =====================================
    # TRAIN TEST SPLIT
    # =====================================

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # =====================================
    # NAIVE BAYES MODEL
    # =====================================

    model = MultinomialNB(alpha=0.5)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return model, vectorizer, accuracy


# =====================================
# LOAD MODEL
# =====================================

model, vectorizer, accuracy = train_model()


# =====================================
# DASHBOARD METRICS
# =====================================

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "Naive Bayes")

with col2:
    st.metric("Accuracy", f"{accuracy:.2f}")

with col3:
    st.metric("Dataset Size", "20,000")


# =====================================
# USER INPUT SECTION
# =====================================

st.markdown("---")

st.subheader("✍ Enter Review Text")

user_input = st.text_area(
    "Type your product review below:",
    height=180,
    placeholder="Example: This product is excellent and worth buying..."
)


# =====================================
# PREDICTION
# =====================================

if st.button("Predict Sentiment"):

    if user_input.strip() == "":

        st.warning("Please enter some review text.")

    else:

        cleaned_text = clean_text(user_input)

        vector = vectorizer.transform([cleaned_text])

        prediction = model.predict(vector)[0]

        # =====================================
        # DISPLAY RESULT
        # =====================================

        st.markdown("---")

        st.subheader("📌 Prediction Result")

        if prediction == "Positive":

            st.success(f"😊 Sentiment: {prediction}")

        elif prediction == "Negative":

            st.error(f"😠 Sentiment: {prediction}")

        else:

            st.info(f"😐 Sentiment: {prediction}")


# =====================================
# SIDEBAR
# =====================================

st.sidebar.title("About Project")

st.sidebar.info(
    """
    This project performs Sentiment Analysis
    using Natural Language Processing (NLP)
    and Machine Learning.

    Model Used:
    - Naive Bayes

    Techniques:
    - TF-IDF
    - VADER
    - Text Preprocessing
    """
)

st.sidebar.success("Built using Streamlit")