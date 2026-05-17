# Sentiment Analysis using NLP and Machine Learning

## Overview

This project focuses on sentiment analysis of customer reviews using Natural Language Processing (NLP) and Machine Learning techniques. The application classifies review text into Positive, Neutral, and Negative sentiment categories based on textual content.

The project uses the Amazon Fine Food Reviews dataset and implements preprocessing, sentiment scoring, feature extraction, and classification using the Naive Bayes algorithm. A Streamlit-based web application is also developed for interactive sentiment prediction.

---

## Objectives

* Perform preprocessing on raw review text
* Analyze review polarity using VADER sentiment analysis
* Convert ratings into sentiment labels
* Train machine learning models for sentiment classification
* Develop an interactive frontend using Streamlit

---

## Dataset

Dataset: Amazon Fine Food Reviews Dataset

Source:
https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews

### Columns Used

| Column | Description                      |
| ------ | -------------------------------- |
| Text   | Review text provided by customer |
| Score  | Product rating (1–5)             |

---

## Sentiment Categories

| Rating Range | Sentiment |
| ------------ | --------- |
| 1–2          | Negative  |
| 3            | Neutral   |
| 4–5          | Positive  |

---

## Technologies Used

* Python
* Pandas
* NumPy
* NLTK
* Scikit-learn
* Streamlit

---

## NLP Techniques

* Text preprocessing
* Stopword removal
* Regular expression cleaning
* VADER sentiment scoring
* TF-IDF vectorization

---

## Machine Learning Model

### Naive Bayes Classifier

The Multinomial Naive Bayes algorithm is used for text classification due to its effectiveness on sparse textual data and NLP applications.

---

## Project Workflow

1. Load dataset
2. Clean and preprocess text
3. Apply VADER sentiment analysis
4. Convert ratings into sentiment labels
5. Balance dataset using upsampling
6. Extract features using TF-IDF
7. Train Naive Bayes classifier
8. Predict sentiment from user input
9. Display results using Streamlit dashboard

---

## Model Performance

| Model         | Accuracy |
| ------------- | -------- |
| Decision Tree | 92%      |
| Naive Bayes   | 82%      |

Naive Bayes was selected as the final model because it provides better generalization for sentiment classification tasks.

---

## Streamlit Application

The project includes a Streamlit dashboard that allows users to:

* Enter custom review text
* Predict sentiment in real time
* View model information and accuracy
* Interact with the NLP model through a simple interface

---

## Installation

### Clone Repository

```bash
git clone <repository-link>
```

### Install Required Libraries

```bash
pip install -r requirements.txt
```

### Run Streamlit Application

```bash
streamlit run app.py
```

---

## Project Structure

```text
NLP-PROJECT/
│
├── app.py
├── sentiment_project.py
├── Reviews.csv.zip
├── requirements.txt
└── README.md
```

---

## Sample Predictions

| Input Review                       | Predicted Sentiment |
| ---------------------------------- | ------------------- |
| This product is amazing and useful | Positive            |
| Worst product I have purchased     | Negative            |
| Product quality is average         | Neutral             |

---

## Future Enhancements

* Deployment using Streamlit Cloud
* Integration of Deep Learning models
* Sentiment visualization dashboards
* Support for multilingual reviews

---

## Conclusion

This project demonstrates the application of NLP and machine learning techniques for sentiment classification of textual reviews. The implemented system successfully predicts user sentiment and provides an interactive interface for real-time analysis.

---
