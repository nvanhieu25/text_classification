import os
import numpy as np
from keras.datasets import imdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

# Ensure model directory exists
os.makedirs('model', exist_ok=True)

# Load IMDB dataset from Keras
print('Downloading IMDB dataset...')
num_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=num_words)

# Get the word index from Keras
word_index = imdb.get_word_index()
index_word = {v+3: k for k, v in word_index.items()}
index_word[0] = '<PAD>'
index_word[1] = '<START>'
index_word[2] = '<UNK>'
index_word[3] = '<UNUSED>'

def decode_review(encoded_review):
    return ' '.join([index_word.get(i, '?') for i in encoded_review])

# Convert integer-encoded reviews back to text
x_train_text = [decode_review(review) for review in x_train]
x_test_text = [decode_review(review) for review in x_test]
y_train = np.array(['positive' if label == 1 else 'negative' for label in y_train])
y_test = np.array(['positive' if label == 1 else 'negative' for label in y_test])

# Create pipeline: TF-IDF + Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=True, stop_words='english', max_features=20000)),
    ('clf', LogisticRegression(max_iter=200))
])

# Train model
print('Training sentiment analysis model...')
pipeline.fit(x_train_text, y_train)

# Save model pipeline
joblib.dump(pipeline, 'model/sentiment_model.pkl')
print('Sentiment model trained and saved to model/sentiment_model.pkl') 