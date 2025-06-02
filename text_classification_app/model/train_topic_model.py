import os
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Ensure model directory exists
os.makedirs('model', exist_ok=True)

# Download 20 Newsgroups dataset
print('Downloading 20 Newsgroups dataset...')
data = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
X = data.data
y = data.target

# Create pipeline: TF-IDF + Multinomial Naive Bayes
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=True, stop_words='english', max_features=20000)),
    ('clf', MultinomialNB())
])

# Train model
print('Training topic classification model...')
pipeline.fit(X, y)

# Save model pipeline
joblib.dump((pipeline, data.target_names), 'model/topic_model.pkl')
print('Topic model trained and saved to model/topic_model.pkl') 