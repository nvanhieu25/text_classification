import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib
import urllib.request

# Ensure model directory exists
os.makedirs('model', exist_ok=True)

# Download SMS Spam Collection Dataset if not present
DATASET_URL = 'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv'
DATASET_PATH = 'model/spam.tsv'

if not os.path.exists(DATASET_PATH):
    print('Downloading dataset...')
    urllib.request.urlretrieve(DATASET_URL, DATASET_PATH)

# Load dataset
df = pd.read_csv(DATASET_PATH, sep='\t', header=None, names=['label', 'text'])

# Basic preprocessing: remove NaN, strip whitespace
df = df.dropna(subset=['text'])
df['text'] = df['text'].astype(str).str.strip()

# Create pipeline: TF-IDF + Naive Bayes
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=True, stop_words='english')),
    ('clf', MultinomialNB())
])

# Train model
pipeline.fit(df['text'], df['label'])

# Save model pipeline
joblib.dump(pipeline, 'model/model.pkl')
print('Model trained and saved to model/model.pkl') 