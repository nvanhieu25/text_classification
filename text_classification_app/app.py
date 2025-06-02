from flask import Flask, render_template, request, redirect, url_for
import joblib
import pandas as pd
import os
from model.bert_sentiment import predict_bert_sentiment
from model.bert_spam import predict_bert_spam
from model.bert_topic import predict_bert_topic

app = Flask(__name__)
spam_model = joblib.load('model/model.pkl')
sentiment_model = joblib.load('model/sentiment_model.pkl')
topic_model_pipeline, topic_names = joblib.load('model/topic_model.pkl')

def topic_predict(texts):
    preds = topic_model_pipeline.predict(texts)
    probs = topic_model_pipeline.predict_proba(texts).max(axis=1)
    labels = [topic_names[i] for i in preds]
    return labels, probs

ALLOWED_EXTENSIONS = {'csv'}

MODEL_OPTIONS = {
    'spam': {
        'name': 'Spam Detection (Logistic Regression)',
        'model': spam_model,
        'type': 'default'
    },
    'bert_spam': {
        'name': 'Spam Detection (BERT)',
        'model': None,
        'type': 'bert_spam'
    },
    'sentiment': {
        'name': 'Sentiment Analysis (Logistic Regression)',
        'model': sentiment_model,
        'type': 'default'
    },
    'bert_sentiment': {
        'name': 'Sentiment Analysis (BERT)',
        'model': None,
        'type': 'bert_sentiment'
    },
    'topic': {
        'name': 'Topic Classification (Naive Bayes)',
        'model': topic_model_pipeline,
        'type': 'topic'
    },
    'bert_topic': {
        'name': 'Topic Classification (BERT)',
        'model': None,
        'type': 'bert_topic'
    }
}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    batch_results = None
    error = None
    selected_model = request.form.get('model', 'spam')
    if request.method == 'POST':
        text = request.form.get('text', '').strip()
        file = request.files.get('file')
        model_info = MODEL_OPTIONS.get(selected_model, MODEL_OPTIONS['spam'])
        model = model_info['model']
        model_type = model_info['type']
        if text and not file:
            if model_type == 'topic':
                label, prob = topic_predict([text])
                result = {'label': label[0], 'confidence': f"{prob[0]*100:.2f}%"}
            elif model_type == 'bert_sentiment':
                bert_result = predict_bert_sentiment(text)[0]
                result = {'label': bert_result['label'], 'confidence': f"{bert_result['score']*100:.2f}%"}
            elif model_type == 'bert_spam':
                bert_result = predict_bert_spam(text)[0]
                result = {'label': bert_result['label'], 'confidence': f"{bert_result['score']*100:.2f}%"}
            elif model_type == 'bert_topic':
                bert_result = predict_bert_topic(text)[0]
                result = {'label': bert_result['label'], 'confidence': f"{bert_result['score']*100:.2f}%"}
            else:
                pred = model.predict([text])[0]
                prob = model.predict_proba([text]).max()
                result = {'label': pred, 'confidence': f"{prob*100:.2f}%"}
        elif file and allowed_file(file.filename):
            try:
                df = pd.read_csv(file)
                if 'text' not in df.columns:
                    error = "CSV must have a 'text' column."
                else:
                    texts = df['text'].astype(str).tolist()
                    if model_type == 'topic':
                        labels, probs = topic_predict(texts)
                        batch_results = [
                            {'text': t, 'label': l, 'confidence': f"{p*100:.2f}%"}
                            for t, l, p in zip(texts, labels, probs)
                        ]
                    elif model_type == 'bert_sentiment':
                        bert_results = predict_bert_sentiment(texts)
                        batch_results = [
                            {'text': t, 'label': r['label'], 'confidence': f"{r['score']*100:.2f}%"}
                            for t, r in zip(texts, bert_results)
                        ]
                    elif model_type == 'bert_spam':
                        bert_results = predict_bert_spam(texts)
                        batch_results = [
                            {'text': t, 'label': r['label'], 'confidence': f"{r['score']*100:.2f}%"}
                            for t, r in zip(texts, bert_results)
                        ]
                    elif model_type == 'bert_topic':
                        bert_results = predict_bert_topic(texts)
                        batch_results = [
                            {'text': t, 'label': r['label'], 'confidence': f"{r['score']*100:.2f}%"}
                            for t, r in zip(texts, bert_results)
                        ]
                    else:
                        preds = model.predict(texts)
                        probs = model.predict_proba(texts).max(axis=1)
                        batch_results = [
                            {'text': t, 'label': l, 'confidence': f"{p*100:.2f}%"}
                            for t, l, p in zip(texts, preds, probs)
                        ]
            except Exception as e:
                error = f"Error processing file: {e}"
        elif not text and not file:
            error = "Please enter text or upload a CSV file."
        else:
            error = "Invalid file type. Only CSV files are allowed."
    return render_template('index.html', result=result, batch_results=batch_results, error=error, selected_model=selected_model, model_options=MODEL_OPTIONS)

if __name__ == '__main__':
    app.run(debug=True) 