import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

from transformers import pipeline

# Load the BERT sentiment analysis pipeline (singleton)
_bert_pipeline = None

def get_bert_sentiment_pipeline():
    global _bert_pipeline
    if _bert_pipeline is None:
        _bert_pipeline = pipeline('sentiment-analysis', model='distilbert-base-uncased-finetuned-sst-2-english')
    return _bert_pipeline

def predict_bert_sentiment(texts):
    """
    texts: str or list of str
    Returns: list of dicts with 'label' and 'score'
    """
    pipe = get_bert_sentiment_pipeline()
    if isinstance(texts, str):
        return pipe([texts])
    return pipe(texts) 