import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

from transformers import pipeline

# Load the BERT spam detection pipeline (singleton)
_bert_spam_pipeline = None

def get_bert_spam_pipeline():
    global _bert_spam_pipeline
    if _bert_spam_pipeline is None:
        _bert_spam_pipeline = pipeline('text-classification', model='mrm8488/bert-tiny-finetuned-sms-spam-detection')
    return _bert_spam_pipeline

def predict_bert_spam(texts):
    """
    texts: str or list of str
    Returns: list of dicts with 'label' and 'score'
    """
    pipe = get_bert_spam_pipeline()
    if isinstance(texts, str):
        return pipe([texts])
    return pipe(texts) 