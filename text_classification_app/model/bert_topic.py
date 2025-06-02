import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

from transformers import pipeline

# 20 Newsgroups topic labels
CANDIDATE_LABELS = [
    "alt.atheism", "comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware",
    "comp.sys.mac.hardware", "comp.windows.x", "misc.forsale", "rec.autos", "rec.motorcycles",
    "rec.sport.baseball", "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med",
    "sci.space", "soc.religion.christian", "talk.politics.guns", "talk.politics.mideast",
    "talk.politics.misc", "talk.religion.misc"
]

# Load the zero-shot classification pipeline (singleton)
_bert_topic_pipeline = None

def get_bert_topic_pipeline():
    global _bert_topic_pipeline
    if _bert_topic_pipeline is None:
        _bert_topic_pipeline = pipeline('zero-shot-classification', model='facebook/bart-large-mnli')
    return _bert_topic_pipeline

def predict_bert_topic(texts):
    """
    texts: str or list of str
    Returns: list of dicts with 'label' and 'score' (top prediction)
    """
    pipe = get_bert_topic_pipeline()
    if isinstance(texts, str):
        texts = [texts]
    results = pipe(texts, candidate_labels=CANDIDATE_LABELS)
    # results is a dict if single input, list of dicts if batch
    if isinstance(results, dict):
        results = [results]
    output = []
    for r in results:
        top_idx = r['scores'].index(max(r['scores']))
        output.append({'label': r['labels'][top_idx], 'score': r['scores'][top_idx]})
    return output 