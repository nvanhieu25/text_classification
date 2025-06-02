# Simple Text Classification Web Application

## Overview
This is a simple web application that allows users to classify text into predefined categories (e.g., spam or not spam) using a pre-trained machine learning model. Users can input text directly or upload a CSV file for batch classification. The app displays predictions and confidence scores with a modern, user-friendly interface.

## Features
- Input text via a web form or upload a CSV file (with a `text` column)
- Classifies text using a Naive Bayes model trained on the SMS Spam Collection dataset
- Displays predicted label and confidence score for each input
- Batch results shown in a table for CSV uploads
- Clear error messages and visual indicators

## Demo
![screenshot](demo_screenshot.png) <!-- Add a screenshot if available -->

## Requirements
- Python 3.7+
- pip

### Python Packages
- Flask
- scikit-learn
- pandas
- joblib

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## Setup & Usage
1. **Clone the repository**
2. **Train the model** (one-time):
    ```bash
    python model/train_model.py
    ```
   This will download the SMS Spam dataset, train the model, and save it to `model/model.pkl`.
3. **Run the web app:**
    ```bash
    python app.py
    ```
4. **Open your browser** and go to [http://127.0.0.1:5000/](http://127.0.0.1:5000/)

## How to Use
- **Single Text:** Enter your text in the textarea and click "Classify". The result will show the predicted label and confidence.
- **Batch (CSV):** Upload a CSV file with a column named `text`. Each row will be classified and results shown in a table.

### Example CSV
| text                       |
|----------------------------|
| Free entry in 2 a wkly comp|
| Hey, are we still meeting? |

### Example Output
- **Label:** spam
- **Confidence:** 98.23%

## Model & Dataset
- **Dataset:** [SMS Spam Collection Dataset](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)
- **Model:** Naive Bayes with TF-IDF vectorization (scikit-learn)
- **Preprocessing:** Lowercasing, stopword removal

## Edge Cases Handled
- Empty input or file: error message
- Invalid file type: error message
- CSV missing `text` column: error message
- Handles special characters and long/short texts

## Customization
- To use a different dataset or model, modify `model/train_model.py` and retrain.
- For more categories, use a multi-class dataset and retrain the model.

## License
MIT License

---
**Author:** Nvanhieu25
