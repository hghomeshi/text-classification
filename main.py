import joblib
import torch

import numpy as np

from flask import Flask, jsonify, request
from preprocess import TextCleaner, LABELS_DICT
from typing import Literal


# Load model and tokenizer
model = joblib.load('DistilBERT_model.joblib')
tokenizer = joblib.load('DistilBERT_tokenizer.joblib')


app = Flask(__name__)


LABELS = Literal[
    "Dementia",
    "ALS",
    "Obsessive Compulsive Disorder",
    "Scoliosis",
    "Parkinson’s Disease",
]


def predict(description: str) -> LABELS:
    """
    Function that should take in the description text, preprocess the text, tokenize it, and return the prediction
    for the class that we identify it to.
    The possible classes are: ['Dementia', 'ALS',
                                'Obsessive Compulsive Disorder',
                                'Scoliosis', 'Parkinson’s Disease']
    """
    cleaned_text = TextCleaner.clean_text(description)
    tokenized_text = tokenizer(cleaned_text, truncation=True, return_tensors='pt')
    model.eval()
    with torch.no_grad():
        prediction = model(**tokenized_text)
        prediction = int(np.argmax(prediction.logits))

    return LABELS_DICT[prediction]


@app.route("/")
def hello_world():
    return "Hello, World!"


@app.route("/predict", methods=["POST"])
def identify_condition():
    data = request.get_json(force=True)
    
    prediction = predict(data["description"])

    return jsonify({"prediction": prediction})


if __name__ == "__main__":
    app.run()
