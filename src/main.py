from flask import Flask, request, jsonify
import joblib
import spacy

app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")
model = joblib.load("../models/model.joblib")
vectorizer = joblib.load("../models/tfidf_vectorizer.joblib")

def preprocess(text):
    doc = nlp(text)
    return " ".join([
        token.lemma_.lower() 
        for token in doc 
        if not token.is_stop and not token.is_punct
    ])

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data["text"]
    clean_text = preprocess(text)
    X = vectorizer.transform([clean_text])
    pred = model.predict(X)[0]
    return jsonify({"label": pred})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)