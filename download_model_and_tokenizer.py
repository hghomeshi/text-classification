import joblib

from transformers import AutoTokenizer,AutoModelForSequenceClassification


# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("hghomeshi/DistilBERT_text_classification_trials")
model = AutoModelForSequenceClassification.from_pretrained("hghomeshi/DistilBERT_text_classification_trials", num_labels=5)

# Dump tokenizer and model as joblib file
joblib.dump(model, 'DistilBERT_model.joblib')
joblib.dump(tokenizer,'DistilBERT_tokenizer.joblib')
