import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import joblib
import spacy
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv("../data/trials.csv")
nlp = spacy.load("en_core_web_sm")

# Preprocessing
def preprocess(text):
    doc = nlp(text)
    return " ".join([
        token.lemma_.lower() 
        for token in doc 
        if not token.is_stop and not token.is_punct
    ])

df["clean_text"] = df["description"].apply(preprocess)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["label"], test_size=0.2, stratify=df["label"], random_state=42
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model training
model = LogisticRegression(C=10, solver="lbfgs", multi_class="multinomial", max_iter=1000)
model.fit(X_train_tfidf, y_train)

# Evaluation
print(classification_report(y_test, model.predict(X_test_tfidf)))

# Generate confusion matrix
cm = confusion_matrix(y_test, model.predict(X_test_tfidf))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap="Blues", xticks_rotation=45)
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("../notebooks/confusion_matrix.png")
plt.close()

# Save artifacts
joblib.dump(model, "../models/model.joblib")
joblib.dump(vectorizer, "../models/tfidf_vectorizer.joblib")