import re


LABELS_DICT = {0: "ALS",
          1: "Dementia",
          2: "Obsessive Compulsive Disorder",
          3: "Parkinson’s Disease",
          4: "Scoliosis"
          }


class TextCleaner():
    def __init__(self):
        pass

    def clean_text(text):
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r"[^a-zA-Z0-9\-\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
