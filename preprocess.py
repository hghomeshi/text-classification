import re

class TextCleaner():
    def __init__(self):
        pass

    def clean_text(self, text):
        text = text.lower()
        text = re.sub(r'<.*?>', '', text)
        text = re.sub(r"[^a-zA-Z0-9\-\s]", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text