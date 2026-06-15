import re
import nltk
import spacy

nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords

nlp = spacy.load("en_core_web_sm")
STOPWORDS = set(stopwords.words('english'))

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    doc = nlp(text)
    tokens = [
        token.lemma_ for token in doc
        if token.text not in STOPWORDS
        and not token.is_punct
        and len(token.text) > 2
    ]
    return " ".join(tokens)
