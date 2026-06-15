import pickle
from src.preprocess import clean_text

with open("models/classifier.pkl", "rb") as f:
    saved = pickle.load(f)

pipeline = saved["pipeline"]

LABEL_COLORS = {
    "Diagnosis":         "🔴",
    "Prescription":      "💊",
    "Lab Report":        "🔬",
    "Follow-up":         "📅",
    "Radiology":         "🩻",
    "Discharge Summary": "📋"
}

def predict(text: str) -> dict:
    cleaned = clean_text(text)
    label = pipeline.predict([cleaned])[0]
    scores = pipeline.decision_function([cleaned])[0]
    confidence = float(max(scores))
    return {
        "label": label,
        "emoji": LABEL_COLORS.get(label, "📄"),
        "confidence": round(confidence, 3),
        "all_classes": dict(zip(pipeline.classes_, scores.tolist()))
    }
