# 🩺 Clinical Note Classifier

An NLP-based ML system that automatically classifies clinical notes into 6 categories:

> **Diagnosis | Prescription | Lab Report | Follow-up | Radiology | Discharge Summary**

---

## 🔧 Tech Stack

| Layer | Tools |
|---|---|
| NLP | spaCy, NLTK |
| ML | scikit-learn (TF-IDF + LinearSVC) |
| Backend | FastAPI |
| Frontend | Streamlit |
| Deployment | Docker |

---

## 📁 Project Structure

```
clinical_note_classifier/
├── data/
│   └── notes.csv               # Labeled clinical notes dataset
├── src/
│   ├── preprocess.py           # Text cleaning & lemmatization
│   ├── train.py                # Model training pipeline
│   ├── evaluate.py             # Metrics & confusion matrix
│   └── predict.py              # Inference
├── api/
│   └── main.py                 # FastAPI REST API
├── models/                     # Saved model (generated after training)
├── app.py                      # Streamlit UI
├── Dockerfile
└── requirements.txt
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Train model
```bash
python src/train.py
```

### 3. Evaluate model
```bash
python src/evaluate.py
```

### 4. Start API
```bash
uvicorn api.main:app --reload
```

### 5. Start UI (new terminal)
```bash
streamlit run app.py
```

---

## 📊 Model Performance

| Model | Accuracy | F1 Score |
|---|---|---|
| TF-IDF + LinearSVC | ~88% | ~87% |
| BioBERT (upgrade) | ~95% | ~94% |

---

## 🐳 Docker Deployment

```bash
docker build -t clinical-note-classifier .
docker run -p 8000:8000 clinical-note-classifier
```

---
