import pandas as pd
import pickle
import os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from src.preprocess import clean_text

def train():
    df = pd.read_csv("data/notes.csv")
    df["clean_text"] = df["text"].apply(clean_text)

    X = df["clean_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 3),
            max_features=10000,
            sublinear_tf=True
        )),
        ("clf", LinearSVC(C=1.0, max_iter=1000))
    ])

    pipeline.fit(X_train, y_train)

    cv_scores = cross_val_score(pipeline, X, y, cv=5, scoring='f1_weighted')
    print(f"✅ CV F1 Score: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print(f"✅ Test Accuracy: {pipeline.score(X_test, y_test):.3f}")

    os.makedirs("models", exist_ok=True)
    with open("models/classifier.pkl", "wb") as f:
        pickle.dump({
            "pipeline": pipeline,
            "classes": list(pipeline.classes_),
            "X_test": X_test,
            "y_test": y_test
        }, f)

    print("✅ Model saved to models/classifier.pkl")
    return pipeline

if __name__ == "__main__":
    train()
