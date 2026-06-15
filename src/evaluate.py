import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

def evaluate():
    with open("models/classifier.pkl", "rb") as f:
        saved = pickle.load(f)

    pipeline = saved["pipeline"]
    X_test   = saved["X_test"]
    y_test   = saved["y_test"]

    y_pred = pipeline.predict(X_test)

    print("\n📊 Classification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
    disp = ConfusionMatrixDisplay(cm, display_labels=pipeline.classes_)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45)
    plt.title("Clinical Note Classifier — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png")
    print("✅ Confusion matrix saved to models/confusion_matrix.png")

if __name__ == "__main__":
    evaluate()
