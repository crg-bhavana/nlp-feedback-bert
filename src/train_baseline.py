from pathlib import Path
import json
import joblib
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    f1_score,
)
from sklearn.model_selection import train_test_split

from src.config import TARGET_COLUMN, TEXT_COLUMN


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "processed_data.csv"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

VECTORIZER_PATH = MODELS_DIR / "baseline_vectorizer.pkl"
MODEL_PATH = MODELS_DIR / "baseline_model.pkl"
BASELINE_METRICS_PATH = RESULTS_DIR / "baseline_metrics.json"


def main() -> None:
    print("Running baseline training...")
    print(f"Reading data from: {INPUT_PATH}")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Processed data not found: {INPUT_PATH}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded rows: {len(df)}")

    X = df[TEXT_COLUMN].astype(str)
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="binary")
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    metrics = {
        "accuracy": acc,
        "f1": f1,
        "classification_report": report,
    }

    with open(BASELINE_METRICS_PATH, "w") as f:
        json.dump(metrics, f, indent=2)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["negative", "positive"])
    disp.plot()
    plt.title("Baseline Confusion Matrix")
    plt.savefig(PLOTS_DIR / "baseline_confusion_matrix.png", bbox_inches="tight")
    plt.close()

    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(model, MODEL_PATH)

    print(f"Baseline Accuracy: {acc:.4f}")
    print(f"Baseline F1: {f1:.4f}")
    print(f"Saved baseline metrics to: {BASELINE_METRICS_PATH}")
    print(f"Saved confusion matrix to: {PLOTS_DIR / 'baseline_confusion_matrix.png'}")
    print(f"Saved vectorizer to: {VECTORIZER_PATH}")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()