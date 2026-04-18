import json

import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from transformers import pipeline

from src.config import BASELINE_MODEL_PATH, BERT_MODEL_DIR, OUTPUT_DIR, RANDOM_STATE, TARGET_COLUMN, TEST_SIZE, TEXT_COLUMN


def compare_models(df):
    x_train, x_test, y_train, y_test = train_test_split(
        df[TEXT_COLUMN],
        df[TARGET_COLUMN],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df[TARGET_COLUMN],
    )

    baseline = joblib.load(BASELINE_MODEL_PATH)
    baseline_preds = baseline.predict(x_test)
    bert_pipe = pipeline("text-classification", model=str(BERT_MODEL_DIR), tokenizer=str(BERT_MODEL_DIR))
    bert_preds = [pred["label"].lower() for pred in bert_pipe(list(x_test))]

    results = {
        "baseline": {
            "accuracy": accuracy_score(y_test, baseline_preds),
            "f1_weighted": f1_score(y_test, baseline_preds, average="weighted"),
        },
        "bert": {
            "accuracy": accuracy_score(y_test, bert_preds),
            "f1_weighted": f1_score(y_test, bert_preds, average="weighted"),
        },
    }

    with open(OUTPUT_DIR / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nModel Comparison\n")
    print(json.dumps(results, indent=2))
    return results
