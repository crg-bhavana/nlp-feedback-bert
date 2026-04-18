from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "processed" / "processed_data.csv"
MODEL_DIR = BASE_DIR / "models" / "bert_model"
RESULTS_DIR = BASE_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"

MODEL_NAME = "distilbert-base-uncased"

TEXT_COLUMN = "text"
TARGET_COLUMN = "label"

label2id = {"negative": 0, "positive": 1}
id2label = {0: "negative", 1: "positive"}


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples[TEXT_COLUMN],
        truncation=True,
        padding="max_length",
        max_length=128,
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="binary"),
    }


def main() -> None:
    print("Running BERT training...")
    print(f"Reading data from: {INPUT_PATH}")

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"Processed data not found: {INPUT_PATH}")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH).copy()
    print(f"Loaded rows: {len(df)}")
    print("Columns before sampling:", df.columns.tolist())

    negative_df = df[df[TARGET_COLUMN] == 0].sample(n=2500, random_state=42)
    positive_df = df[df[TARGET_COLUMN] == 1].sample(n=2500, random_state=42)

    df = pd.concat([negative_df, positive_df], ignore_index=True)
    print(f"Using subset of rows: {len(df)}")
    print("Columns after sampling:", df.columns.tolist())

    df["label_id"] = df[TARGET_COLUMN]

    train_df, test_df = train_test_split(
        df,
        test_size=0.25,
        random_state=42,
        stratify=df["label_id"],
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    train_ds = Dataset.from_pandas(
        train_df[[TEXT_COLUMN, "label_id"]].rename(columns={"label_id": "labels"})
    )
    test_ds = Dataset.from_pandas(
        test_df[[TEXT_COLUMN, "label_id"]].rename(columns={"label_id": "labels"})
    )

    train_ds = train_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)
    test_ds = test_ds.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    cols = ["input_ids", "attention_mask", "labels"]
    train_ds.set_format(type="torch", columns=cols)
    test_ds.set_format(type="torch", columns=cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label=id2label,
        label2id=label2id,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR / "checkpoints"),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        num_train_epochs=2,
        weight_decay=0.01,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()

    pred_output = trainer.predict(test_ds)
    y_true = pred_output.label_ids
    y_pred = np.argmax(pred_output.predictions, axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="binary")
    cm = confusion_matrix(y_true, y_pred)

    metrics["accuracy"] = acc
    metrics["f1"] = f1

    with open(RESULTS_DIR / "bert_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["negative", "positive"])
    disp.plot()
    plt.title("BERT Confusion Matrix")
    plt.savefig(PLOTS_DIR / "bert_confusion_matrix.png", bbox_inches="tight")
    plt.close()

    trainer_state_file = None
    checkpoint_dir = MODEL_DIR / "checkpoints"
    if checkpoint_dir.exists():
        checkpoint_folders = sorted(checkpoint_dir.glob("checkpoint-*"))
        if checkpoint_folders:
            trainer_state_file = checkpoint_folders[-1] / "trainer_state.json"

    if trainer_state_file and trainer_state_file.exists():
        with open(trainer_state_file, "r") as f:
            state = json.load(f)

        log_history = state.get("log_history", [])
        train_epochs, train_losses = [], []
        eval_epochs, eval_losses = [], []
        eval_accuracies, eval_f1s = [], []

        for entry in log_history:
            if "loss" in entry and "epoch" in entry:
                train_epochs.append(entry["epoch"])
                train_losses.append(float(entry["loss"]))
            if "eval_loss" in entry and "epoch" in entry:
                eval_epochs.append(entry["epoch"])
                eval_losses.append(float(entry["eval_loss"]))
                if "eval_accuracy" in entry:
                    eval_accuracies.append(float(entry["eval_accuracy"]))
                if "eval_f1" in entry:
                    eval_f1s.append(float(entry["eval_f1"]))

        if train_epochs and train_losses:
            plt.figure()
            plt.plot(train_epochs, train_losses, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Training Loss")
            plt.title("BERT Training Loss")
            plt.savefig(PLOTS_DIR / "bert_training_loss.png", bbox_inches="tight")
            plt.close()

        if eval_epochs and eval_losses:
            plt.figure()
            plt.plot(eval_epochs, eval_losses, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Validation Loss")
            plt.title("BERT Validation Loss")
            plt.savefig(PLOTS_DIR / "bert_validation_loss.png", bbox_inches="tight")
            plt.close()

        if eval_epochs and eval_accuracies:
            plt.figure()
            plt.plot(eval_epochs[:len(eval_accuracies)], eval_accuracies, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("BERT Validation Accuracy")
            plt.savefig(PLOTS_DIR / "bert_validation_accuracy.png", bbox_inches="tight")
            plt.close()

        if eval_epochs and eval_f1s:
            plt.figure()
            plt.plot(eval_epochs[:len(eval_f1s)], eval_f1s, marker="o")
            plt.xlabel("Epoch")
            plt.ylabel("F1 Score")
            plt.title("BERT Validation F1")
            plt.savefig(PLOTS_DIR / "bert_validation_f1.png", bbox_inches="tight")
            plt.close()

    model.save_pretrained(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)

    print("BERT evaluation metrics:")
    print(metrics)
    print(f"Saved BERT model to: {MODEL_DIR}")
    print(f"Saved metrics to: {RESULTS_DIR / 'bert_metrics.json'}")
    print(f"Saved plots to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()