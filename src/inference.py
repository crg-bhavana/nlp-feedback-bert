from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models" / "bert_model"

ID2LABEL = {0: "negative", 1: "positive"}


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.eval()
    return tokenizer, model


def predict(text: str, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128,
    )

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_id = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_id].item()

    label = ID2LABEL.get(pred_id, str(pred_id))
    return label, confidence


def main():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")

    tokenizer, model = load_model()

    sample_texts = [
        "The product is amazing and works perfectly.",
        "The experience was okay, nothing special.",
        "Very disappointed with the service and quality.",
    ]

    for text in sample_texts:
        label, confidence = predict(text, tokenizer, model)
        print(f"\nInput: {text}")
        print(f"Prediction: {label}")
        print(f"Confidence: {confidence:.4f}")


if __name__ == "__main__":
    main()