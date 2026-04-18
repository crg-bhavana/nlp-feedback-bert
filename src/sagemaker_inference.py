"""SageMaker-compatible inference entrypoint.

This file can be packaged with the model artifact and used by a custom inference container.
"""

import json
from pathlib import Path

from transformers import pipeline

MODEL = None
MODEL_DIR = Path("/opt/ml/model")


def model_fn(model_dir: str):
    global MODEL
    MODEL = pipeline("text-classification", model=model_dir, tokenizer=model_dir)
    return MODEL


def input_fn(request_body, request_content_type):
    if request_content_type != "application/json":
        raise ValueError("Unsupported content type")
    payload = json.loads(request_body)
    return payload["inputs"]


def predict_fn(input_data, model):
    return model(input_data)


def output_fn(prediction, accept):
    if accept != "application/json":
        raise ValueError("Unsupported accept type")
    return json.dumps(prediction), accept
