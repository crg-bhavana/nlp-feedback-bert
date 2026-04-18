from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"
OUTPUT_DIR = ARTIFACT_DIR / "outputs"
BERT_MODEL_DIR = MODEL_DIR / "bert_feedback_classifier"
BASELINE_MODEL_PATH = MODEL_DIR / "baseline_sentiment_model.joblib"
LABELS = [0, 1]
TEXT_COLUMN = "text"
TARGET_COLUMN = "label"
RANDOM_STATE = 42
TEST_SIZE = 0.2
MAX_SEQ_LENGTH = 128
PRETRAINED_MODEL_NAME = "distilbert-base-uncased"


def ensure_dirs() -> None:
    for path in [ARTIFACT_DIR, MODEL_DIR, OUTPUT_DIR, BERT_MODEL_DIR]:
        path.mkdir(parents=True, exist_ok=True)
