import re
from pathlib import Path

import pandas as pd

from src.config import LABELS, TARGET_COLUMN, TEXT_COLUMN

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_PATH = BASE_DIR / "data" / "sample" / "imdb_train.csv"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_PATH = OUTPUT_DIR / "processed_data.csv"


def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_and_prepare_data(input_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    required = {TEXT_COLUMN, TARGET_COLUMN}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].astype(str).map(clean_text)
    df = df[df[TARGET_COLUMN].isin(LABELS)].reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid rows found after preprocessing")

    return df


def main() -> None:
    print("Running preprocessing...")
    print(f"Input file: {INPUT_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_and_prepare_data(INPUT_PATH)
    print(f"Loaded rows: {len(df)}")

    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved processed data to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()