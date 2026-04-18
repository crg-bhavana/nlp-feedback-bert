source venv/bin/activatefrom pathlib import Path

import pandas as pd

from src.config import OUTPUT_DIR, TEXT_COLUMN
from src.inference import batch_predict


def export_dashboard_summary(input_path: Path) -> None:
    output = batch_predict(input_path)
    summary = (
        output.groupby("predicted_label")
        .size()
        .reset_index(name="record_count")
        .sort_values("record_count", ascending=False)
    )
    summary["share_pct"] = (summary["record_count"] / summary["record_count"].sum() * 100).round(2)

    if TEXT_COLUMN in output.columns:
        output["text_length"] = output[TEXT_COLUMN].astype(str).str.len()
        length_summary = output.groupby("predicted_label", as_index=False)["text_length"].mean()
        summary = summary.merge(length_summary, on="predicted_label", how="left")

    summary_path = OUTPUT_DIR / "dashboard_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved dashboard summary to {summary_path}")
