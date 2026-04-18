import argparse
from pathlib import Path

from src.config import ensure_dirs
from src.data_preprocessing import load_and_prepare_data
from src.train_baseline import train_baseline_model
from src.train_bert import train_bert_model
from src.evaluate import compare_models
from src.inference import predict_single_text, batch_predict
from src.dashboard_export import export_dashboard_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Customer feedback NLP pipeline")
    parser.add_argument("--mode", required=True, choices=[
        "train_all", "train_baseline", "train_bert", "predict", "batch_predict", "export_dashboard"
    ])
    parser.add_argument("--input", help="Path to CSV input file")
    parser.add_argument("--text", help="Single text for inference")
    args = parser.parse_args()

    ensure_dirs()

    if args.mode in {"train_all", "train_baseline", "train_bert", "batch_predict", "export_dashboard"} and not args.input:
        raise ValueError("--input is required for this mode")

    if args.mode == "train_all":
        df = load_and_prepare_data(args.input)
        train_baseline_model(df)
        train_bert_model(df)
        compare_models(df)
    elif args.mode == "train_baseline":
        df = load_and_prepare_data(args.input)
        train_baseline_model(df)
    elif args.mode == "train_bert":
        df = load_and_prepare_data(args.input)
        train_bert_model(df)
    elif args.mode == "predict":
        if not args.text:
            raise ValueError("--text is required for predict mode")
        print(predict_single_text(args.text))
    elif args.mode == "batch_predict":
        batch_predict(Path(args.input))
    elif args.mode == "export_dashboard":
        export_dashboard_summary(Path(args.input))


if __name__ == "__main__":
    main()
