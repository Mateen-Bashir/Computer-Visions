from pathlib import Path
import sys

import pandas as pd
import argparse

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from omr.cnn_model import load_cnn_model
from omr.config import ANSWER_KEY_PATH, MODEL_DIR, OMRConfig, OUTPUT_DIR
from omr.data_utils import get_answers_for_sheet, load_answer_key
from omr.omr_engine import evaluate_sheet


def main() -> None:
    parser = argparse.ArgumentParser(description="Run batch evaluation on demo sheets.")
    _ = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    demo_dir = root / "demo_inputs"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    answer_df = load_answer_key(ANSWER_KEY_PATH)
    cnn_model = load_cnn_model(MODEL_DIR / "bubble_cnn_transfer.keras")
    if cnn_model is None:
        raise ValueError("CNN model not found. Train it first using scripts/train_cnn_transfer.py")

    rows = []
    for idx, img_path in enumerate(sorted(demo_dir.glob("sheet_*.png")), start=1):
        gt = get_answers_for_sheet(answer_df, idx)
        _, summary, _ = evaluate_sheet(img_path, OMRConfig(), gt, cnn_model=cnn_model)
        summary["file"] = img_path.name
        summary["sheet_id"] = idx
        rows.append(summary)

    if not rows:
        raise ValueError("No demo sheets found. Run scripts/generate_demo_sheets.py first.")

    df = pd.DataFrame(rows)
    out = OUTPUT_DIR / "demo_batch_results.csv"
    df.to_csv(out, index=False)
    print(f"Saved batch results: {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
