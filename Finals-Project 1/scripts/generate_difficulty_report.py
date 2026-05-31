from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    outputs = root / "outputs"
    prediction_files = sorted(outputs.glob("*_predictions.csv"))
    prediction_files = [p for p in prediction_files if p.name != "student_results.csv"]

    rows = []
    for fp in prediction_files:
        df = pd.read_csv(fp)
        required = {"question_id", "predicted_answer", "ground_truth"}
        if not required.issubset(df.columns):
            continue
        df["is_correct"] = df["predicted_answer"] == df["ground_truth"]
        rows.append(df[["question_id", "is_correct"]])

    if not rows:
        raise ValueError("No prediction CSV files found in outputs/ to build difficulty report.")

    all_df = pd.concat(rows, ignore_index=True)
    grp = all_df.groupby("question_id", as_index=False)["is_correct"].mean()
    grp["accuracy_pct"] = (grp["is_correct"] * 100.0).round(2)
    grp["difficulty_index"] = (1.0 - grp["is_correct"]).round(4)
    grp = grp.drop(columns=["is_correct"]).sort_values("difficulty_index", ascending=False)

    out = outputs / "question_difficulty_report.csv"
    grp.to_csv(out, index=False)
    print(f"Saved: {out}")
    print("Top difficult questions:")
    print(grp.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
