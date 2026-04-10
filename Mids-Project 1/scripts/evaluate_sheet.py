from pathlib import Path
import argparse
import json
from datetime import datetime
import sys

import cv2
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from omr.calibration import get_calibrated_template_points, load_calibration
from omr.cnn_model import load_cnn_model
from omr.config import ANSWER_KEY_PATH, MODEL_DIR, OMRConfig, OUTPUT_DIR
from omr.data_utils import get_answers_for_sheet, load_answer_key
from omr.omr_engine import evaluate_sheet, predictions_to_dataframe
from omr.sheet_template import BubblePoint


def _append_student_result(
    output_path: Path,
    row: dict,
) -> None:
    if output_path.exists():
        df = pd.read_csv(output_path)
        # Replace row for same sheet if it exists to keep latest run.
        df = df[df["sheet_id"] != row["sheet_id"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(output_path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate one OMR sheet image.")
    parser.add_argument("--image", required=True, type=str, help="Path to OMR image")
    parser.add_argument("--sheet-id", required=False, type=int, help="Image_ID in answer_key.csv")
    parser.add_argument("--use-cnn", action="store_true", help="Use transfer-learning CNN enhancement if model exists")
    parser.add_argument(
        "--calibration",
        default="models/template_calibration.json",
        help="Path to optional perspective calibration JSON (if available)",
    )
    parser.add_argument(
        "--template-points",
        default="models/template_points.json",
        help="Path to optional exact template points JSON (if available)",
    )
    parser.add_argument("--marks-correct", type=float, default=1.0, help="Marks for each correct answer")
    parser.add_argument("--marks-wrong", type=float, default=-0.25, help="Marks for each wrong answer")
    parser.add_argument("--marks-unattempted", type=float, default=0.0, help="Marks for each unattempted answer")
    args = parser.parse_args()

    image_path = Path(args.image)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = OMRConfig()

    cnn_model = load_cnn_model(MODEL_DIR / "bubble_cnn_transfer.keras") if args.use_cnn else None
    if args.use_cnn and cnn_model is None:
        raise ValueError(
            "CNN model not found. Expected models/bubble_cnn_transfer.keras (add trained weights to models/)."
        )
    sheet_id = args.sheet_id
    if sheet_id is None:
        try:
            sheet_id = int(image_path.stem)
            print(f"Inferred sheet-id from filename: {sheet_id}")
        except ValueError as exc:
            raise ValueError(
                "Could not infer sheet-id from filename. Provide --sheet-id explicitly."
            ) from exc

    answer_df = load_answer_key(ANSWER_KEY_PATH)
    gt = get_answers_for_sheet(answer_df, sheet_id)
    calibrated_points = None
    template_points = None
    template_path = Path(args.template_points)
    if template_path.exists():
        payload = json.loads(template_path.read_text(encoding="utf-8"))
        template_points = [
            BubblePoint(
                question_id=int(p["question_id"]),
                option=str(p["option"]),
                x=int(p["x"]),
                y=int(p["y"]),
                radius=int(p.get("radius", 14)),
            )
            for p in payload.get("points", [])
        ]
        if len(template_points) == 120:
            calibrated_points = template_points
            print(f"Using exact template points: {template_path}")
        else:
            print("Template points file found but invalid; expected 120 points. Ignoring.")

    calibration = load_calibration(Path(args.calibration))
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    if calibrated_points is None and calibration is not None:
        calibrated_points = get_calibrated_template_points(img.shape[:2], config, calibration)
        print(f"Using calibration: {Path(args.calibration)}")
    elif calibrated_points is None:
        print("No calibration file found. Using auto-detection.")

    preds, summary, overlay = evaluate_sheet(
        image_path=image_path,
        config=config,
        answer_key=gt,
        cnn_model=cnn_model,
        calibrated_points=calibrated_points,
    )
    pred_df = predictions_to_dataframe(preds, gt)
    final_marks = (
        summary["correct"] * args.marks_correct
        + summary["wrong"] * args.marks_wrong
        + summary["unattempted"] * args.marks_unattempted
    )
    summary["final_marks"] = round(final_marks, 2)

    out_csv = OUTPUT_DIR / f"{image_path.stem}_predictions.csv"
    out_img = OUTPUT_DIR / f"{image_path.stem}_overlay.png"
    out_audit = OUTPUT_DIR / f"{image_path.stem}_audit.json"
    pred_df.to_csv(out_csv, index=False)
    cv2.imwrite(str(out_img), overlay)

    # Store per-student summary in a centralized CSV for quick record keeping.
    central_results_csv = OUTPUT_DIR / "student_results.csv"
    model_name = "cnn_transfer"
    _append_student_result(
        central_results_csv,
        {
            "sheet_id": sheet_id,
            "image_file": image_path.name,
            "model": model_name,
            "accuracy": summary["accuracy"],
            "correct": summary["correct"],
            "wrong": summary["wrong"],
            "unattempted": summary["unattempted"],
            "avg_confidence": summary["avg_confidence"],
            "section_1_accuracy": summary.get("section_1_accuracy", 0.0),
            "section_2_accuracy": summary.get("section_2_accuracy", 0.0),
            "section_3_accuracy": summary.get("section_3_accuracy", 0.0),
            "final_marks": summary["final_marks"],
            "template_drift_px": summary.get("template_drift_px", 0.0),
            "template_drift_warning": summary.get("template_drift_warning", False),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
    )

    # Audit report for reproducibility and transparent grading.
    review_rows = pred_df[pred_df["review_required"]]
    audit = {
        "sheet_id": sheet_id,
        "image_file": image_path.name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "summary": summary,
        "marking_scheme": {
            "correct": args.marks_correct,
            "wrong": args.marks_wrong,
            "unattempted": args.marks_unattempted,
        },
        "review_queue_count": int(len(review_rows)),
        "review_queue_questions": review_rows["question_id"].astype(int).tolist(),
    }
    out_audit.write_text(json.dumps(audit, indent=2), encoding="utf-8")

    print("Evaluation complete.")
    print(f"Predictions CSV: {out_csv}")
    print(f"Overlay image   : {out_img}")
    print(f"Audit report    : {out_audit}")
    print(f"Student records : {central_results_csv}")
    print("Score summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
