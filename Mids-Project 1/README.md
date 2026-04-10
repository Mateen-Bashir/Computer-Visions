# AI-Based OMR Sheet Evaluation (CPU Friendly)

A complete, presentation-ready Computer Vision project for OMR sheet evaluation that works on a normal laptop (no GPU required).

## Why this project is unique

- Hybrid approach: robust OMR geometry + transfer-learning CNN bubble classifier.
- Confidence-aware scoring: flags low-confidence answers instead of blindly assigning marks.
- Ambiguity detection: automatically detects `MULTI`, `BLANK`, and uncertain bubbles.
- CPU-optimized: OpenCV + TensorFlow CNN inference.

## Project structure

- `src/omr/` core pipeline modules
- `scripts/` runnable demo and training scripts
- `models/` trained ML artifacts (created after training)
- `outputs/` prediction CSV and visualization images
- `bubble_dataset/` your bubble-label dataset CSVs
- `csv/answer_key.csv` official answer key

## 1) Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Frontend App (Streamlit)

Run the web app:

```bash
streamlit run app.py
```

Features in UI:
- upload image and evaluate
- CNN toggle
- negative marking controls
- section analytics
- review queue table
- downloadable CSV/overlay/audit
- batch evaluate a folder of numeric image files
- one-click full report ZIP export from `outputs/`

Transfer-learning CNN runs on CPU with TensorFlow (`tf-nightly` for Python 3.13).

## 2) Train transfer-learning CNN (CPU friendly)

This trains a MobileNetV2-based bubble-state classifier with a frozen backbone using:
- real images from `bubble_dataset/train/images`, `bubble_dataset/valid/images`, `bubble_dataset/test/images`
- labels from each split's `labels.csv`
- optional synthetic augmentation for better robustness

```bash
python scripts/train_cnn_transfer.py --epochs 8 --per-class 600
```

This creates:
- `models/bubble_cnn_transfer.keras`

## 3) Generate sample OMR sheets for demo

Creates synthetic answer sheets so you can present end-to-end even without scanner images.

```bash
python scripts/generate_demo_sheets.py
```

Generated files go to `demo_inputs/`.

## 4) One-time calibration for real sheets (recommended)

Use one real sheet image and click 4 page corners:

```bash
python scripts/calibrate_template.py --image images/1.jpg
```

This creates:
- `models/template_calibration.json`

Optional (more accurate): create exact 120 bubble template points from one reference sheet:

```bash
python scripts/create_template_points.py --image images/1.jpg
```

This creates:
- `models/template_points.json`

## 5) Evaluate OMR sheet

```bash
python scripts/evaluate_sheet.py --image demo_inputs/sheet_1.png --sheet-id 1 --use-cnn
python scripts/evaluate_sheet.py --image images/1.jpg --use-cnn
```

Outputs:
- `outputs/sheet_1_predictions.csv`
- `outputs/sheet_1_overlay.png`
- `outputs/sheet_1_audit.json`
- `outputs/student_results.csv` (centralized student-wise records)
- terminal score summary

For real images whose filename is numeric (e.g., `images/1.jpg`), sheet-id is auto inferred from filename.

Negative marking is supported:

```bash
python scripts/evaluate_sheet.py --image images/1.jpg --use-cnn --marks-correct 1 --marks-wrong -0.25 --marks-unattempted 0
```

## 6) Batch evaluate all demo sheets

```bash
python scripts/run_demo_batch.py --use-cnn
```

Creates `outputs/demo_batch_results.csv`.

Generate question-wise difficulty analytics from stored prediction CSV files:

```bash
python scripts/generate_difficulty_report.py
```

This creates:
- `outputs/question_difficulty_report.csv`

## How scoring works

For each question:
- If one bubble has strong confidence -> select `A/B/C/D`.
- If multiple bubbles are near top confidence -> label `MULTI`.
- If all bubbles are weak -> label `BLANK`.
- Compare with answer key and compute:
  - Correct, Wrong, Unattempted
  - Accuracy
  - Confidence statistics

## Presentation-ready talking points

1. Built a full OMR pipeline under CPU constraints.
2. Added uncertainty estimation to avoid false certainty.
3. Added robust handling of ambiguous marks (crossed/invalid/multi-fill).
4. Designed modular architecture for easy future scaling (deep models, mobile capture, multi-template support).

## Notes

- Your current `answer_key.csv` contains some `MULTI`/`BLANK` values for certain rows. The pipeline supports this.
- If you later add real sheet photos, this system can evaluate them by updating the sheet template mapping.
