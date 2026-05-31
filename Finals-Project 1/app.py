from __future__ import annotations

import json
import io
from datetime import datetime
from pathlib import Path
import sys
import zipfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from omr.calibration import get_calibrated_template_points, load_calibration
from omr.cnn_model import load_cnn_model
from omr.config import ANSWER_KEY_PATH, MODEL_DIR, OMRConfig, OUTPUT_DIR
from omr.data_utils import get_answers_for_sheet, load_answer_key
from omr.omr_engine import evaluate_sheet, predictions_to_dataframe
from omr.sheet_template import BubblePoint


@st.cache_resource
def _cached_cnn_model():
    return load_cnn_model(MODEL_DIR / "bubble_cnn_transfer.keras")


@st.cache_data
def _cached_answer_key():
    return load_answer_key(ANSWER_KEY_PATH)


def _inject_professional_style() -> None:
    st.markdown(
        """
        <style>
            .block-container {padding-top: 1.2rem; padding-bottom: 1.0rem;}
            .omr-hero {
                background: linear-gradient(90deg, #0f172a 0%, #1e293b 60%, #334155 100%);
                padding: 18px 22px;
                border-radius: 14px;
                color: #f8fafc;
                margin-bottom: 1.2rem;
                border: 1px solid rgba(148,163,184,0.25);
            }
            .omr-title {font-size: 1.5rem; font-weight: 700; letter-spacing: -0.02em;}
            .omr-sub {color: #cbd5e1; margin-top: 6px; font-size: 0.92rem;}
            .metric-card {
                border: 1px solid rgba(148,163,184,0.25);
                border-radius: 12px;
                padding: 12px;
                background: rgba(15,23,42,0.02);
            }
            .section-title {
                font-size: 1.05rem;
                font-weight: 600;
                margin-top: 0.4rem;
                margin-bottom: 0.6rem;
            }
            .footer-note {
                margin-top: 1.2rem;
                color: #64748b;
                font-size: 0.82rem;
                border-top: 1px solid rgba(148,163,184,0.25);
                padding-top: 0.6rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _load_template_points(path: Path) -> list[BubblePoint] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    points = [
        BubblePoint(
            question_id=int(p["question_id"]),
            option=str(p["option"]),
            x=int(p["x"]),
            y=int(p["y"]),
            radius=int(p.get("radius", 14)),
        )
        for p in payload.get("points", [])
    ]
    return points if len(points) == 120 else None


def _resolve_calibrated_points(
    img: np.ndarray,
    config: OMRConfig,
) -> tuple[list[BubblePoint] | None, str]:
    template_points = _load_template_points(ROOT / "models" / "template_points.json")
    if template_points is not None:
        return template_points, "exact template points"

    calibration = load_calibration(ROOT / "models" / "template_calibration.json")
    if calibration is not None:
        return get_calibrated_template_points(img.shape[:2], config, calibration), "calibration transform"

    return None, "auto-detection"


def _select_cnn_model(use_cnn: bool, cnn_model) -> tuple[object | None, str]:
    if use_cnn:
        if cnn_model is None:
            return None, "opencv_only"
        return cnn_model, "cnn_transfer"
    return None, "opencv_only"


def _append_student_result(csv_path: Path, row: dict) -> None:
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df = df[df["sheet_id"] != row["sheet_id"]]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])
    df.to_csv(csv_path, index=False)


def _to_downloadable_image_bytes(image_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", image_bgr)
    if not ok:
        return b""
    return buf.tobytes()


def _build_outputs_zip_bytes(output_dir: Path) -> bytes:
    memory = io.BytesIO()
    with zipfile.ZipFile(memory, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(output_dir.glob("*")):
            if fp.is_file():
                zf.write(fp, arcname=fp.name)
    return memory.getvalue()


def _init_session_state() -> None:
    if "single_result" not in st.session_state:
        st.session_state.single_result = None
    if "batch_result" not in st.session_state:
        st.session_state.batch_result = None


def _render_single_results(result: dict) -> None:
    summary = result["summary"]
    pred_df = result["pred_df"]
    review_df = result["review_df"]
    img = result["img"]
    overlay = result["overlay"]
    template_source = result["template_source"]
    model_name = result["model_name"]

    st.success("Evaluation complete.")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Accuracy %", summary["accuracy"])
        st.markdown("</div>", unsafe_allow_html=True)
    with c2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Final Marks", summary["final_marks"])
        st.markdown("</div>", unsafe_allow_html=True)
    with c3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Correct", summary["correct"])
        st.markdown("</div>", unsafe_allow_html=True)
    with c4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Review Queue", int(len(review_df)))
        st.markdown("</div>", unsafe_allow_html=True)

    st.caption(f"Model: **{model_name}**")
    st.write(
        f"Template source: **{template_source}** | Drift: **{summary.get('template_drift_px', 0.0)} px** | "
        f"Drift warning: **{summary.get('template_drift_warning', False)}**"
    )

    tab_overview, tab_review, tab_data = st.tabs(["Overview", "Review Queue", "Detailed Data"])
    with tab_overview:
        left, right = st.columns([1, 1])
        with left:
            st.subheader("Uploaded Sheet")
            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), width="stretch")
        with right:
            st.subheader("Evaluation Overlay")
            st.image(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB), width="stretch")

        st.subheader("Section Analytics")
        sec_df = pd.DataFrame(
            {
                "Section": ["Q1-10", "Q11-20", "Q21-30"],
                "Accuracy %": [
                    summary.get("section_1_accuracy", 0.0),
                    summary.get("section_2_accuracy", 0.0),
                    summary.get("section_3_accuracy", 0.0),
                ],
            }
        )
        st.bar_chart(sec_df.set_index("Section"))
        st.dataframe(sec_df, width="stretch", hide_index=True)

    with tab_review:
        st.subheader("Manual Review Queue")
        if review_df.empty:
            st.info("No questions flagged for manual review.")
        else:
            st.dataframe(
                review_df[["question_id", "predicted_answer", "ground_truth", "confidence", "reason"]],
                width="stretch",
                hide_index=True,
            )

    with tab_data:
        st.subheader("Per-question Predictions")
        st.dataframe(pred_df, width="stretch", hide_index=True)

    st.download_button(
        "Download Predictions CSV",
        data=result["csv_bytes"],
        file_name=result["csv_name"],
        mime="text/csv",
        key="download_single_csv",
    )
    st.download_button(
        "Download Overlay PNG",
        data=_to_downloadable_image_bytes(overlay),
        file_name=result["overlay_name"],
        mime="image/png",
        key="download_single_overlay",
    )
    st.download_button(
        "Download Audit JSON",
        data=result["audit_bytes"],
        file_name=result["audit_name"],
        mime="application/json",
        key="download_single_audit",
    )


def _run_single_evaluation(
    uploaded,
    manual_sheet_id: str,
    answer_df: pd.DataFrame,
    config: OMRConfig,
    selected_cnn,
    model_name: str,
    marks_correct: float,
    marks_wrong: float,
    marks_unattempted: float,
) -> dict | None:
    image_name = uploaded.name
    image_bytes = uploaded.getvalue()
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        st.error("Could not decode uploaded image.")
        return None

    if manual_sheet_id.strip():
        try:
            sheet_id = int(manual_sheet_id.strip())
        except ValueError:
            st.error("Sheet ID must be an integer.")
            return None
    else:
        stem = Path(image_name).stem
        try:
            sheet_id = int(stem)
        except ValueError:
            st.error("Could not infer sheet-id from filename. Please provide Sheet ID.")
            return None

    try:
        answer_key = get_answers_for_sheet(answer_df, sheet_id)
    except Exception as exc:
        st.error(str(exc))
        return None

    calibrated_points, template_source = _resolve_calibrated_points(img, config)
    temp_image_path = OUTPUT_DIR / f"_uploaded_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    cv2.imwrite(str(temp_image_path), img)

    with st.spinner("Evaluating sheet… this may take 15–30 seconds on CPU."):
        preds, summary, overlay = evaluate_sheet(
            image_path=temp_image_path,
            config=config,
            answer_key=answer_key,
            cnn_model=selected_cnn,
            calibrated_points=calibrated_points,
        )

    pred_df = predictions_to_dataframe(preds, answer_key)
    final_marks = (
        summary["correct"] * marks_correct
        + summary["wrong"] * marks_wrong
        + summary["unattempted"] * marks_unattempted
    )
    summary["final_marks"] = round(final_marks, 2)

    out_prefix = Path(image_name).stem
    out_csv = OUTPUT_DIR / f"{out_prefix}_predictions.csv"
    out_overlay = OUTPUT_DIR / f"{out_prefix}_overlay.png"
    out_audit = OUTPUT_DIR / f"{out_prefix}_audit.json"
    out_records = OUTPUT_DIR / "student_results.csv"

    pred_df.to_csv(out_csv, index=False)
    cv2.imwrite(str(out_overlay), overlay)

    _append_student_result(
        out_records,
        {
            "sheet_id": sheet_id,
            "image_file": image_name,
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

    review_df = pred_df[pred_df["review_required"]]
    audit = {
        "sheet_id": sheet_id,
        "image_file": image_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "template_source": template_source,
        "summary": summary,
        "marking_scheme": {
            "correct": marks_correct,
            "wrong": marks_wrong,
            "unattempted": marks_unattempted,
        },
        "review_queue_count": int(len(review_df)),
        "review_queue_questions": review_df["question_id"].astype(int).tolist(),
    }
    audit_text = json.dumps(audit, indent=2)
    out_audit.write_text(audit_text, encoding="utf-8")

    return {
        "summary": summary,
        "pred_df": pred_df,
        "review_df": review_df,
        "img": img,
        "overlay": overlay,
        "template_source": template_source,
        "model_name": model_name,
        "csv_bytes": out_csv.read_bytes(),
        "csv_name": out_csv.name,
        "overlay_name": out_overlay.name,
        "audit_bytes": audit_text.encode("utf-8"),
        "audit_name": out_audit.name,
    }


def _list_numeric_images(folder: Path) -> list[Path]:
    if not folder.exists() or not folder.is_dir():
        return []
    files: list[Path] = []
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        files.extend(folder.glob(ext))
    numeric_files = []
    for fp in files:
        try:
            _ = int(fp.stem)
            numeric_files.append(fp)
        except ValueError:
            continue
    return sorted(numeric_files, key=lambda p: int(p.stem))


def _save_uploaded_batch_files(uploads) -> list[tuple[Path, str]]:
    items: list[tuple[Path, str]] = []
    for uploaded in uploads:
        temp_path = OUTPUT_DIR / f"_batch_{datetime.now().strftime('%H%M%S%f')}_{uploaded.name}"
        temp_path.write_bytes(uploaded.getvalue())
        items.append((temp_path, uploaded.name))
    return items


def _process_batch_file(
    fp: Path,
    answer_df: pd.DataFrame,
    config: OMRConfig,
    selected_cnn,
    model_name: str,
    marks_correct: float,
    marks_wrong: float,
    marks_unattempted: float,
    template_points: list[BubblePoint] | None,
    calibration,
    source_name: str | None = None,
) -> dict | None:
    label_name = source_name or fp.name
    try:
        sheet_id = int(Path(label_name).stem)
    except ValueError:
        st.warning(f"Skipped `{label_name}` — filename must be numeric (e.g. `12.jpg`) to infer sheet ID.")
        return None

    try:
        answer_key = get_answers_for_sheet(answer_df, sheet_id)
    except Exception as exc:
        st.warning(f"Skipped `{label_name}` — {exc}")
        return None

    img = cv2.imread(str(fp))
    if img is None:
        st.warning(f"Skipped `{label_name}` — could not read image.")
        return None

    calibrated_points = None
    if template_points is not None:
        calibrated_points = template_points
    elif calibration is not None:
        calibrated_points = get_calibrated_template_points(img.shape[:2], config, calibration)

    preds, summary, overlay = evaluate_sheet(
        image_path=fp,
        config=config,
        answer_key=answer_key,
        cnn_model=selected_cnn,
        calibrated_points=calibrated_points,
    )

    pred_df = predictions_to_dataframe(preds, answer_key)
    final_marks = (
        summary["correct"] * marks_correct
        + summary["wrong"] * marks_wrong
        + summary["unattempted"] * marks_unattempted
    )
    summary["final_marks"] = round(final_marks, 2)

    out_prefix = Path(label_name).stem
    out_csv = OUTPUT_DIR / f"{out_prefix}_predictions.csv"
    out_overlay = OUTPUT_DIR / f"{out_prefix}_overlay.png"
    out_audit = OUTPUT_DIR / f"{out_prefix}_audit.json"
    out_records = OUTPUT_DIR / "student_results.csv"
    pred_df.to_csv(out_csv, index=False)
    cv2.imwrite(str(out_overlay), overlay)

    _append_student_result(
        out_records,
        {
            "sheet_id": sheet_id,
            "image_file": label_name,
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
    out_audit.write_text(
        json.dumps(
            {
                "sheet_id": sheet_id,
                "image_file": label_name,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "model": model_name,
                "summary": summary,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return {
        "sheet_id": sheet_id,
        "image_file": label_name,
        "accuracy": summary["accuracy"],
        "final_marks": summary["final_marks"],
        "correct": summary["correct"],
        "wrong": summary["wrong"],
        "section_1_accuracy": summary.get("section_1_accuracy", 0.0),
        "section_2_accuracy": summary.get("section_2_accuracy", 0.0),
        "section_3_accuracy": summary.get("section_3_accuracy", 0.0),
    }


def _run_batch_evaluation(
    batch_items: list[tuple[Path, str]],
    answer_df: pd.DataFrame,
    config: OMRConfig,
    selected_cnn,
    model_name: str,
    marks_correct: float,
    marks_wrong: float,
    marks_unattempted: float,
) -> dict | None:
    if not batch_items:
        st.warning("No images selected for batch evaluation.")
        return None

    template_points = _load_template_points(ROOT / "models" / "template_points.json")
    calibration = load_calibration(ROOT / "models" / "template_calibration.json")

    rows = []
    progress = st.progress(0, text="Starting batch evaluation...")
    for idx, (fp, source_name) in enumerate(batch_items, start=1):
        with st.spinner(f"Evaluating {source_name} ({idx}/{len(batch_items)})…"):
            row = _process_batch_file(
                fp,
                answer_df,
                config,
                selected_cnn,
                model_name,
                marks_correct,
                marks_wrong,
                marks_unattempted,
                template_points,
                calibration,
                source_name=source_name,
            )
        if row is not None:
            rows.append(row)
        progress.progress(int(idx * 100 / len(batch_items)), text=f"Processed {idx}/{len(batch_items)}")

    if not rows:
        st.warning("No files were processed.")
        return None

    result_df = pd.DataFrame(rows).sort_values("sheet_id")
    batch_out = OUTPUT_DIR / "batch_summary.csv"
    result_df.to_csv(batch_out, index=False)

    return {
        "result_df": result_df,
        "batch_csv_bytes": batch_out.read_bytes(),
        "batch_csv_name": batch_out.name,
        "model_name": model_name,
    }


def _render_batch_results(result: dict) -> None:
    result_df = result["result_df"]
    st.success(f"Batch evaluation completed for {len(result_df)} files.")
    st.caption(f"Model: **{result['model_name']}**")
    st.dataframe(result_df, width="stretch")
    b1, b2 = st.columns(2)
    b1.metric("Average Accuracy %", f"{result_df['accuracy'].mean():.2f}")
    b2.metric("Average Marks", f"{result_df['final_marks'].mean():.2f}")
    st.download_button(
        "Download Batch Summary CSV",
        data=result["batch_csv_bytes"],
        file_name=result["batch_csv_name"],
        mime="text/csv",
        key="download_batch_csv",
    )


def main() -> None:
    st.set_page_config(page_title="AI OMR Evaluator", page_icon="📋", layout="wide")
    _inject_professional_style()
    _init_session_state()

    st.markdown(
        """
        <div class="omr-hero">
          <div class="omr-title">AI OMR Evaluator</div>
          <div class="omr-sub">Automated sheet grading &bull; CNN + OpenCV &bull; CPU-friendly</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config = OMRConfig()
    answer_df = _cached_answer_key()
    cnn_model = _cached_cnn_model()
    template_points_file = ROOT / "models" / "template_points.json"
    template_calib_file = ROOT / "models" / "template_calibration.json"

    with st.sidebar:
        st.header("Control Panel")
        with st.expander("System Status", expanded=True):
            st.write(f"**CNN model:** {'Loaded' if cnn_model is not None else 'Not loaded'}")
            st.write(f"**Template points:** {'Available' if template_points_file.exists() else 'Not found'}")
            st.write(f"**Calibration file:** {'Available' if template_calib_file.exists() else 'Not found'}")
        use_cnn = st.checkbox("Use CNN model", value=True, help="When off, grading uses OpenCV fill-ratio only.")
        if use_cnn and cnn_model is None:
            st.warning("CNN model not found. Falling back to OpenCV-only scoring.")
        marks_correct = st.number_input("Marks for correct", value=1.0, step=0.25, format="%.2f")
        marks_wrong = st.number_input("Marks for wrong", value=-0.25, step=0.25, format="%.2f")
        marks_unattempted = st.number_input("Marks for unattempted", value=0.0, step=0.25, format="%.2f")
        st.divider()
        st.markdown("**Template source priority**")
        st.caption("1) `models/template_points.json`  2) `models/template_calibration.json`  3) auto-detection")
        st.divider()
        st.markdown("**Export Reports**")
        if any(OUTPUT_DIR.glob("*")):
            zip_bytes = _build_outputs_zip_bytes(OUTPUT_DIR)
            st.download_button(
                "Download Full Report ZIP",
                data=zip_bytes,
                file_name=f"omr_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip",
            )
        else:
            st.caption("No outputs yet. Run an evaluation first.")

    selected_cnn, model_name = _select_cnn_model(use_cnn, cnn_model)
    if use_cnn and cnn_model is None:
        model_name = "opencv_only"

    single_tab, batch_tab = st.tabs(["Single Evaluation", "Batch Evaluation"])

    with single_tab:
        st.markdown('<div class="section-title">Evaluate One Sheet</div>', unsafe_allow_html=True)
        left_in, right_in = st.columns([2, 1])
        with left_in:
            uploaded = st.file_uploader("Upload OMR sheet image (.jpg/.png)", type=["jpg", "jpeg", "png"])
        with right_in:
            manual_sheet_id = st.text_input("Sheet ID (optional)", placeholder="Auto from filename")

        btn_col1, btn_col2 = st.columns([1, 4])
        with btn_col1:
            run_single = st.button("Evaluate Sheet", type="primary", disabled=uploaded is None)
        with btn_col2:
            if st.session_state.single_result is not None:
                if st.button("Clear Results"):
                    st.session_state.single_result = None
                    st.rerun()

        if run_single and uploaded is not None:
            result = _run_single_evaluation(
                uploaded,
                manual_sheet_id,
                answer_df,
                config,
                selected_cnn,
                model_name,
                marks_correct,
                marks_wrong,
                marks_unattempted,
            )
            if result is not None:
                st.session_state.single_result = result

        if st.session_state.single_result is not None:
            _render_single_results(st.session_state.single_result)

    with batch_tab:
        st.markdown('<div class="section-title">Batch Run & Summary</div>', unsafe_allow_html=True)
        batch_mode = st.radio(
            "Image source",
            ["Select from folder", "Upload images"],
            horizontal=True,
            help="Pick specific sheets from a folder, or upload your own image files.",
        )

        batch_items: list[tuple[Path, str]] = []

        if batch_mode == "Select from folder":
            batch_folder = st.text_input("Folder path", value=str((ROOT / "images").resolve()))
            folder = Path(batch_folder)
            available_images = _list_numeric_images(folder)

            if not folder.exists() or not folder.is_dir():
                st.error(f"Invalid folder: {folder}")
            elif not available_images:
                st.info("No numeric-named images found in this folder (e.g. `1.jpg`, `42.png`).")
            else:
                image_options = [fp.name for fp in available_images]
                pick_col1, pick_col2 = st.columns([3, 1])
                with pick_col2:
                    if st.button("Select all", use_container_width=True):
                        st.session_state.batch_image_select = image_options
                selected_names = st.multiselect(
                    "Select sheet images",
                    options=image_options,
                    key="batch_image_select",
                    help="Sheet ID is inferred from the filename (e.g. `15.jpg` → sheet 15).",
                )
                batch_items = [(folder / name, name) for name in selected_names]
                if selected_names:
                    st.caption(f"{len(selected_names)} image(s) selected.")
        else:
            uploaded_batch = st.file_uploader(
                "Upload OMR sheet images",
                type=["jpg", "jpeg", "png"],
                accept_multiple_files=True,
                help="Use numeric filenames so sheet IDs can be matched to the answer key.",
            )
            if uploaded_batch:
                batch_items = _save_uploaded_batch_files(uploaded_batch)
                st.caption(f"{len(uploaded_batch)} file(s) ready: {', '.join(f.name for f in uploaded_batch)}")

        btn_col1, btn_col2 = st.columns([1, 4])
        with btn_col1:
            run_batch = st.button(
                "Run Batch",
                type="primary",
                disabled=len(batch_items) == 0,
            )
        with btn_col2:
            if st.session_state.batch_result is not None:
                if st.button("Clear Batch Results"):
                    st.session_state.batch_result = None
                    st.rerun()

        if run_batch:
            batch_result = _run_batch_evaluation(
                batch_items,
                answer_df,
                config,
                selected_cnn,
                model_name,
                marks_correct,
                marks_wrong,
                marks_unattempted,
            )
            if batch_result is not None:
                st.session_state.batch_result = batch_result

        if st.session_state.batch_result is not None:
            _render_batch_results(st.session_state.batch_result)

    st.markdown(
        "<div class='footer-note'>AI OMR Tool • Built for CPU-only execution • Outputs saved in <code>outputs/</code></div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
