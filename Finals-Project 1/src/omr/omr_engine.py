from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .cnn_model import BubbleCNNModel
from .config import OMRConfig
from .sheet_template import BubblePoint, build_template_points


@dataclass
class QuestionPrediction:
    question_id: int
    answer: str
    confidence: float
    reason: str
    option_scores: dict[str, float]


def _kmeans_1d(values: np.ndarray, k: int, max_iter: int = 30) -> np.ndarray:
    values = values.astype(np.float32)
    if len(values) < k:
        raise ValueError("Not enough values for kmeans")
    centers = np.linspace(values.min(), values.max(), k, dtype=np.float32)
    for _ in range(max_iter):
        d = np.abs(values[:, None] - centers[None, :])
        labels = d.argmin(axis=1)
        new_centers = np.array(
            [values[labels == i].mean() if np.any(labels == i) else centers[i] for i in range(k)],
            dtype=np.float32,
        )
        if np.allclose(new_centers, centers, atol=0.5):
            centers = new_centers
            break
        centers = new_centers
    return np.sort(centers)


def _cluster_rows(circles: np.ndarray, y_tol: float) -> list[list[np.ndarray]]:
    rows: list[list[np.ndarray]] = []
    for c in sorted(circles, key=lambda z: z[1]):
        if not rows:
            rows.append([c])
            continue
        last_mean_y = float(np.mean([x[1] for x in rows[-1]]))
        if abs(c[1] - last_mean_y) <= y_tol:
            rows[-1].append(c)
        else:
            rows.append([c])
    return rows


def _detect_raw_circles(gray: np.ndarray) -> np.ndarray | None:
    blur = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=12,
        param1=120,
        param2=16,
        minRadius=5,
        maxRadius=22,
    )
    if circles is None:
        return None
    raw = np.round(circles[0]).astype(int)
    if len(raw) == 0:
        return None
    r = raw[:, 2]
    r_med = float(np.median(r))
    raw = raw[(r >= 0.65 * r_med) & (r <= 1.6 * r_med)]
    return raw if len(raw) else None


def _template_drift_pixels(gray: np.ndarray, points: list[BubblePoint]) -> float:
    raw = _detect_raw_circles(gray)
    if raw is None or len(points) == 0:
        return 999.0
    dists = []
    for p in points:
        d = np.sqrt((raw[:, 0] - p.x) ** 2 + (raw[:, 1] - p.y) ** 2)
        dists.append(float(np.min(d)))
    return float(np.mean(dists)) if dists else 999.0


def _snap_points_to_detected_circles(gray: np.ndarray, points: list[BubblePoint]) -> list[BubblePoint] | None:
    raw = _detect_raw_circles(gray)
    if raw is None:
        return None
    snapped: list[BubblePoint] = []
    # Keep snapping local so layout semantics are preserved.
    max_dist = 24.0
    for pt in points:
        d = np.sqrt((raw[:, 0] - pt.x) ** 2 + (raw[:, 1] - pt.y) ** 2)
        idx = int(np.argmin(d))
        if float(d[idx]) <= max_dist:
            c = raw[idx]
            snapped.append(
                BubblePoint(
                    question_id=pt.question_id,
                    option=pt.option,
                    x=int(c[0]),
                    y=int(c[1]),
                    radius=int(max(6, min(24, c[2]))),
                )
            )
        else:
            snapped.append(pt)
    return snapped


def _points_for_block(points: list[BubblePoint], start_q: int, end_q: int) -> list[BubblePoint]:
    return [p for p in points if start_q <= p.question_id <= end_q]


def _estimate_block_transform(expected: list[BubblePoint], raw_circles: np.ndarray) -> np.ndarray | None:
    if not expected or raw_circles is None or len(raw_circles) < 8:
        return None
    src = []
    dst = []
    max_dist = 28.0
    for p in expected:
        d = np.sqrt((raw_circles[:, 0] - p.x) ** 2 + (raw_circles[:, 1] - p.y) ** 2)
        idx = int(np.argmin(d))
        if float(d[idx]) <= max_dist:
            src.append([p.x, p.y])
            dst.append([raw_circles[idx, 0], raw_circles[idx, 1]])
    if len(src) < 8:
        return None
    src_arr = np.array(src, dtype=np.float32)
    dst_arr = np.array(dst, dtype=np.float32)
    M, _inliers = cv2.estimateAffinePartial2D(src_arr, dst_arr, method=cv2.RANSAC, ransacReprojThreshold=4.0)
    return M


def _apply_affine_to_block(points: list[BubblePoint], start_q: int, end_q: int, M: np.ndarray) -> list[BubblePoint]:
    out: list[BubblePoint] = []
    for p in points:
        if start_q <= p.question_id <= end_q:
            vec = np.array([p.x, p.y, 1.0], dtype=np.float32)
            mapped = M @ vec
            out.append(
                BubblePoint(
                    question_id=p.question_id,
                    option=p.option,
                    x=int(round(mapped[0])),
                    y=int(round(mapped[1])),
                    radius=p.radius,
                )
            )
        else:
            out.append(p)
    return out


def _local_block_warp_points(gray: np.ndarray, points: list[BubblePoint]) -> list[BubblePoint] | None:
    raw = _detect_raw_circles(gray)
    if raw is None:
        return None
    out = points
    for start_q, end_q in ((1, 10), (11, 20), (21, 30)):
        block_pts = _points_for_block(out, start_q, end_q)
        M = _estimate_block_transform(block_pts, raw)
        if M is not None:
            out = _apply_affine_to_block(out, start_q, end_q, M)
    return out


def _detect_template_points(gray: np.ndarray, config: OMRConfig) -> list[BubblePoint] | None:
    raw = _detect_raw_circles(gray)
    if raw is None:
        return None

    if len(raw) < config.questions * len(config.options):
        return None

    # Filter obvious outliers by radius.
    radii = raw[:, 2]
    r_med = float(np.median(radii))
    raw = raw[(radii >= 0.7 * r_med) & (radii <= 1.5 * r_med)]
    if len(raw) < config.questions * len(config.options):
        return None

    row_groups = _cluster_rows(raw, y_tol=max(6.0, r_med * 1.2))
    row_groups = [g for g in row_groups if len(g) >= 3]
    if len(row_groups) < config.questions:
        return None

    # Select the best consecutive 30 rows (regular spacing + enough circles).
    best_score = -1e9
    best_rows = None
    for i in range(0, len(row_groups) - config.questions + 1):
        window = row_groups[i : i + config.questions]
        ys = np.array([np.mean([c[1] for c in row]) for row in window], dtype=np.float32)
        gaps = np.diff(ys)
        regularity = -float(np.std(gaps)) if len(gaps) else -999.0
        richness = float(sum(min(len(row), 6) for row in window))
        score = regularity * 8.0 + richness
        if score > best_score:
            best_score = score
            best_rows = window
    if best_rows is None:
        return None

    x_values = np.array([c[0] for row in best_rows for c in row], dtype=np.float32)
    try:
        x_centers = _kmeans_1d(x_values, k=len(config.options))
    except ValueError:
        return None

    points: list[BubblePoint] = []
    for q_idx, row in enumerate(best_rows, start=1):
        row_sorted = sorted(row, key=lambda c: c[0])
        selected = []
        for xc in x_centers:
            nearest = min(row_sorted, key=lambda c: abs(c[0] - xc))
            selected.append(nearest)
        selected = sorted(selected, key=lambda c: c[0])
        if len(selected) != len(config.options):
            return None
        for opt, circ in zip(config.options, selected):
            points.append(
                BubblePoint(
                    question_id=q_idx,
                    option=opt,
                    x=int(circ[0]),
                    y=int(circ[1]),
                    radius=int(max(6, min(24, circ[2]))),
                )
            )
    if len(points) != config.questions * len(config.options):
        return None
    return points


def _detect_template_points_three_blocks(gray: np.ndarray, config: OMRConfig) -> list[BubblePoint] | None:
    """
    Fallback detector for common OMR layouts: 3 blocks x 10 questions,
    each question having 4 options (A-D).
    """
    raw = _detect_raw_circles(gray)
    if raw is None:
        return None
    if len(raw) < 100:
        return None
    r_med = int(np.median(raw[:, 2]))
    xs = raw[:, 0].astype(np.float32)
    ys = raw[:, 1].astype(np.float32)

    try:
        x_centers = _kmeans_1d(xs, 12)
        y_centers = _kmeans_1d(ys, 10)
    except ValueError:
        return None

    x_centers = np.sort(x_centers)
    y_centers = np.sort(y_centers)

    points: list[BubblePoint] = []
    # q1-10 in block 1, q11-20 in block 2, q21-30 in block 3.
    for block_idx in range(3):
        start_q = block_idx * 10 + 1
        block_x = x_centers[block_idx * 4 : block_idx * 4 + 4]
        for row_idx in range(10):
            qid = start_q + row_idx
            y = int(round(y_centers[row_idx]))
            for opt_idx, opt in enumerate(config.options):
                x = int(round(block_x[opt_idx]))
                points.append(
                    BubblePoint(
                        question_id=qid,
                        option=opt,
                        x=x,
                        y=y,
                        radius=max(6, min(24, r_med)),
                    )
                )
    if len(points) != config.questions * len(config.options):
        return None
    return points


def _detect_template_points_three_blocks_vertical(gray: np.ndarray, config: OMRConfig) -> list[BubblePoint] | None:
    """
    Alternative layout:
    3 horizontal blocks of 10 questions, each question with options stacked vertically.
    """
    raw = _detect_raw_circles(gray)
    if raw is None:
        return None
    if len(raw) < 100:
        return None

    r_med = int(np.median(raw[:, 2]))
    xs = raw[:, 0].astype(np.float32)
    ys = raw[:, 1].astype(np.float32)
    try:
        x_centers = _kmeans_1d(xs, 10)
        y_centers = _kmeans_1d(ys, 12)
    except ValueError:
        return None

    x_centers = np.sort(x_centers)
    y_centers = np.sort(y_centers)

    points: list[BubblePoint] = []
    for block_idx in range(3):
        start_q = block_idx * 10 + 1
        block_y = y_centers[block_idx * 4 : block_idx * 4 + 4]
        for col_idx in range(10):
            qid = start_q + col_idx
            x = int(round(x_centers[col_idx]))
            for opt_idx, opt in enumerate(config.options):
                y = int(round(block_y[opt_idx]))
                points.append(
                    BubblePoint(
                        question_id=qid,
                        option=opt,
                        x=x,
                        y=y,
                        radius=max(6, min(24, r_med)),
                    )
                )

    if len(points) != config.questions * len(config.options):
        return None
    return points


def _safe_crop(gray: np.ndarray, x: int, y: int, r: int) -> np.ndarray:
    h, w = gray.shape
    x1 = max(0, x - r)
    y1 = max(0, y - r)
    x2 = min(w, x + r)
    y2 = min(h, y + r)
    return gray[y1:y2, x1:x2]


def _fill_ratio(gray_patch: np.ndarray) -> float:
    if gray_patch.size == 0:
        return 0.0
    blur = cv2.GaussianBlur(gray_patch, (3, 3), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return float(np.count_nonzero(th)) / float(th.size)


def _cnn_bonus(cnn_model: BubbleCNNModel | None, patch_bgr: np.ndarray) -> float:
    if cnn_model is None or patch_bgr.size == 0:
        return 0.0
    proba = cnn_model.predict_label_proba(patch_bgr)
    # Encourage filled, penalize invalid/crossed for selected option confidence.
    return 0.35 * proba.get("filled", 0.0) - 0.15 * proba.get("invalid", 0.0) - 0.05 * proba.get("crossed", 0.0)


def _predict_from_points(
    gray: np.ndarray,
    points: list[BubblePoint],
    config: OMRConfig,
    cnn_model: BubbleCNNModel | None,
    use_cnn: bool = True,
) -> list[QuestionPrediction]:
    grouped: dict[int, list[tuple[str, float]]] = {q: [] for q in range(1, config.questions + 1)}
    for pt in points:
        patch = _safe_crop(gray, pt.x, pt.y, pt.radius)
        ratio = _fill_ratio(patch)
        patch_bgr = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR) if patch.size else np.zeros((1, 1, 3), dtype=np.uint8)
        cnn_boost = _cnn_bonus(cnn_model, patch_bgr) if use_cnn else 0.0
        score = ratio + cnn_boost
        grouped[pt.question_id].append((pt.option, score))

    predictions: list[QuestionPrediction] = []
    for qid in range(1, config.questions + 1):
        option_scores = dict(grouped[qid])
        sorted_opts = sorted(option_scores.items(), key=lambda kv: kv[1], reverse=True)
        top_opt, top_score = sorted_opts[0]
        second_score = sorted_opts[1][1]
        confidence = float(np.clip(top_score - second_score + top_score, 0.0, 1.0))

        if top_score < config.blank_threshold:
            answer = "BLANK"
            reason = "low_fill"
        elif abs(top_score - second_score) <= config.multi_margin:
            answer = "MULTI"
            reason = "close_top_two"
        elif confidence < config.uncertain_threshold:
            answer = top_opt
            reason = "low_confidence"
        else:
            answer = top_opt
            reason = "confident"

        predictions.append(
            QuestionPrediction(
                question_id=qid,
                answer=answer,
                confidence=confidence,
                reason=reason,
                option_scores=option_scores,
            )
        )
    return predictions


def _perturb_points(points: list[BubblePoint], dx: int, dy: int, r_scale: float) -> list[BubblePoint]:
    out: list[BubblePoint] = []
    for p in points:
        out.append(
            BubblePoint(
                question_id=p.question_id,
                option=p.option,
                x=p.x + dx,
                y=p.y + dy,
                radius=max(6, min(28, int(round(p.radius * r_scale)))),
            )
        )
    return out


def _alignment_score(
    gray: np.ndarray,
    points: list[BubblePoint],
    config: OMRConfig,
    cnn_model: BubbleCNNModel | None = None,
    use_cnn: bool = False,
) -> tuple[tuple[float, float], list[QuestionPrediction]]:
    """Score a template alignment without using the answer key."""
    preds = _predict_from_points(gray, points, config, cnn_model, use_cnn=use_cnn and cnn_model is not None)
    drift = _template_drift_pixels(gray, points)
    avg_conf = float(np.mean([p.confidence for p in preds])) if preds else 0.0
    # Prefer lower drift and higher confidence.
    return (-drift, avg_conf), preds


def evaluate_sheet(
    image_path: Path,
    config: OMRConfig,
    answer_key: dict[int, str],
    cnn_model: BubbleCNNModel | None = None,
    calibrated_points: list[BubblePoint] | None = None,
) -> tuple[list[QuestionPrediction], dict[str, float], np.ndarray]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    candidates: list[list[BubblePoint]] = []
    if calibrated_points is not None:
        candidates.append(calibrated_points)
        warped = _local_block_warp_points(gray, calibrated_points)
        if warped is not None:
            candidates.append(warped)
        snapped = _snap_points_to_detected_circles(gray, calibrated_points)
        if snapped is not None:
            candidates.append(snapped)
            warped_snapped = _local_block_warp_points(gray, snapped)
            if warped_snapped is not None:
                candidates.append(warped_snapped)
            # Fine local perturbation around calibrated+snapped points.
            for dx, dy, rs in [
                (-8, 0, 1.0),
                (8, 0, 1.0),
                (0, -8, 1.0),
                (0, 8, 1.0),
                (-6, -6, 1.05),
                (6, 6, 1.05),
                (0, 0, 0.9),
                (0, 0, 1.1),
            ]:
                candidates.append(_perturb_points(snapped, dx=dx, dy=dy, r_scale=rs))
    for fn in (
        _detect_template_points,
        _detect_template_points_three_blocks,
        _detect_template_points_three_blocks_vertical,
    ):
        pts = fn(gray, config)
        if pts is not None:
            candidates.append(pts)
            snapped = _snap_points_to_detected_circles(gray, pts)
            if snapped is not None:
                candidates.append(snapped)
    candidates.append(build_template_points(config))

    # Stage-1: fast alignment selection without CNN or answer-key leakage.
    scored_candidates: list[tuple[tuple[float, float], list[BubblePoint]]] = []
    for pts in candidates:
        score_tuple, _ = _alignment_score(gray, pts, config, cnn_model=None, use_cnn=False)
        scored_candidates.append((score_tuple, pts))

    scored_candidates.sort(key=lambda x: x[0], reverse=True)
    shortlisted = [pts for _, pts in scored_candidates[:4]]

    # Stage-2: run full model (with CNN if available) only on top candidates.
    best_predictions: list[QuestionPrediction] | None = None
    best_points: list[BubblePoint] | None = None
    best_score_tuple = (-9999.0, -1.0)
    for pts in shortlisted:
        score_tuple, preds = _alignment_score(gray, pts, config, cnn_model=cnn_model, use_cnn=True)
        if score_tuple > best_score_tuple:
            best_score_tuple = score_tuple
            best_predictions = preds
            best_points = pts

    assert best_predictions is not None and best_points is not None
    score_summary = compute_score_summary(best_predictions, answer_key)
    drift_px = _template_drift_pixels(gray, best_points)
    score_summary["template_drift_px"] = round(drift_px, 2)
    score_summary["template_drift_warning"] = bool(drift_px > 10.0)
    overlay = draw_overlay(img.copy(), best_predictions, answer_key, best_points)
    return best_predictions, score_summary, overlay


def compute_score_summary(predictions: list[QuestionPrediction], answer_key: dict[int, str]) -> dict[str, float]:
    total = len(predictions)
    correct = 0
    wrong = 0
    unattempted = 0
    low_conf_count = 0
    confidences = []

    for p in predictions:
        gt = answer_key.get(p.question_id, "BLANK")
        pred = p.answer
        confidences.append(p.confidence)
        if p.reason == "low_confidence":
            low_conf_count += 1
        if pred == "BLANK":
            unattempted += 1
        elif pred == gt:
            correct += 1
        else:
            wrong += 1

    accuracy = (correct / total) if total else 0.0
    section_bounds = {"section_1": (1, 10), "section_2": (11, 20), "section_3": (21, 30)}
    section_accuracy: dict[str, float] = {}
    for name, (s, e) in section_bounds.items():
        sec_preds = [p for p in predictions if s <= p.question_id <= e]
        sec_total = len(sec_preds)
        sec_correct = sum(1 for p in sec_preds if p.answer == answer_key.get(p.question_id, ""))
        section_accuracy[name] = round((sec_correct / sec_total) * 100.0, 2) if sec_total else 0.0

    return {
        "total": total,
        "correct": correct,
        "wrong": wrong,
        "unattempted": unattempted,
        "accuracy": round(accuracy * 100.0, 2),
        "avg_confidence": round(float(np.mean(confidences)) if confidences else 0.0, 3),
        "low_confidence_count": low_conf_count,
        "section_1_accuracy": section_accuracy["section_1"],
        "section_2_accuracy": section_accuracy["section_2"],
        "section_3_accuracy": section_accuracy["section_3"],
    }


def draw_overlay(
    image: np.ndarray,
    predictions: list[QuestionPrediction],
    answer_key: dict[int, str],
    points: list[BubblePoint],
) -> np.ndarray:
    pred_map = {p.question_id: p for p in predictions}
    for pt in points:
        pred = pred_map[pt.question_id]
        gt = answer_key.get(pt.question_id, "")
        color = (170, 170, 170)
        if pred.answer == pt.option and pred.answer == gt:
            color = (0, 180, 0)
        elif pred.answer == pt.option and pred.answer != gt:
            color = (0, 0, 220)
        cv2.circle(image, (pt.x, pt.y), pt.radius + 2, color, 2)

    y = 35
    for p in predictions[:8]:
        txt = f"Q{p.question_id:02d}: {p.answer} ({p.confidence:.2f}) {p.reason}"
        cv2.putText(image, txt, (560, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (20, 20, 20), 1)
        y += 22
    return image


def predictions_to_dataframe(predictions: list[QuestionPrediction], answer_key: dict[int, str]) -> pd.DataFrame:
    rows = []
    for p in predictions:
        is_correct = p.answer == answer_key.get(p.question_id, "")
        review_required = (
            (p.reason in {"low_confidence", "close_top_two"})
            or p.answer == "MULTI"
            or (p.confidence < 0.75)
        )
        rows.append(
            {
                "question_id": p.question_id,
                "predicted_answer": p.answer,
                "ground_truth": answer_key.get(p.question_id, ""),
                "is_correct": is_correct,
                "review_required": review_required,
                "confidence": p.confidence,
                "reason": p.reason,
                "score_A": p.option_scores.get("A", 0.0),
                "score_B": p.option_scores.get("B", 0.0),
                "score_C": p.option_scores.get("C", 0.0),
                "score_D": p.option_scores.get("D", 0.0),
            }
        )
    return pd.DataFrame(rows)
