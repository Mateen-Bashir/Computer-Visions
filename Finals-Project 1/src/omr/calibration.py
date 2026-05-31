from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .config import OMRConfig
from .sheet_template import BubblePoint, build_template_points


def _order_points(pts: np.ndarray) -> np.ndarray:
    # Order: top-left, top-right, bottom-right, bottom-left
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).ravel()
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = pts[np.argmin(s)]
    ordered[2] = pts[np.argmax(s)]
    ordered[1] = pts[np.argmin(diff)]
    ordered[3] = pts[np.argmax(diff)]
    return ordered


def save_calibration(calib_path: Path, matrix: np.ndarray, width: int, height: int) -> None:
    calib_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "matrix": matrix.tolist(),
        "target_width": int(width),
        "target_height": int(height),
    }
    calib_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_calibration(calib_path: Path) -> tuple[np.ndarray, int, int] | None:
    if not calib_path.exists():
        return None
    data = json.loads(calib_path.read_text(encoding="utf-8"))
    matrix = np.array(data["matrix"], dtype=np.float32)
    return matrix, int(data["target_width"]), int(data["target_height"])


def get_calibrated_template_points(
    image_shape: tuple[int, int],
    config: OMRConfig,
    calibration: tuple[np.ndarray, int, int],
) -> list[BubblePoint]:
    matrix, target_w, target_h = calibration
    h, w = image_shape

    # Build canonical points in calibrated target space, then map back.
    base_cfg = OMRConfig(
        questions=config.questions,
        options=config.options,
        bubble_radius=max(6, int(min(target_w, target_h) * 0.012)),
        start_x=int(target_w * 0.24),
        start_y=int(target_h * 0.16),
        col_gap=int(target_w * 0.145),
        row_gap=int(target_h * 0.026),
        blank_threshold=config.blank_threshold,
        multi_margin=config.multi_margin,
        uncertain_threshold=config.uncertain_threshold,
    )
    canonical = build_template_points(base_cfg)

    inv_m = np.linalg.inv(matrix)
    result: list[BubblePoint] = []
    for pt in canonical:
        src = np.array([[[pt.x, pt.y]]], dtype=np.float32)
        mapped = cv2.perspectiveTransform(src, inv_m)[0, 0]
        mapped_x = int(np.clip(mapped[0], 0, w - 1))
        mapped_y = int(np.clip(mapped[1], 0, h - 1))
        result.append(
            BubblePoint(
                question_id=pt.question_id,
                option=pt.option,
                x=mapped_x,
                y=mapped_y,
                radius=max(6, int(pt.radius * (w / max(target_w, 1)))),
            )
        )
    return result


def compute_calibration_from_points(
    clicked_points: list[tuple[int, int]],
    target_width: int = 1000,
    target_height: int = 1400,
) -> tuple[np.ndarray, int, int]:
    if len(clicked_points) != 4:
        raise ValueError("Exactly 4 points are required.")
    src = _order_points(np.array(clicked_points, dtype=np.float32))
    dst = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return matrix, target_width, target_height
