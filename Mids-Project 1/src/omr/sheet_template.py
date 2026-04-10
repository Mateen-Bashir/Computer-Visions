from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import OMRConfig


@dataclass
class BubblePoint:
    question_id: int
    option: str
    x: int
    y: int
    radius: int


def build_template_points(config: OMRConfig) -> list[BubblePoint]:
    points: list[BubblePoint] = []
    for q in range(1, config.questions + 1):
        y = config.start_y + (q - 1) * config.row_gap
        for i, opt in enumerate(config.options):
            x = config.start_x + i * config.col_gap
            points.append(BubblePoint(question_id=q, option=opt, x=x, y=y, radius=config.bubble_radius))
    return points


def create_blank_sheet(config: OMRConfig, width: int = 900, height: int = 1600) -> np.ndarray:
    img = np.full((height, width, 3), 255, dtype=np.uint8)
    cv2.putText(img, "AI-Based OMR Demo Sheet", (180, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)
    cv2.putText(img, "Options: A    B    C    D", (190, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 60, 60), 2)
    for pt in build_template_points(config):
        if pt.option == "A":
            cv2.putText(img, f"Q{pt.question_id:02d}", (80, pt.y + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.circle(img, (pt.x, pt.y), pt.radius, (0, 0, 0), 2)
    return img

