from pathlib import Path
import random
import sys

import cv2

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from omr.config import OMRConfig
from omr.data_utils import get_answers_for_sheet, load_answer_key
from omr.sheet_template import build_template_points, create_blank_sheet


def draw_answers(image, config: OMRConfig, answer_map: dict[int, str], noise_seed: int) -> None:
    random.seed(noise_seed)
    option_to_idx = {o: i for i, o in enumerate(config.options)}
    points = build_template_points(config)

    # Group by question for easier marking.
    grouped = {}
    for pt in points:
        grouped.setdefault(pt.question_id, []).append(pt)

    for qid, options in grouped.items():
        gt = answer_map.get(qid, "A")
        if gt not in option_to_idx:
            continue
        mark_multi = random.random() < 0.06
        mark_blank = random.random() < 0.06
        if mark_blank:
            continue

        idx = option_to_idx[gt]
        chosen = [idx]
        if mark_multi:
            extra = random.choice([i for i in range(4) if i != idx])
            chosen.append(extra)

        for ci in chosen:
            pt = options[ci]
            cv2.circle(image, (pt.x, pt.y), pt.radius - 4, (0, 0, 0), -1)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "demo_inputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = OMRConfig()
    answer_df = load_answer_key(root / "csv" / "answer_key.csv")

    for sheet_id in [1, 2, 3]:
        img = create_blank_sheet(config)
        gt = get_answers_for_sheet(answer_df, sheet_id)
        draw_answers(img, config, gt, noise_seed=sheet_id)
        out_path = out_dir / f"sheet_{sheet_id}.png"
        cv2.imwrite(str(out_path), img)
        print(f"Created {out_path}")


if __name__ == "__main__":
    main()
