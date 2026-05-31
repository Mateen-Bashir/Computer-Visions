from pathlib import Path
import argparse
import json
import sys

import cv2

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))


def _interpolate_block(tl, tr, bl, br, start_q: int):
    """
    Build 10x4 points for one block from 4 corner bubble centers:
    tl=A@firstQ, tr=D@firstQ, bl=A@lastQ, br=D@lastQ
    """
    points = []
    for r in range(10):
        t = r / 9.0
        left_x = tl[0] * (1 - t) + bl[0] * t
        left_y = tl[1] * (1 - t) + bl[1] * t
        right_x = tr[0] * (1 - t) + br[0] * t
        right_y = tr[1] * (1 - t) + br[1] * t
        qid = start_q + r
        for c, opt in enumerate(("A", "B", "C", "D")):
            s = c / 3.0
            x = left_x * (1 - s) + right_x * s
            y = left_y * (1 - s) + right_y * s
            points.append(
                {
                    "question_id": qid,
                    "option": opt,
                    "x": int(round(x)),
                    "y": int(round(y)),
                    "radius": 14,
                }
            )
    return points


def main() -> None:
    parser = argparse.ArgumentParser(description="Create exact template points for real OMR sheet.")
    parser.add_argument("--image", required=True, help="Path to reference sheet image")
    parser.add_argument("--out", default="models/template_points.json", help="Output JSON for template points")
    args = parser.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        raise ValueError(f"Could not read image: {args.image}")

    display = img.copy()
    clicked = []
    labels = [
        "B1 Q1-A", "B1 Q1-D", "B1 Q10-A", "B1 Q10-D",
        "B2 Q11-A", "B2 Q11-D", "B2 Q20-A", "B2 Q20-D",
        "B3 Q21-A", "B3 Q21-D", "B3 Q30-A", "B3 Q30-D",
    ]

    def on_mouse(event, x, y, _flags, _param):
        nonlocal display
        if event == cv2.EVENT_LBUTTONDOWN and len(clicked) < len(labels):
            clicked.append((x, y))
            idx = len(clicked) - 1
            cv2.circle(display, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(display, str(idx + 1), (x + 6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.namedWindow("template_points")
    cv2.setMouseCallback("template_points", on_mouse)

    print("Click 12 points in this exact order:")
    for i, name in enumerate(labels, start=1):
        print(f"{i:2d}. {name}")
    print("Keys: r=reset, s=save (after 12 clicks), q=quit")

    while True:
        cv2.imshow("template_points", display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            clicked.clear()
            display = img.copy()
        if key == ord("s") and len(clicked) == len(labels):
            b1 = _interpolate_block(clicked[0], clicked[1], clicked[2], clicked[3], start_q=1)
            b2 = _interpolate_block(clicked[4], clicked[5], clicked[6], clicked[7], start_q=11)
            b3 = _interpolate_block(clicked[8], clicked[9], clicked[10], clicked[11], start_q=21)
            data = {"points": b1 + b2 + b3}
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            print(f"Saved: {out_path.resolve()}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
