from pathlib import Path
import argparse
import sys

import cv2

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from omr.calibration import compute_calibration_from_points, save_calibration


def main() -> None:
    parser = argparse.ArgumentParser(description="One-time 4-point calibration for real OMR sheets.")
    parser.add_argument("--image", required=True, help="Path to a real OMR sheet image")
    parser.add_argument("--out", default="models/template_calibration.json", help="Output calibration json path")
    parser.add_argument("--width", type=int, default=1000, help="Canonical width")
    parser.add_argument("--height", type=int, default=1400, help="Canonical height")
    args = parser.parse_args()

    image_path = Path(args.image)
    out_path = Path(args.out)
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    points: list[tuple[int, int]] = []
    display = img.copy()

    def on_mouse(event, x, y, _flags, _param):
        nonlocal display
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            cv2.circle(display, (x, y), 6, (0, 0, 255), -1)
            cv2.putText(display, str(len(points)), (x + 8, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.namedWindow("calibration")
    cv2.setMouseCallback("calibration", on_mouse)

    print("Click 4 corners in order: top-left, top-right, bottom-right, bottom-left.")
    print("Press 'r' to reset points, 's' to save after 4 points, 'q' to quit.")

    while True:
        cv2.imshow("calibration", display)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            points.clear()
            display = img.copy()
        if key == ord("s") and len(points) == 4:
            matrix, w, h = compute_calibration_from_points(points, args.width, args.height)
            save_calibration(out_path, matrix, w, h)
            print(f"Saved calibration to: {out_path.resolve()}")
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
