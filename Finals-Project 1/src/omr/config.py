from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OMRConfig:
    questions: int = 30
    options: tuple[str, ...] = ("A", "B", "C", "D")
    bubble_radius: int = 16
    start_x: int = 220
    start_y: int = 150
    col_gap: int = 130
    row_gap: int = 45
    blank_threshold: float = 0.16
    multi_margin: float = 0.08
    uncertain_threshold: float = 0.45


ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT_DIR / "models"
OUTPUT_DIR = ROOT_DIR / "outputs"
ANSWER_KEY_PATH = ROOT_DIR / "csv" / "answer_key.csv"
BUBBLE_DATASET_DIR = ROOT_DIR / "bubble_dataset"
