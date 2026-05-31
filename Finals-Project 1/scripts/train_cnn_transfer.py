from pathlib import Path
import argparse
import sys

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from omr.cnn_model import train_and_save_cnn
from omr.config import BUBBLE_DATASET_DIR, MODEL_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CPU-friendly transfer learning CNN for bubble states on real dataset images.")
    parser.add_argument("--per-class", type=int, default=600, help="Synthetic samples per class (used only when augmentation is on)")
    parser.add_argument("--epochs", type=int, default=4, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--no-synth-aug", action="store_true", help="Disable synthetic augmentation")
    args = parser.parse_args()

    model_path = MODEL_DIR / "bubble_cnn_transfer.keras"
    metrics = train_and_save_cnn(
        model_path=model_path,
        dataset_dir=BUBBLE_DATASET_DIR,
        per_class=args.per_class,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_synthetic_aug=not args.no_synth_aug,
    )
    print("Transfer-learning CNN training complete.")
    print(f"Saved model: {model_path}")
    print(f"Real samples loaded: {metrics['real_samples']}")
    print(f"Train samples: {metrics['train_samples_total']}")
    print(f"Val samples: {metrics['val_samples']}")
    print(f"Validation accuracy: {metrics['val_accuracy']:.4f}")
    print(f"Validation loss: {metrics['val_loss']:.4f}")


if __name__ == "__main__":
    main()
