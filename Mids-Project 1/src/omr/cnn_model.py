from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
try:
    from tensorflow import keras
except Exception:  
    keras = None


CLASS_NAMES = ["crossed", "default", "filled", "invalid"]
IMG_SIZE = 96


def _draw_synthetic_bubble(label: str, size: int = IMG_SIZE) -> np.ndarray:
    img = np.full((size, size, 3), 255, dtype=np.uint8)
    center = (size // 2, size // 2)
    radius = random.randint(size // 4, size // 3)


    cv2.circle(img, center, radius, (0, 0, 0), random.randint(2, 4))

    if label == "filled":
        cv2.circle(img, center, radius - 4, (0, 0, 0), -1)
    elif label == "crossed":
        cv2.line(img, (center[0] - radius + 5, center[1] - radius + 5), (center[0] + radius - 5, center[1] + radius - 5), (0, 0, 0), 3)
        cv2.line(img, (center[0] - radius + 5, center[1] + radius - 5), (center[0] + radius - 5, center[1] - radius + 5), (0, 0, 0), 3)
    elif label == "invalid":
        pts = np.array(
            [
                [center[0] - radius + 6, center[1] - radius + 10],
                [center[0] + radius - 8, center[1] - radius + 12],
                [center[0] + radius - 12, center[1] + radius - 5],
                [center[0] - radius + 4, center[1] + radius - 8],
            ],
            np.int32,
        )
        cv2.fillPoly(img, [pts], (0, 0, 0))
  

 
    if random.random() < 0.5:
        noise = np.random.normal(0, random.uniform(5, 20), img.shape).astype(np.int16)
        img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    angle = random.uniform(-12, 12)
    M = cv2.getRotationMatrix2D(center, angle, random.uniform(0.95, 1.05))
    img = cv2.warpAffine(img, M, (size, size), borderValue=(255, 255, 255))
    return img


def make_synthetic_dataset(per_class: int = 1200) -> tuple[np.ndarray, np.ndarray]:
    if keras is None:
        raise ImportError("TensorFlow/Keras is not installed. CNN features are optional.")
    X, y = [], []
    for idx, label in enumerate(CLASS_NAMES):
        for _ in range(per_class):
            patch = _draw_synthetic_bubble(label)
            X.append(patch.astype(np.float32) / 255.0)
            y.append(idx)
    X = np.array(X, dtype=np.float32)
    y = keras.utils.to_categorical(np.array(y), num_classes=len(CLASS_NAMES))
    return X, y


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def _row_to_label(row: pd.Series) -> str | None:
    vals = [int(row.get(name, 0)) for name in CLASS_NAMES]
    if sum(vals) == 0:
        return None
    return CLASS_NAMES[int(np.argmax(vals))]


def load_real_dataset_from_splits(dataset_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    if keras is None:
        raise ImportError("TensorFlow/Keras is not installed. CNN features are optional.")
    X, y = [], []
    for split in ("train", "valid", "test"):
        split_dir = dataset_dir / split
        labels_path = split_dir / "labels.csv"
        images_dir = split_dir / "images"
        if not labels_path.exists() or not images_dir.exists():
            continue

        df = _normalize_columns(pd.read_csv(labels_path))
        if "filename" not in df.columns:
            continue

        for _, row in df.iterrows():
            label = _row_to_label(row)
            if label is None:
                continue
            img_name = str(row["filename"]).strip()
            img_path = images_dir / img_name
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            patch = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
            X.append(patch.astype(np.float32) / 255.0)
            y.append(CLASS_NAMES.index(label))

    if not X:
        raise ValueError(
            "No real bubble images loaded. Expected files in bubble_dataset/<split>/images with labels in labels.csv"
        )

    X_arr = np.array(X, dtype=np.float32)
    y_arr = keras.utils.to_categorical(np.array(y), num_classes=len(CLASS_NAMES))
    return X_arr, y_arr


def build_transfer_model(num_classes: int = 4) -> keras.Model:
    if keras is None:
        raise ImportError("TensorFlow/Keras is not installed. CNN features are optional.")
    base = keras.applications.MobileNetV2(
        include_top=False,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        weights="imagenet",
    )
    base.trainable = False
    inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = keras.applications.mobilenet_v2.preprocess_input(inputs * 255.0)
    x = base(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="categorical_crossentropy", metrics=["accuracy"])
    return model


@dataclass
class BubbleCNNModel:
    model: keras.Model

    def predict_label_proba(self, patch_bgr: np.ndarray) -> dict[str, float]:
        patch = cv2.resize(patch_bgr, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)
        x = patch.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)
        probs = self.model.predict(x, verbose=0)[0]
        return {name: float(prob) for name, prob in zip(CLASS_NAMES, probs)}


def train_and_save_cnn(
    model_path: Path,
    dataset_dir: Path,
    per_class: int = 600,
    epochs: int = 4,
    batch_size: int = 32,
    use_synthetic_aug: bool = True,
) -> dict[str, float]:
    if keras is None:
        raise ImportError(
            "TensorFlow/Keras is not available for this Python version. "
            "Run CNN on a supported Python version (e.g. 3.13 with tf-nightly)."
        )
    X_real, y_real = load_real_dataset_from_splits(dataset_dir)
    X, y = X_real, y_real
    if use_synthetic_aug:
        X_syn, y_syn = make_synthetic_dataset(per_class=per_class)
        X = np.concatenate([X_real, X_syn], axis=0)
        y = np.concatenate([y_real, y_syn], axis=0)

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(X))
    X, y = X[idx], y[idx]
    split = int(len(X) * 0.85)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = build_transfer_model(num_classes=len(CLASS_NAMES))
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size, verbose=2)
    loss, acc = model.evaluate(X_val, y_val, verbose=0)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    return {
        "val_loss": float(loss),
        "val_accuracy": float(acc),
        "real_samples": int(len(X_real)),
        "train_samples_total": int(len(X_train)),
        "val_samples": int(len(X_val)),
    }


def load_cnn_model(model_path: Path) -> BubbleCNNModel | None:
    if keras is None:
        return None
    if not model_path.exists():
        return None
    model = keras.models.load_model(model_path)
    return BubbleCNNModel(model=model)
