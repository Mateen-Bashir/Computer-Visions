from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    return df


def load_answer_key(answer_key_path: Path) -> pd.DataFrame:
    df = pd.read_csv(answer_key_path)
    df = _normalize_columns(df)
    df = df.dropna(how="all")
    if "Image_ID" not in df.columns:
        raise ValueError("answer_key.csv must contain Image_ID column")
    return df


def get_answers_for_sheet(answer_key_df: pd.DataFrame, sheet_id: int) -> dict[int, str]:
    row = answer_key_df[answer_key_df["Image_ID"] == sheet_id]
    if row.empty:
        raise ValueError(f"Sheet id {sheet_id} not found in answer key")
    row = row.iloc[0]
    answers: dict[int, str] = {}
    for col in answer_key_df.columns:
        if col.startswith("Q"):
            qid = int(col[1:])
            val = str(row[col]).strip().upper()
            answers[qid] = val
    return answers


def load_split_labels(split_csv_paths: Iterable[Path]) -> pd.DataFrame:
    rows = []
    for csv_path in split_csv_paths:
        df = pd.read_csv(csv_path)
        df = _normalize_columns(df)
        df["split"] = csv_path.parent.name
        rows.append(df)
    if not rows:
        raise ValueError("No label CSV files found.")
    return pd.concat(rows, ignore_index=True)
