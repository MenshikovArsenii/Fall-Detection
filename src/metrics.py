from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd


def evaluate_csv(pred_csv: str | Path, gt_csv: str | Path, frame_col: str = "frame") -> dict:
    pred = pd.read_csv(pred_csv)
    gt = pd.read_csv(gt_csv)

    if "label" not in pred.columns:
        raise ValueError("Prediction CSV must contain 'label' column")
    if "label" not in gt.columns:
        raise ValueError("Ground-truth CSV must contain 'label' column")

    merged = gt[[frame_col, "label"]].rename(columns={"label": "gt_label"}).merge(
        pred[[frame_col, "label"]].rename(columns={"label": "pred_label"}),
        on=frame_col,
        how="inner",
    )
    merged = merged[merged["gt_label"] != "ignore"]
    merged = merged[merged["pred_label"] != "ignore"]

    y_true = merged["gt_label"].eq("fall")
    y_pred = merged["pred_label"].eq("fall")

    tp = int((y_true & y_pred).sum())
    tn = int((~y_true & ~y_pred).sum())
    fp = int((~y_true & y_pred).sum())
    fn = int((y_true & ~y_pred).sum())

    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "frames_compared": int(total),
    }
