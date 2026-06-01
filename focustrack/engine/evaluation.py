from __future__ import annotations

from collections import Counter

import pandas as pd


def build_labeled_dataset(
    history: pd.DataFrame,
    human_labels: pd.DataFrame,
    prediction_column: str = "productivity_label",
) -> pd.DataFrame:
    if history.empty or human_labels.empty:
        return pd.DataFrame()

    history = history.copy()
    human_labels = human_labels.copy()
    if "timestamp" in history.columns:
        history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce")
    for column in ["start_time", "end_time"]:
        if column in human_labels.columns:
            human_labels[column] = pd.to_datetime(human_labels[column], errors="coerce")

    labeled_rows: list[dict[str, object]] = []
    for label_row in human_labels.to_dict(orient="records"):
        session_id = label_row.get("session_id")
        if not session_id:
            continue

        session_history = history[history["session_id"] == session_id].copy()
        if session_history.empty:
            continue

        start_time = label_row.get("start_time")
        end_time = label_row.get("end_time")
        if pd.notna(start_time):
            session_history = session_history[session_history["timestamp"] >= start_time]
        if pd.notna(end_time):
            session_history = session_history[session_history["timestamp"] <= end_time]
        if session_history.empty:
            continue

        for snapshot_row in session_history.to_dict(orient="records"):
            labeled_rows.append(
                {
                    "session_id": session_id,
                    "timestamp": snapshot_row.get("timestamp"),
                    "predicted_label": snapshot_row.get(prediction_column),
                    "human_label": label_row.get("label"),
                    "prediction_score": snapshot_row.get("productivity_score"),
                    "human_notes": label_row.get("notes", ""),
                }
            )

    return pd.DataFrame(labeled_rows)


def evaluate_label_predictions(
    labeled_dataset: pd.DataFrame,
) -> dict[str, object]:
    if labeled_dataset.empty:
        return {
            "total_samples": 0,
            "accuracy": None,
            "macro_f1": None,
            "per_label": {},
            "confusion_matrix": {},
        }

    dataset = labeled_dataset.copy()
    dataset["predicted_label"] = dataset["predicted_label"].fillna("Sin prediccion")
    dataset["human_label"] = dataset["human_label"].fillna("Sin etiqueta")

    labels = sorted(
        set(dataset["predicted_label"].tolist()) | set(dataset["human_label"].tolist())
    )
    confusion_matrix: dict[str, dict[str, int]] = {
        truth: {pred: 0 for pred in labels} for truth in labels
    }
    correct = 0
    for row in dataset.to_dict(orient="records"):
        truth = str(row["human_label"])
        pred = str(row["predicted_label"])
        confusion_matrix[truth][pred] += 1
        if truth == pred:
            correct += 1

    per_label: dict[str, dict[str, float | int | None]] = {}
    f1_values: list[float] = []
    for label in labels:
        tp = confusion_matrix[label][label]
        fp = sum(confusion_matrix[truth][label] for truth in labels if truth != label)
        fn = sum(confusion_matrix[label][pred] for pred in labels if pred != label)
        support = sum(confusion_matrix[label].values())
        precision = tp / (tp + fp) if (tp + fp) else None
        recall = tp / (tp + fn) if (tp + fn) else None
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision is not None and recall is not None and (precision + recall) > 0
            else None
        )
        if f1 is not None:
            f1_values.append(f1)
        per_label[label] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    total_samples = int(len(dataset))
    return {
        "total_samples": total_samples,
        "accuracy": correct / total_samples if total_samples else None,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else None,
        "per_label": per_label,
        "confusion_matrix": confusion_matrix,
        "label_distribution": dict(Counter(dataset["human_label"].tolist())),
    }
