from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


FEATURE_COLUMNS = [
    "attention_component",
    "object_component",
    "posture_component",
    "screen_component",
    "fatigue_score",
    "avg_ear",
    "posture_score",
    "phone_detected",
    "hand_on_face",
    "person_present",
    "attention_state",
    "gaze_direction",
    "posture_state",
    "object_state",
    "screen_category",
]
TARGET_COLUMN = "productivity_label"


@dataclass
class TrainingResult:
    trained: bool
    source: str
    rows: int
    accuracy: float | None
    labels: list[str]
    message: str
    model_path: Path
    report: str = ""


def model_path(data_dir: Path) -> Path:
    return data_dir / "productivity_classifier.joblib"


def prepare_features(history: pd.DataFrame) -> pd.DataFrame:
    frame = history.copy()

    for column in FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan

    boolean_columns = ["phone_detected", "hand_on_face", "person_present"]
    for column in boolean_columns:
        frame[column] = frame[column].fillna(False).astype(bool).astype(int)

    numeric_columns = [
        "attention_component",
        "object_component",
        "posture_component",
        "screen_component",
        "fatigue_score",
        "avg_ear",
        "posture_score",
    ]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0.0)

    categorical_columns = [
        "attention_state",
        "gaze_direction",
        "posture_state",
        "object_state",
        "screen_category",
    ]
    for column in categorical_columns:
        frame[column] = frame[column].fillna("desconocido").astype(str)

    return frame[FEATURE_COLUMNS]


def train_classifier(history: pd.DataFrame, data_dir: Path) -> TrainingResult:
    training_data, source = _training_dataset(history)
    destination = model_path(data_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if training_data.empty or TARGET_COLUMN not in training_data.columns:
        return TrainingResult(
            trained=False,
            source="sin_datos",
            rows=0,
            accuracy=None,
            labels=[],
            message="No hay datos suficientes para entrenar el clasificador.",
            model_path=destination,
        )

    x = prepare_features(training_data)
    y = training_data[TARGET_COLUMN].astype(str)
    labels = sorted(y.unique().tolist())

    pipeline = _build_pipeline()
    accuracy: float | None = None
    report = ""

    if len(training_data) >= 12 and len(labels) >= 2 and y.value_counts().min() >= 2:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=0.25,
            random_state=42,
            stratify=y,
        )
        pipeline.fit(x_train, y_train)
        predictions = pipeline.predict(x_test)
        accuracy = float(accuracy_score(y_test, predictions))
        report = classification_report(y_test, predictions, zero_division=0)
    else:
        pipeline.fit(x, y)

    joblib.dump(
        {
            "pipeline": pipeline,
            "features": FEATURE_COLUMNS,
            "labels": labels,
            "source": source,
            "rows": len(training_data),
        },
        destination,
    )

    message = "Clasificador entrenado con historico real." if source == "historico" else "Clasificador entrenado con datos simulados para demo academica."
    return TrainingResult(
        trained=True,
        source=source,
        rows=len(training_data),
        accuracy=accuracy,
        labels=labels,
        message=message,
        model_path=destination,
        report=report,
    )


def predict_history(history: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    destination = model_path(data_dir)
    if history.empty or not destination.exists():
        return pd.DataFrame()

    bundle = joblib.load(destination)
    pipeline: Pipeline = bundle["pipeline"]
    predictions = pipeline.predict(prepare_features(history))

    result = history.copy()
    result["ai_prediction"] = predictions
    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(prepare_features(history))
        result["ai_confidence"] = probabilities.max(axis=1).round(3)
    else:
        result["ai_confidence"] = np.nan

    return result


def _build_pipeline() -> Pipeline:
    numeric_features = [
        "attention_component",
        "object_component",
        "posture_component",
        "screen_component",
        "fatigue_score",
        "avg_ear",
        "posture_score",
        "phone_detected",
        "hand_on_face",
        "person_present",
    ]
    categorical_features = [
        "attention_state",
        "gaze_direction",
        "posture_state",
        "object_state",
        "screen_category",
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        random_state=42,
        class_weight="balanced",
    )
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])


def _training_dataset(history: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    if not history.empty and TARGET_COLUMN in history.columns and history[TARGET_COLUMN].nunique() >= 2 and len(history) >= 12:
        return history.dropna(subset=[TARGET_COLUMN]).copy(), "historico"

    return _synthetic_dataset(), "simulado"


def _synthetic_dataset() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    templates = [
        ("Productivo", 95, 95, 90, 95, "atento", "centro", "correcta", "sin_objetos", "trabajo", False, False, True, 0.0),
        ("Productivo", 90, 100, 80, 100, "atento", "centro", "mejorable", "sin_objetos", "trabajo", False, False, True, 0.05),
        ("Regular", 55, 100, 65, 60, "desviado", "derecha", "mejorable", "sin_objetos", "neutral", False, False, True, 0.1),
        ("Regular", 65, 75, 55, 60, "atento", "centro", "encorvada", "mano_en_rostro", "neutral", False, True, True, 0.15),
        ("Distraido", 15, 30, 35, 15, "somnoliento", "desconocida", "encorvada", "celular_detectado", "distraccion", True, True, True, 0.9),
        ("Distraido", 0, 0, 50, 50, "ausente", "desconocida", "sin_datos", "usuario_ausente", "neutral", False, False, False, 0.0),
        ("Distraido", 35, 30, 40, 15, "desviado", "izquierda", "encorvada", "celular_detectado", "distraccion", True, False, True, 0.2),
    ]

    rng = np.random.default_rng(42)
    for label, attention, objects, posture, screen, attention_state, gaze, posture_state, object_state, screen_category, phone, hand, present, fatigue in templates:
        for _ in range(18):
            rows.append(
                {
                    "productivity_label": label,
                    "attention_component": _jitter(attention, rng),
                    "object_component": _jitter(objects, rng),
                    "posture_component": _jitter(posture, rng),
                    "screen_component": _jitter(screen, rng),
                    "fatigue_score": max(0.0, min(1.0, float(fatigue + rng.normal(0, 0.04)))),
                    "avg_ear": max(0.0, float(0.28 - fatigue * 0.12 + rng.normal(0, 0.02))),
                    "posture_score": _jitter(posture, rng),
                    "phone_detected": phone,
                    "hand_on_face": hand,
                    "person_present": present,
                    "attention_state": attention_state,
                    "gaze_direction": gaze,
                    "posture_state": posture_state,
                    "object_state": object_state,
                    "screen_category": screen_category,
                }
            )

    return pd.DataFrame(rows)


def _jitter(value: float, rng: np.random.Generator) -> float:
    return max(0.0, min(100.0, float(value + rng.normal(0, 6))))
