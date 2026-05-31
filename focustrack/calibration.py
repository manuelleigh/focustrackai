from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd

from focustrack.config import DetectionThresholds


@dataclass
class CalibrationProfile:
    ear_open_baseline: float = 0.28
    gaze_center_baseline: float = 0.50
    posture_score_baseline: float = 80.0
    ear_closed_threshold: float = 0.20
    gaze_center_min: float = 0.35
    gaze_center_max: float = 0.65


def calibration_path(data_dir: Path) -> Path:
    return data_dir / "calibration_profile.json"


def build_calibration_profile(samples: pd.DataFrame) -> CalibrationProfile:
    if samples.empty:
        return CalibrationProfile()

    ear = pd.to_numeric(samples.get("avg_ear", pd.Series(dtype=float)), errors="coerce").dropna()
    gaze = pd.to_numeric(samples.get("gaze_ratio", pd.Series(dtype=float)), errors="coerce").dropna()
    posture = pd.to_numeric(samples.get("posture_score", pd.Series(dtype=float)), errors="coerce").dropna()

    ear_open = float(ear.median()) if not ear.empty else 0.28
    gaze_center = float(gaze.median()) if not gaze.empty else 0.50
    posture_score = float(posture.median()) if not posture.empty else 80.0

    return CalibrationProfile(
        ear_open_baseline=round(ear_open, 4),
        gaze_center_baseline=round(gaze_center, 4),
        posture_score_baseline=round(posture_score, 2),
        ear_closed_threshold=round(max(0.12, ear_open * 0.72), 4),
        gaze_center_min=round(max(0.15, gaze_center - 0.15), 4),
        gaze_center_max=round(min(0.85, gaze_center + 0.15), 4),
    )


def save_calibration_profile(profile: CalibrationProfile, data_dir: Path) -> Path:
    destination = calibration_path(data_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(asdict(profile), indent=2), encoding="utf-8")
    return destination


def load_calibration_profile(data_dir: Path) -> CalibrationProfile | None:
    source = calibration_path(data_dir)
    if not source.exists():
        return None
    return CalibrationProfile(**json.loads(source.read_text(encoding="utf-8")))


def apply_calibration(thresholds: DetectionThresholds, profile: CalibrationProfile | None) -> DetectionThresholds:
    if profile is None:
        return thresholds
    thresholds.ear_closed = profile.ear_closed_threshold
    thresholds.gaze_center_min = profile.gaze_center_min
    thresholds.gaze_center_max = profile.gaze_center_max
    return thresholds
