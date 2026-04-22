from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from focustrack.config import DetectionThresholds
from focustrack.models import AttentionMetrics

try:
    import dlib
except ImportError:  # pragma: no cover - optional dependency
    dlib = None


LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = [468, 469, 470, 471, 472]
RIGHT_IRIS = [473, 474, 475, 476, 477]
FACE_BBOX_REFERENCE = [10, 152, 234, 454]

class AttentionAnalyzer:
    def __init__(self, thresholds: DetectionThresholds, enable_dlib: bool = False, dlib_shape_predictor: Path | None = None):
        self.thresholds = thresholds
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.closed_eye_frames = 0
        self.blink_count = 0
        self.previous_eyes_closed = False
        self.use_dlib = False
        self.dlib_detector = None