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

        if enable_dlib and dlib is not None:
            self.dlib_detector = dlib.get_frontal_face_detector()
            if dlib_shape_predictor and Path(dlib_shape_predictor).exists():
                self.use_dlib = True

    def analyze(self, frame: np.ndarray) -> tuple[AttentionMetrics, dict[str, object]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb_frame)
        backend = "mediapipe+dlib" if self.use_dlib else "mediapipe"
        debug: dict[str, object] = {
            "face_landmarks": None,
            "face_bbox": None,
        }

        if not result.multi_face_landmarks:
            face_detected = self._dlib_face_detected(frame) if self.dlib_detector is not None else False
            self.closed_eye_frames = 0
            self.previous_eyes_closed = False
            metrics = AttentionMetrics(
                face_detected=face_detected,
                eyes_detected=False,
                eyes_closed=False,
                attention_state="ausente",
                gaze_direction="desconocida",
                fatigue_score=0.0,
                blink_count=self.blink_count,
                backend=backend,
            )
            return metrics, debug
        