from __future__ import annotations

import cv2

from focustrack.config import DetectionThresholds
from focustrack.models import PostureMetrics
from focustrack.vision.mp_compat import HAS_MEDIAPIPE_SOLUTIONS, MP_SOLUTIONS


class PostureAnalyzer:
    def __init__(self, thresholds: DetectionThresholds):
        self.thresholds = thresholds
        self.pose = None
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )

        if HAS_MEDIAPIPE_SOLUTIONS:
            self.pose = MP_SOLUTIONS.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def analyze(self, frame) -> tuple[PostureMetrics, dict[str, object]]:
        return PostureMetrics(
            posture_state="sin_datos",
            posture_score=50.0,
        ), {}

    def close(self) -> None:
        if self.pose is not None:
            self.pose.close()