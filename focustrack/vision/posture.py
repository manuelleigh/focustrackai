from __future__ import annotations

import cv2
import mediapipe as mp

from focustrack.config import DetectionThresholds
from focustrack.models import PostureMetrics


class PostureAnalyzer:
    def __init__(self, thresholds: DetectionThresholds):
        self.thresholds = thresholds

        self.pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def analyze(self, frame) -> tuple[PostureMetrics, dict[str, object]]:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb_frame)
        debug: dict[str, object] = {"pose_landmarks": None}

        if not result.pose_landmarks:
            return PostureMetrics(posture_state="sin_datos", posture_score=50.0), debug
