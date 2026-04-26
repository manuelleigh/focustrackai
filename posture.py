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
