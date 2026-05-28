from __future__ import annotations

import cv2

from focustrack.config import DetectionThresholds
from focustrack.models import PostureMetrics
from focustrack.vision.mp_compat import HAS_MEDIAPIPE_SOLUTIONS, MP_SOLUTIONS


class PostureAnalyzer:
    def __init__(self, thresholds: DetectionThresholds):
        self.thresholds = thresholds
        self.pose = None

        if HAS_MEDIAPIPE_SOLUTIONS:
            self.pose = MP_SOLUTIONS.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def analyze(self, frame) -> tuple[PostureMetrics, dict[str, object]]:
        if self.pose is None:
            return PostureMetrics(
                posture_state="sin_datos",
                posture_score=50.0,
            ), {}

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb_frame)

        debug: dict[str, object] = {
            "pose_landmarks": None
        }

        if not result.pose_landmarks:
            return PostureMetrics(
                posture_state="sin_datos",
                posture_score=50.0,
            ), debug

        debug["pose_landmarks"] = result.pose_landmarks

        landmarks = result.pose_landmarks.landmark

        nose = landmarks[
            MP_SOLUTIONS.pose.PoseLandmark.NOSE.value
        ]

        left_shoulder = landmarks[
            MP_SOLUTIONS.pose.PoseLandmark.LEFT_SHOULDER.value
        ]

        right_shoulder = landmarks[
            MP_SOLUTIONS.pose.PoseLandmark.RIGHT_SHOULDER.value
        ]

        left_hip = landmarks[
            MP_SOLUTIONS.pose.PoseLandmark.LEFT_HIP.value
        ]

        right_hip = landmarks[
            MP_SOLUTIONS.pose.PoseLandmark.RIGHT_HIP.value
        ]

        shoulder_tilt = abs(
            left_shoulder.y - right_shoulder.y
        )

        torso_center_x = (
            left_shoulder.x + right_shoulder.x
        ) / 2.0

        hip_center_x = (
            left_hip.x + right_hip.x
        ) / 2.0

        torso_lean = abs(
            torso_center_x - hip_center_x
        )

        head_offset = abs(
            nose.x - torso_center_x
        )

        tilt_penalty = (
            shoulder_tilt /
            max(self.thresholds.shoulder_tilt_max, 1e-6)
        ) * 25.0

        lean_penalty = (
            torso_lean /
            max(self.thresholds.torso_lean_max, 1e-6)
        ) * 35.0

        head_penalty = (
            head_offset /
            max(self.thresholds.head_offset_max, 1e-6)
        ) * 25.0

        score = max(
            0.0,
            100.0
            - tilt_penalty
            - lean_penalty
            - head_penalty
        )

        if score >= 75.0:
            posture_state = "correcta"
        elif score >= 50.0:
            posture_state = "mejorable"
        else:
            posture_state = "encorvada"

        metrics = PostureMetrics(
            posture_state=posture_state,
            posture_score=score,
            shoulder_tilt=shoulder_tilt,
            torso_lean=torso_lean,
            head_offset=head_offset,
            confidence=round(
                min(1.0, max(0.45, score / 100.0)),
                4,
            ),
        )

        return metrics, debug

    def close(self) -> None:
        if self.pose is not None:
            self.pose.close()