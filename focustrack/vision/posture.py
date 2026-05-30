from __future__ import annotations

import cv2

from focustrack.config import DetectionThresholds
from focustrack.models import PostureMetrics
from focustrack.vision.mp_compat import HAS_MEDIAPIPE_SOLUTIONS, MP_SOLUTIONS


class PostureAnalyzer:
    def __init__(self, thresholds: DetectionThresholds):
        self.thresholds = thresholds
        self.pose = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        if HAS_MEDIAPIPE_SOLUTIONS:
            self.pose = MP_SOLUTIONS.pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def analyze(self, frame) -> tuple[PostureMetrics, dict[str, object]]:
        if self.pose is None:
            return self._analyze_with_opencv(frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb_frame)
        debug: dict[str, object] = {"pose_landmarks": None}

        if not result.pose_landmarks:
            return PostureMetrics(posture_state="sin_datos", posture_score=50.0), debug

        debug["pose_landmarks"] = result.pose_landmarks
        landmarks = result.pose_landmarks.landmark
        nose = landmarks[MP_SOLUTIONS.pose.PoseLandmark.NOSE.value]
        left_shoulder = landmarks[MP_SOLUTIONS.pose.PoseLandmark.LEFT_SHOULDER.value]
        right_shoulder = landmarks[MP_SOLUTIONS.pose.PoseLandmark.RIGHT_SHOULDER.value]
        left_hip = landmarks[MP_SOLUTIONS.pose.PoseLandmark.LEFT_HIP.value]
        right_hip = landmarks[MP_SOLUTIONS.pose.PoseLandmark.RIGHT_HIP.value]

        shoulder_tilt = abs(left_shoulder.y - right_shoulder.y)
        torso_center_x = (left_shoulder.x + right_shoulder.x) / 2.0
        hip_center_x = (left_hip.x + right_hip.x) / 2.0
        torso_lean = abs(torso_center_x - hip_center_x)
        head_offset = abs(nose.x - torso_center_x)

        tilt_penalty = (shoulder_tilt / max(self.thresholds.shoulder_tilt_max, 1e-6)) * 25.0
        lean_penalty = (torso_lean / max(self.thresholds.torso_lean_max, 1e-6)) * 35.0
        head_penalty = (head_offset / max(self.thresholds.head_offset_max, 1e-6)) * 25.0
        score = max(0.0, 100.0 - tilt_penalty - lean_penalty - head_penalty)

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
            confidence=round(min(1.0, max(0.45, score / 100.0)), 4),
        )
        return metrics, debug

    def close(self) -> None:
        if self.pose is not None:
            self.pose.close()

    def _analyze_with_opencv(self, frame) -> tuple[PostureMetrics, dict[str, object]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        debug: dict[str, object] = {"pose_landmarks": None, "posture_bbox": None}

        if len(faces) == 0:
            return PostureMetrics(posture_state="sin_datos", posture_score=50.0, confidence=0.1), debug

        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        debug["posture_bbox"] = (int(x), int(y), int(x + w), int(y + h))

        frame_height, frame_width = frame.shape[:2]
        face_center_x = (x + (w / 2.0)) / max(frame_width, 1)
        face_center_y = (y + (h / 2.0)) / max(frame_height, 1)
        head_offset = abs(face_center_x - 0.5)
        vertical_offset = abs(face_center_y - 0.35)
        face_ratio = h / max(frame_height, 1)
        distance_penalty = abs(face_ratio - 0.28) * 180.0
        score = max(0.0, 100.0 - (head_offset * 140.0) - (vertical_offset * 100.0) - distance_penalty)

        if score >= 75.0:
            posture_state = "correcta"
        elif score >= 50.0:
            posture_state = "mejorable"
        else:
            posture_state = "encorvada"

        return PostureMetrics(
            posture_state=posture_state,
            posture_score=score,
            shoulder_tilt=None,
            torso_lean=vertical_offset,
            head_offset=head_offset,
            confidence=round(min(0.75, max(0.35, score / 100.0)), 4),
        ), debug
