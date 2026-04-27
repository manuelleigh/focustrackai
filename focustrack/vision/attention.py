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
HAS_MEDIAPIPE_SOLUTIONS = hasattr(mp, "solutions")
FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


class AttentionAnalyzer:
    def __init__(
        self,
        thresholds: DetectionThresholds,
        enable_dlib: bool = False,
        dlib_shape_predictor: Path | None = None,
    ):
        self.thresholds = thresholds
        self.use_mediapipe = HAS_MEDIAPIPE_SOLUTIONS
        self.face_mesh = None
        if self.use_mediapipe:
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
        if not self.use_mediapipe or self.face_mesh is None:
            return self._analyze_without_mediapipe(frame)

        result = self.face_mesh.process(rgb_frame)
        backend = "mediapipe+dlib" if self.use_dlib else "mediapipe"
        debug: dict[str, object] = {
            "face_landmarks": None,
            "face_bbox": None,
        }

        if not result.multi_face_landmarks:
            face_detected = (
                self._dlib_face_detected(frame)
                if self.dlib_detector is not None
                else False
            )
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

        face_landmarks = result.multi_face_landmarks[0]
        debug["face_landmarks"] = face_landmarks
        face_bbox = self._face_bbox(
            face_landmarks.landmark, frame.shape[1], frame.shape[0]
        )
        debug["face_bbox"] = face_bbox

        left_eye_points = self._pixels(
            face_landmarks.landmark, LEFT_EYE, frame.shape[1], frame.shape[0]
        )
        right_eye_points = self._pixels(
            face_landmarks.landmark, RIGHT_EYE, frame.shape[1], frame.shape[0]
        )
        left_ear = self._eye_aspect_ratio(left_eye_points)
        right_ear = self._eye_aspect_ratio(right_eye_points)
        avg_ear = float(np.mean([left_ear, right_ear]))

        eyes_closed = avg_ear < self.thresholds.ear_closed
        if eyes_closed:
            self.closed_eye_frames += 1
            if not self.previous_eyes_closed:
                self.blink_count += 1
        else:
            self.closed_eye_frames = 0
        self.previous_eyes_closed = eyes_closed

        gaze_ratio = self._gaze_ratio(face_landmarks.landmark)
        gaze_direction = self._gaze_direction(gaze_ratio)
        fatigue_score = min(
            1.0, self.closed_eye_frames / max(1, self.thresholds.fatigue_frame_window)
        )

        if eyes_closed and fatigue_score >= 0.75:
            attention_state = "somnoliento"
        elif gaze_direction == "centro":
            attention_state = "atento"
        else:
            attention_state = "desviado"

        metrics = AttentionMetrics(
            face_detected=True,
            eyes_detected=True,
            eyes_closed=eyes_closed,
            attention_state=attention_state,
            gaze_direction=gaze_direction,
            left_ear=left_ear,
            right_ear=right_ear,
            avg_ear=avg_ear,
            gaze_ratio=gaze_ratio,
            fatigue_score=fatigue_score,
            blink_count=self.blink_count,
            backend=backend,
        )
        return metrics, debug

    def _analyze_without_mediapipe(
        self, frame: np.ndarray
    ) -> tuple[AttentionMetrics, dict[str, object]]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = FACE_CASCADE.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        debug: dict[str, object] = {
            "face_landmarks": None,
            "face_bbox": None,
        }

        if len(faces) == 0:
            self.closed_eye_frames = 0
            self.previous_eyes_closed = False
            return (
                AttentionMetrics(
                    face_detected=False,
                    eyes_detected=False,
                    eyes_closed=False,
                    attention_state="ausente",
                    gaze_direction="desconocida",
                    fatigue_score=0.0,
                    blink_count=self.blink_count,
                    backend="fallback-haar",
                ),
                debug,
            )

        x, y, w, h = max(faces, key=lambda item: item[2] * item[3])
        debug["face_bbox"] = (x, y, x + w, y + h)
        return (
            AttentionMetrics(
                face_detected=True,
                eyes_detected=False,
                eyes_closed=False,
                attention_state="atento",
                gaze_direction="desconocida",
                fatigue_score=0.0,
                blink_count=self.blink_count,
                backend="fallback-haar",
            ),
            debug,
        )

    def close(self) -> None:
        if self.face_mesh is not None:
            self.face_mesh.close()

    def _dlib_face_detected(self, frame: np.ndarray) -> bool:
        if self.dlib_detector is None:
            return False

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.dlib_detector(gray, 0)
        return bool(faces)

    def _face_bbox(
        self, landmarks, frame_width: int, frame_height: int
    ) -> tuple[int, int, int, int]:
        xs = []
        ys = []
        for index in FACE_BBOX_REFERENCE:
            landmark = landmarks[index]
            xs.append(int(landmark.x * frame_width))
            ys.append(int(landmark.y * frame_height))

        x1 = max(0, min(xs))
        y1 = max(0, min(ys))
        x2 = min(frame_width - 1, max(xs))
        y2 = min(frame_height - 1, max(ys))
        return x1, y1, x2, y2

    def _pixels(
        self, landmarks, indexes: list[int], frame_width: int, frame_height: int
    ) -> np.ndarray:
        return np.array(
            [
                (
                    int(landmarks[index].x * frame_width),
                    int(landmarks[index].y * frame_height),
                )
                for index in indexes
            ],
            dtype=np.float32,
        )

    def _eye_aspect_ratio(self, points: np.ndarray) -> float:
        vertical_1 = np.linalg.norm(points[1] - points[5])
        vertical_2 = np.linalg.norm(points[2] - points[4])
        horizontal = np.linalg.norm(points[0] - points[3])
        return float((vertical_1 + vertical_2) / (2.0 * horizontal + 1e-6))

    def _gaze_ratio(self, landmarks) -> float:
        left_ratio = self._single_eye_gaze_ratio(
            landmarks, LEFT_IRIS, LEFT_EYE[0], LEFT_EYE[3]
        )
        right_ratio = self._single_eye_gaze_ratio(
            landmarks, RIGHT_IRIS, RIGHT_EYE[0], RIGHT_EYE[3]
        )
        return float(np.mean([left_ratio, right_ratio]))

    def _single_eye_gaze_ratio(
        self,
        landmarks,
        iris_indices: list[int],
        eye_left_index: int,
        eye_right_index: int,
    ) -> float:
        iris_points = np.array(
            [[landmarks[index].x, landmarks[index].y] for index in iris_indices],
            dtype=np.float32,
        )
        iris_center = iris_points.mean(axis=0)
        eye_left = landmarks[eye_left_index]
        eye_right = landmarks[eye_right_index]
        min_x = min(eye_left.x, eye_right.x)
        max_x = max(eye_left.x, eye_right.x)
        return float((iris_center[0] - min_x) / (max_x - min_x + 1e-6))

    def _gaze_direction(self, gaze_ratio: float) -> str:
        if (
            self.thresholds.gaze_center_min
            <= gaze_ratio
            <= self.thresholds.gaze_center_max
        ):
            return "centro"
        if gaze_ratio < self.thresholds.gaze_center_min:
            return "izquierda"
        return "derecha"
