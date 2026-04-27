from __future__ import annotations

import uuid

import cv2
import mediapipe as mp
import numpy as np

from focustrack.config import FocusTrackConfig
from focustrack.engine.scoring import evaluate_productivity
from focustrack.models import ProductivitySnapshot
from focustrack.monitoring.screen import ScreenActivityMonitor
from focustrack.monitoring.storage import StorageManager
from focustrack.vision.attention import AttentionAnalyzer
from focustrack.vision.objects import ObjectAnalyzer
from focustrack.vision.posture import PostureAnalyzer


class FocusTrackMonitor:
    def __init__(self, config: FocusTrackConfig, camera_index: int = 0):
        self.config = config
        self.camera_index = camera_index
        self.storage = StorageManager(config.data_dir)
        self.attention = AttentionAnalyzer(
            thresholds=config.thresholds,
            enable_dlib=config.models.enable_dlib,
            dlib_shape_predictor=config.models.dlib_shape_predictor,
        )
        self.posture = PostureAnalyzer(config.thresholds)
        self.objects = ObjectAnalyzer(config.thresholds, config.models)
        self.screen = ScreenActivityMonitor(config)
        self.capture: cv2.VideoCapture | None = None
        self.frame_number = 0
        self.session_id = uuid.uuid4().hex[:8]
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands

    def start(self) -> None:
        if self.capture is not None and self.capture.isOpened():
            return

        self.capture = cv2.VideoCapture(self.camera_index)
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.camera_width)
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.camera_height)

        if not self.capture.isOpened():
            raise RuntimeError("No se pudo abrir la camara seleccionada.")

    def stop(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.attention.close()
        self.posture.close()
        self.objects.close()

    def process_next(self) -> tuple[ProductivitySnapshot, np.ndarray]:
        self.start()
        assert self.capture is not None

        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise RuntimeError("No se pudo capturar un frame desde la camara.")

        self.frame_number += 1
        frame = cv2.flip(frame, 1)

        attention_metrics, attention_debug = self.attention.analyze(frame)
        posture_metrics, posture_debug = self.posture.analyze(frame)
        object_metrics, object_debug = self.objects.analyze(
            frame,
            face_bbox=attention_debug.get("face_bbox"),
            frame_number=self.frame_number,
        )
        screen_metrics = self.screen.sample()

        snapshot = evaluate_productivity(
            session_id=self.session_id,
            attention=attention_metrics,
            posture=posture_metrics,
            objects=object_metrics,
            screen=screen_metrics,
            weights=self.config.weights,
        )
        self.storage.append_snapshot(snapshot)

        annotated = self._annotate_frame(
            frame=frame.copy(),
            snapshot=snapshot,
            attention_debug=attention_debug,
            posture_debug=posture_debug,
            object_debug=object_debug,
        )
        return snapshot, annotated

    def _annotate_frame(
        self,
        frame: np.ndarray,
        snapshot: ProductivitySnapshot,
        attention_debug: dict[str, object],
        posture_debug: dict[str, object],
        object_debug: dict[str, object],
    ) -> np.ndarray:
        face_landmarks = attention_debug.get("face_landmarks")
        if face_landmarks is not None:
            self.mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(80, 180, 255), thickness=1, circle_radius=1
                ),
            )

        pose_landmarks = posture_debug.get("pose_landmarks")
        if pose_landmarks is not None:
            self.mp_drawing.draw_landmarks(
                frame,
                pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(0, 255, 170), thickness=2, circle_radius=2
                ),
                connection_drawing_spec=self.mp_drawing.DrawingSpec(
                    color=(255, 255, 255), thickness=2
                ),
            )

        hand_landmarks = object_debug.get("hand_landmarks")
        if hand_landmarks:
            for hand in hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand,
                    self.mp_hands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(255, 204, 0), thickness=2, circle_radius=2
                    ),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(
                        color=(255, 255, 255), thickness=1
                    ),
                )

        face_bbox = attention_debug.get("face_bbox")
        if face_bbox:
            x1, y1, x2, y2 = face_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 255), 2)

        for (x1, y1, x2, y2), label, confidence in object_debug.get("yolo_boxes", []):
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 130, 80), 2)
            cv2.putText(
                frame,
                f"{label} {confidence:.2f}",
                (x1, max(18, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 130, 80),
                2,
            )

        header_color = self._score_color(snapshot.productivity_score)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 120), (15, 15, 15), -1)
        cv2.putText(
            frame,
            f"Score: {snapshot.productivity_score:.1f}",
            (16, 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            header_color,
            2,
        )
        cv2.putText(
            frame,
            f"Estado: {snapshot.productivity_label}",
            (16, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Atencion: {snapshot.attention.attention_state} | Mirada: {snapshot.attention.gaze_direction}",
            (16, 88),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"Postura: {snapshot.posture.posture_state} | Objetos: {snapshot.objects.object_state}",
            (16, 112),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            frame,
            f"App activa: {snapshot.screen.active_app}",
            (16, frame.shape[0] - 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
        )
        return frame

    def _score_color(self, score: float) -> tuple[int, int, int]:
        if score >= 75.0:
            return (60, 210, 90)
        if score >= 45.0:
            return (0, 220, 255)
        return (70, 70, 255)
