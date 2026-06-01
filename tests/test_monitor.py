from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from focustrack.config import FocusTrackConfig
from focustrack.models import (
    AttentionMetrics,
    ObjectMetrics,
    PostureMetrics,
    ProductivitySnapshot,
    ScreenMetrics,
)
from focustrack.monitor import FocusTrackMonitor


class MonitorTests(unittest.TestCase):
    def test_backend_summary_reflects_snapshot_backends(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = FocusTrackConfig(data_dir=Path(temp_dir))
            monitor = FocusTrackMonitor(config=config)
            snapshot = ProductivitySnapshot(
                session_id="session-demo",
                attention=AttentionMetrics(attention_state="atento", backend="fallback-haar"),
                posture=PostureMetrics(posture_state="correcta", posture_score=90.0, confidence=0.3),
                objects=ObjectMetrics(person_present=True, backend="yolo+hands", confidence=0.7),
                screen=ScreenMetrics(active_app="Code", category="trabajo", productivity_score=90.0),
            )
            summary = monitor._backend_summary(snapshot)
            self.assertEqual(summary, "A:fallback-haar | P:opencv | O:yolo+hands")
            monitor.stop()

    def test_snapshot_row_includes_backend_columns(self) -> None:
        snapshot = ProductivitySnapshot(
            session_id="session-demo",
            attention=AttentionMetrics(attention_state="atento", backend="mediapipe"),
            posture=PostureMetrics(posture_state="correcta", posture_score=90.0, confidence=0.9),
            objects=ObjectMetrics(person_present=True, backend="hands", confidence=0.7),
            screen=ScreenMetrics(active_app="Code", category="trabajo", productivity_score=90.0),
        )
        row = snapshot.to_row()
        self.assertEqual(row["attention_backend"], "mediapipe")
        self.assertEqual(row["object_backend"], "hands")


if __name__ == "__main__":
    unittest.main()
