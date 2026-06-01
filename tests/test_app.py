from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from app import (
    _build_alert_signature,
    _evaluate_alert,
    _register_human_label,
    _rule_label,
    _save_session_note,
)
from focustrack.models import (
    AttentionMetrics,
    ObjectMetrics,
    PostureMetrics,
    ProductivitySnapshot,
    ScreenMetrics,
)
from focustrack.monitoring.storage import StorageManager


class AppAlertTests(unittest.TestCase):
    def _snapshot(self, score: float, label: str) -> ProductivitySnapshot:
        return ProductivitySnapshot(
            session_id="session-demo",
            attention=AttentionMetrics(attention_state="atento"),
            posture=PostureMetrics(posture_state="correcta", posture_score=90.0, confidence=0.9),
            objects=ObjectMetrics(person_present=True, confidence=0.6),
            screen=ScreenMetrics(active_app="Code", category="trabajo", productivity_score=90.0),
            productivity_score=score,
            productivity_label=label,
        )

    def test_evaluate_alert_returns_low_rule(self) -> None:
        result = _evaluate_alert(
            self._snapshot(30.0, "Distraido"),
            {
                "productivity_low": {"enabled": True, "threshold": 45.0, "severity": "error"},
                "productivity_medium": {"enabled": True, "threshold": 75.0, "severity": "warning"},
            },
        )
        self.assertEqual(result["rule_key"], "productivity_low")
        self.assertEqual(result["severity"], "error")

    def test_evaluate_alert_returns_medium_rule(self) -> None:
        result = _evaluate_alert(
            self._snapshot(60.0, "Regular"),
            {
                "productivity_low": {"enabled": True, "threshold": 45.0, "severity": "error"},
                "productivity_medium": {"enabled": True, "threshold": 75.0, "severity": "warning"},
            },
        )
        self.assertEqual(result["rule_key"], "productivity_medium")
        self.assertEqual(result["severity"], "warning")

    def test_evaluate_alert_returns_success_when_score_is_healthy(self) -> None:
        result = _evaluate_alert(
            self._snapshot(88.0, "Productivo"),
            {
                "productivity_low": {"enabled": True, "threshold": 45.0, "severity": "error"},
                "productivity_medium": {"enabled": True, "threshold": 75.0, "severity": "warning"},
            },
        )
        self.assertEqual(result["rule_key"], "success")
        self.assertEqual(result["severity"], "success")

    def test_alert_signature_is_stable(self) -> None:
        alert_result = {
            "rule_key": "productivity_low",
            "severity": "error",
            "score": 42.0,
            "message": "Score por debajo del umbral critico (42.0).",
        }
        self.assertEqual(
            _build_alert_signature(alert_result, "session-demo"),
            "session-demo|productivity_low|error|42.0",
        )

    def test_rule_label_is_human_readable(self) -> None:
        self.assertEqual(_rule_label("productivity_low"), "Productividad baja")

    def test_save_session_note_registers_audit_event(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = StorageManager(Path(temp_dir))
            _save_session_note(
                storage=storage,
                session_id="session-demo",
                name="Sesion demo",
                description="Actualizada desde UI",
                approved_for_training=True,
                status="en_revision",
            )
            notes = storage.load_session_notes(session_id="session-demo")
            events = storage.load_audit_events(limit=10, session_id="session-demo")
            self.assertEqual(len(notes), 1)
            self.assertEqual(notes.iloc[0]["status"], "en_revision")
            self.assertEqual(events.iloc[-1]["event_type"], "session_note_updated")

    def test_register_human_label_registers_audit_event(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = StorageManager(Path(temp_dir))
            _register_human_label(
                storage=storage,
                session_id="session-demo",
                label="Productivo",
                start_time="2026-06-01T10:00:00",
                end_time="2026-06-01T10:05:00",
                notes="Revision manual",
            )
            labels = storage.load_human_labels(session_id="session-demo")
            events = storage.load_audit_events(limit=10, session_id="session-demo")
            self.assertEqual(len(labels), 1)
            self.assertEqual(labels.iloc[0]["label"], "Productivo")
            self.assertEqual(events.iloc[-1]["event_type"], "human_label_registered")


if __name__ == "__main__":
    unittest.main()
