from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from focustrack.models import (
    AttentionMetrics,
    ObjectMetrics,
    PostureMetrics,
    ProductivitySnapshot,
    ScreenMetrics,
)
from focustrack.monitoring.storage import StorageManager


class StorageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.storage = StorageManager(self.data_dir)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_storage_manager_initializes_schema(self) -> None:
        health = self.storage.storage_health()
        self.assertTrue(health["sqlite_exists"])
        self.assertEqual(health["snapshots"], 0)
        self.assertGreaterEqual(health["alert_rules"], 2)

    def test_snapshot_roundtrip_uses_sqlite(self) -> None:
        snapshot = ProductivitySnapshot(
            session_id="session-demo",
            attention=AttentionMetrics(attention_state="atento"),
            posture=PostureMetrics(
                posture_state="correcta",
                posture_score=88.0,
                confidence=0.91,
            ),
            objects=ObjectMetrics(
                person_present=True,
                object_state="sin_objetos",
                confidence=0.62,
            ),
            screen=ScreenMetrics(
                active_app="Code",
                category="trabajo",
                productivity_score=100.0,
            ),
            productivity_score=92.0,
            productivity_label="Productivo",
            attention_component=100.0,
            object_component=100.0,
            posture_component=88.0,
            screen_component=100.0,
        )
        self.storage.append_snapshot(snapshot)

        history = self.storage.load_history(limit=10)
        self.assertEqual(len(history), 1)
        self.assertEqual(history.iloc[0]["session_id"], "session-demo")
        self.assertEqual(history.iloc[0]["productivity_label"], "Productivo")
        self.assertEqual(history.iloc[0]["active_app"], "Code")
        self.assertTrue((self.data_dir / "productivity_history.csv").exists())

    def test_session_notes_and_human_labels_roundtrip(self) -> None:
        self.storage.upsert_session_note(
            session_id="session-demo",
            name="Sesion demo",
            description="Revision manual",
            approved_for_training=True,
            status="en_revision",
        )
        self.storage.append_human_label(
            session_id="session-demo",
            label="Productivo",
            start_time="2026-06-01T10:00:00",
            end_time="2026-06-01T10:05:00",
            notes="Coincide con el score",
        )

        notes = self.storage.load_session_notes(session_id="session-demo")
        labels = self.storage.load_human_labels(session_id="session-demo")

        self.assertEqual(len(notes), 1)
        self.assertEqual(notes.iloc[0]["name"], "Sesion demo")
        self.assertTrue(bool(notes.iloc[0]["approved_for_training"]))
        self.assertEqual(len(labels), 1)
        self.assertEqual(labels.iloc[0]["label"], "Productivo")

    def test_alert_rules_and_audit_events_roundtrip(self) -> None:
        self.storage.upsert_alert_rule(
            rule_key="productivity_low",
            enabled=True,
            threshold=40.0,
            window_seconds=30.0,
            severity="error",
        )
        self.storage.append_audit_event(
            "snapshot_recorded",
            {"productivity_score": 40.0},
            session_id="session-demo",
        )

        rules = self.storage.load_alert_rules()
        events = self.storage.load_audit_events(limit=10)

        low_rule = rules[rules["rule_key"] == "productivity_low"].iloc[0]
        self.assertEqual(float(low_rule["threshold"]), 40.0)
        self.assertEqual(str(low_rule["severity"]), "error")
        self.assertEqual(len(events), 1)
        self.assertEqual(events.iloc[0]["event_type"], "snapshot_recorded")
        self.assertEqual(events.iloc[0]["details"]["productivity_score"], 40.0)

    def test_load_session_summaries_aggregates_snapshots_and_notes(self) -> None:
        self.storage.upsert_session_note(
            session_id="session-1",
            name="Sesion principal",
            description="Consolidada",
            approved_for_training=True,
            status="finalizada",
        )
        first_snapshot = ProductivitySnapshot(
            session_id="session-1",
            attention=AttentionMetrics(attention_state="atento"),
            posture=PostureMetrics(posture_state="correcta", posture_score=80.0, confidence=0.8),
            objects=ObjectMetrics(person_present=True, confidence=0.6),
            screen=ScreenMetrics(active_app="Code", category="trabajo", productivity_score=90.0),
            productivity_score=80.0,
            productivity_label="Productivo",
        )
        second_snapshot = ProductivitySnapshot(
            session_id="session-1",
            attention=AttentionMetrics(attention_state="desviado"),
            posture=PostureMetrics(posture_state="mejorable", posture_score=60.0, confidence=0.5),
            objects=ObjectMetrics(person_present=True, confidence=0.4),
            screen=ScreenMetrics(active_app="Browser", category="neutral", productivity_score=60.0),
            productivity_score=60.0,
            productivity_label="Regular",
        )
        self.storage.append_snapshot(first_snapshot)
        self.storage.append_snapshot(second_snapshot)

        summaries = self.storage.load_session_summaries(limit=10)

        self.assertEqual(len(summaries), 1)
        summary = summaries.iloc[0]
        self.assertEqual(summary["session_id"], "session-1")
        self.assertEqual(summary["session_name"], "Sesion principal")
        self.assertEqual(summary["session_status"], "finalizada")
        self.assertEqual(int(summary["snapshot_count"]), 2)
        self.assertTrue(bool(summary["approved_for_training"]))
        self.assertAlmostEqual(float(summary["avg_productivity_score"]), 70.0, places=2)


if __name__ == "__main__":
    unittest.main()
