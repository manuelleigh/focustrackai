from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from focustrack.models import ProductivitySnapshot

DEFAULT_ALERT_RULES: tuple[dict[str, Any], ...] = (
    {
        "rule_key": "productivity_low",
        "enabled": True,
        "threshold": 45.0,
        "window_seconds": 0.0,
        "severity": "warning",
    },
    {
        "rule_key": "productivity_medium",
        "enabled": True,
        "threshold": 75.0,
        "window_seconds": 0.0,
        "severity": "info",
    },
)


class StorageManager:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.data_dir / "productivity_history.csv"
        self.db_path = self.data_dir / "focustrack.db"
        self._ensure_schema()
        self.ensure_default_alert_rules()

    def append_snapshot(self, snapshot: ProductivitySnapshot) -> None:
        row = snapshot.to_row()
        self._append_snapshot_sqlite(row)
        self._append_snapshot_csv(row)

    def append_history_row(self, row: dict[str, object]) -> None:
        self._append_snapshot_sqlite(row)
        self._append_snapshot_csv(row)

    def load_history(self, limit: int | None = None) -> pd.DataFrame:
        sqlite_history = self._load_history_sqlite(limit)
        if not sqlite_history.empty:
            return sqlite_history

        if not self.csv_path.exists():
            return pd.DataFrame()

        history = pd.read_csv(self.csv_path)
        return self._normalize_history_frame(history, limit)

    def load_session_summaries(self, limit: int = 20) -> pd.DataFrame:
        query = """
            select
                s.session_id as session_id,
                min(s.timestamp) as started_at,
                max(s.timestamp) as last_seen_at,
                count(*) as snapshot_count,
                avg(s.productivity_score) as avg_productivity_score,
                max(sn.name) as session_name,
                max(sn.status) as session_status,
                max(sn.approved_for_training) as approved_for_training
            from snapshots s
            left join session_notes sn on sn.session_id = s.session_id
            group by s.session_id
            order by max(s.timestamp) desc
            limit ?
        """
        with self._connect() as connection:
            rows = connection.execute(query, (limit,)).fetchall()

        frame = pd.DataFrame(
            rows,
            columns=[
                "session_id",
                "started_at",
                "last_seen_at",
                "snapshot_count",
                "avg_productivity_score",
                "session_name",
                "session_status",
                "approved_for_training",
            ],
        )
        if not frame.empty:
            frame["started_at"] = pd.to_datetime(frame["started_at"], errors="coerce")
            frame["last_seen_at"] = pd.to_datetime(frame["last_seen_at"], errors="coerce")
            frame["snapshot_count"] = pd.to_numeric(frame["snapshot_count"], errors="coerce").fillna(0).astype(int)
            frame["avg_productivity_score"] = pd.to_numeric(
                frame["avg_productivity_score"], errors="coerce"
            )
            frame["approved_for_training"] = frame["approved_for_training"].fillna(0).astype(bool)
            frame["session_name"] = frame["session_name"].fillna("")
            frame["session_status"] = frame["session_status"].fillna("")
        return frame

    def append_audit_event(
        self,
        event_type: str,
        details: dict[str, object] | None = None,
        session_id: str | None = None,
    ) -> None:
        payload = details or {}
        with self._connect() as connection:
            connection.execute(
                """
                insert into audit_events(timestamp, event_type, session_id, details_json)
                values (?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    event_type,
                    session_id,
                    json.dumps(payload, ensure_ascii=False, default=str),
                ),
            )

    def load_audit_events(self, limit: int | None = 200) -> pd.DataFrame:
        query = """
            select timestamp, event_type, session_id, details_json
            from audit_events
            order by id desc
        """
        params: tuple[object, ...] = ()
        if limit is not None:
            query += " limit ?"
            params = (limit,)

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()

        rows = list(reversed(rows))
        events = [
            {
                "timestamp": row["timestamp"],
                "event_type": row["event_type"],
                "session_id": row["session_id"],
                "details": row["details_json"],
            }
            for row in rows
        ]
        frame = pd.DataFrame(events)
        if not frame.empty:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
            frame["details"] = frame["details"].apply(self._deserialize_json)
        return frame

    def storage_health(self) -> dict[str, object]:
        counts = {
            "snapshots": 0,
            "audit_events": 0,
            "human_labels": 0,
            "session_notes": 0,
            "alert_rules": 0,
        }
        with self._connect() as connection:
            for table_name in counts:
                counts[table_name] = int(
                    connection.execute(f"select count(*) from {table_name}").fetchone()[0]
                )

        return {
            "sqlite_path": str(self.db_path),
            "sqlite_exists": self.db_path.exists(),
            "csv_path": str(self.csv_path),
            "csv_exists": self.csv_path.exists(),
            **counts,
        }

    def append_human_label(
        self,
        session_id: str,
        label: str,
        start_time: str | None = None,
        end_time: str | None = None,
        notes: str = "",
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                insert into human_labels(created_at, session_id, start_time, end_time, label, notes)
                values (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.now().isoformat(),
                    session_id,
                    start_time,
                    end_time,
                    label,
                    notes,
                ),
            )

    def load_human_labels(self, session_id: str | None = None) -> pd.DataFrame:
        query = """
            select created_at, session_id, start_time, end_time, label, notes
            from human_labels
        """
        params: tuple[object, ...] = ()
        if session_id:
            query += " where session_id = ?"
            params = (session_id,)
        query += " order by id desc"

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()

        frame = pd.DataFrame(
            rows,
            columns=["created_at", "session_id", "start_time", "end_time", "label", "notes"],
        )
        for column in ["created_at", "start_time", "end_time"]:
            if column in frame.columns:
                frame[column] = pd.to_datetime(frame[column], errors="coerce")
        return frame

    def upsert_session_note(
        self,
        session_id: str,
        name: str,
        description: str = "",
        approved_for_training: bool = False,
        status: str = "registrada",
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                insert into session_notes(session_id, name, description, approved_for_training, status, updated_at)
                values (?, ?, ?, ?, ?, ?)
                on conflict(session_id) do update set
                    name=excluded.name,
                    description=excluded.description,
                    approved_for_training=excluded.approved_for_training,
                    status=excluded.status,
                    updated_at=excluded.updated_at
                """,
                (
                    session_id,
                    name,
                    description,
                    int(approved_for_training),
                    status,
                    datetime.now().isoformat(),
                ),
            )

    def load_session_notes(self, session_id: str | None = None) -> pd.DataFrame:
        query = """
            select session_id, name, description, approved_for_training, status, updated_at
            from session_notes
        """
        params: tuple[object, ...] = ()
        if session_id:
            query += " where session_id = ?"
            params = (session_id,)
        query += " order by updated_at desc"

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()

        frame = pd.DataFrame(
            rows,
            columns=[
                "session_id",
                "name",
                "description",
                "approved_for_training",
                "status",
                "updated_at",
            ],
        )
        if not frame.empty:
            frame["approved_for_training"] = frame["approved_for_training"].astype(bool)
            frame["updated_at"] = pd.to_datetime(frame["updated_at"], errors="coerce")
        return frame

    def upsert_alert_rule(
        self,
        rule_key: str,
        enabled: bool,
        threshold: float,
        window_seconds: float,
        severity: str,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                insert into alert_rules(rule_key, enabled, threshold, window_seconds, severity, updated_at)
                values (?, ?, ?, ?, ?, ?)
                on conflict(rule_key) do update set
                    enabled=excluded.enabled,
                    threshold=excluded.threshold,
                    window_seconds=excluded.window_seconds,
                    severity=excluded.severity,
                    updated_at=excluded.updated_at
                """,
                (
                    rule_key,
                    int(enabled),
                    threshold,
                    window_seconds,
                    severity,
                    datetime.now().isoformat(),
                ),
            )

    def load_alert_rules(self) -> pd.DataFrame:
        with self._connect() as connection:
            rows = connection.execute(
                """
                select rule_key, enabled, threshold, window_seconds, severity, updated_at
                from alert_rules
                order by rule_key
                """
            ).fetchall()

        frame = pd.DataFrame(
            rows,
            columns=[
                "rule_key",
                "enabled",
                "threshold",
                "window_seconds",
                "severity",
                "updated_at",
            ],
        )
        if not frame.empty:
            frame["enabled"] = frame["enabled"].astype(bool)
            frame["updated_at"] = pd.to_datetime(frame["updated_at"], errors="coerce")
        return frame

    def ensure_default_alert_rules(self) -> None:
        existing = self.load_alert_rules()
        existing_keys = set(existing["rule_key"]) if not existing.empty else set()
        for rule in DEFAULT_ALERT_RULES:
            if rule["rule_key"] not in existing_keys:
                self.upsert_alert_rule(
                    rule_key=str(rule["rule_key"]),
                    enabled=bool(rule["enabled"]),
                    threshold=float(rule["threshold"]),
                    window_seconds=float(rule["window_seconds"]),
                    severity=str(rule["severity"]),
                )

    def get_alert_rules_map(self) -> dict[str, dict[str, object]]:
        frame = self.load_alert_rules()
        if frame.empty:
            return {}
        rules: dict[str, dict[str, object]] = {}
        for row in frame.to_dict(orient="records"):
            rule_key = str(row.pop("rule_key"))
            rules[rule_key] = row
        return rules

    def _append_snapshot_csv(self, row: dict[str, object]) -> None:
        write_header = not self.csv_path.exists()
        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _append_snapshot_sqlite(self, row: dict[str, object]) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                insert into snapshots(
                    timestamp,
                    session_id,
                    productivity_label,
                    productivity_score,
                    attention_state,
                    posture_state,
                    object_state,
                    active_app,
                    screen_category,
                    payload_json
                )
                values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(row.get("timestamp", "")),
                    str(row.get("session_id", "")),
                    str(row.get("productivity_label", "")),
                    float(row.get("productivity_score", 0.0) or 0.0),
                    str(row.get("attention_state", "")),
                    str(row.get("posture_state", "")),
                    str(row.get("object_state", "")),
                    str(row.get("active_app", "")),
                    str(row.get("screen_category", "")),
                    json.dumps(row, ensure_ascii=False, default=str),
                ),
            )

    def _load_history_sqlite(self, limit: int | None = None) -> pd.DataFrame:
        query = """
            select timestamp, session_id, productivity_score, productivity_label,
                   attention_state, posture_state, object_state, active_app,
                   screen_category, payload_json
            from snapshots
            order by id desc
        """
        params: tuple[object, ...] = ()
        if limit is not None:
            query += " limit ?"
            params = (limit,)

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()

        if not rows:
            return pd.DataFrame()

        rows = list(reversed(rows))
        records: list[dict[str, object]] = []
        for row in rows:
            payload = self._deserialize_json(row["payload_json"])
            if isinstance(payload, dict):
                record = payload
            else:
                record = {}
            record.update(
                {
                    "timestamp": row["timestamp"],
                    "session_id": row["session_id"],
                    "productivity_score": row["productivity_score"],
                    "productivity_label": row["productivity_label"],
                    "attention_state": row["attention_state"],
                    "posture_state": row["posture_state"],
                    "object_state": row["object_state"],
                    "active_app": row["active_app"],
                    "screen_category": row["screen_category"],
                }
            )
            records.append(record)

        history = pd.DataFrame(records)
        return self._normalize_history_frame(history, limit=None)

    def _normalize_history_frame(
        self, history: pd.DataFrame, limit: int | None = None
    ) -> pd.DataFrame:
        if history.empty:
            return history

        if "timestamp" in history.columns:
            history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce")

        numeric_columns = [
            "productivity_score",
            "attention_component",
            "object_component",
            "posture_component",
            "screen_component",
            "avg_ear",
            "fatigue_score",
            "posture_score",
            "shoulder_tilt",
            "torso_lean",
            "head_offset",
            "screen_productivity_score",
        ]
        for column in numeric_columns:
            if column in history.columns:
                history[column] = pd.to_numeric(history[column], errors="coerce")

        if limit is not None and not history.empty:
            history = history.tail(limit)
        return history

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        connection.execute("pragma journal_mode=wal")
        connection.execute("pragma synchronous=normal")
        return connection

    def _ensure_schema(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                create table if not exists snapshots (
                    id integer primary key autoincrement,
                    timestamp text not null,
                    session_id text not null,
                    productivity_label text not null,
                    productivity_score real not null,
                    attention_state text not null default '',
                    posture_state text not null default '',
                    object_state text not null default '',
                    active_app text not null default '',
                    screen_category text not null default '',
                    payload_json text not null
                )
                """
            )
            connection.execute(
                """
                create index if not exists idx_snapshots_session_timestamp
                on snapshots(session_id, timestamp)
                """
            )
            self._ensure_column(connection, "snapshots", "attention_state", "text not null default ''")
            self._ensure_column(connection, "snapshots", "posture_state", "text not null default ''")
            self._ensure_column(connection, "snapshots", "object_state", "text not null default ''")
            self._ensure_column(connection, "snapshots", "active_app", "text not null default ''")
            self._ensure_column(connection, "snapshots", "screen_category", "text not null default ''")

            connection.execute(
                """
                create table if not exists audit_events (
                    id integer primary key autoincrement,
                    timestamp text not null,
                    event_type text not null,
                    session_id text,
                    details_json text not null
                )
                """
            )
            connection.execute(
                """
                create index if not exists idx_audit_session_timestamp
                on audit_events(session_id, timestamp)
                """
            )
            connection.execute(
                """
                create table if not exists human_labels (
                    id integer primary key autoincrement,
                    created_at text not null,
                    session_id text not null,
                    start_time text,
                    end_time text,
                    label text not null,
                    notes text not null default ''
                )
                """
            )
            connection.execute(
                """
                create index if not exists idx_human_labels_session
                on human_labels(session_id, start_time, end_time)
                """
            )
            connection.execute(
                """
                create table if not exists session_notes (
                    session_id text primary key,
                    name text not null,
                    description text not null default '',
                    approved_for_training integer not null default 0,
                    status text not null default 'registrada',
                    updated_at text not null
                )
                """
            )
            connection.execute(
                """
                create table if not exists alert_rules (
                    id integer primary key autoincrement,
                    rule_key text not null unique,
                    enabled integer not null,
                    threshold real not null,
                    window_seconds real not null,
                    severity text not null,
                    updated_at text not null
                )
                """
            )

    def _ensure_column(
        self, connection: sqlite3.Connection, table_name: str, column_name: str, column_sql: str
    ) -> None:
        columns = {
            row["name"]
            for row in connection.execute(f"pragma table_info({table_name})").fetchall()
        }
        if column_name not in columns:
            connection.execute(
                f"alter table {table_name} add column {column_name} {column_sql}"
            )

    def _deserialize_json(self, value: object) -> object:
        if isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError:
                return value
        return value
