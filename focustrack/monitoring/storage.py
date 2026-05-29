from __future__ import annotations

import csv
import json
import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

from focustrack.models import ProductivitySnapshot

class StorageManager:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.data_dir / "productivity_history.csv"
        self.db_path = self.data_dir / "focustrack.db"
        self._ensure_schema()

    def append_snapshot(self, snapshot: ProductivitySnapshot) -> None:
        row = snapshot.to_row()
        self.append_history_row(row)
        write_header = not self.csv_path.exists()

        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def load_history(self, limit: int | None = None) -> pd.DataFrame:
        sqlite_history = self._load_history_sqlite(limit)
        if not sqlite_history.empty:
            return sqlite_history

        if not self.csv_path.exists():
            return pd.DataFrame()

        history = pd.read_csv(self.csv_path)
        if "timestamp" in history.columns:
            history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce")

        if limit is not None and not history.empty:
            history = history.tail(limit)

        return history

    def append_audit_event(self, event_type: str, details: dict[str, object] | None = None, session_id: str | None = None) -> None:
        payload = details or {}
        with self._connect() as connection:
            connection.execute(
                """
                insert into audit_events(timestamp, event_type, session_id, details_json)
                values (?, ?, ?, ?)
                """,
                (datetime.now().isoformat(), event_type, session_id, json.dumps(payload, ensure_ascii=False, default=str)),
            )
    
    def load_audit_events(self, limit: int | None = 200) -> pd.DataFrame:
        query = "select timestamp, event_type, session_id, details_json from audit_events order by id"
        params: tuple[object, ...] = ()
        if limit is not None:
            query += " desc limit ?"
            params = (limit,)

        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()

        if limit is not None:
            rows = list(reversed(rows))

        events = [
            {
                "timestamp": row[0],
                "event_type": row[1],
                "session_id": row[2],
                "details": row[3],
            }
            for row in rows
        ]
        frame = pd.DataFrame(events)
        if "timestamp" in frame.columns:
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        return frame
    
     def storage_health(self) -> dict[str, object]:
        snapshot_count = 0
        audit_count = 0
        with self._connect() as connection:
            snapshot_count = int(connection.execute("select count(*) from snapshots").fetchone()[0])
            audit_count = int(connection.execute("select count(*) from audit_events").fetchone()[0])

        return {
            "sqlite_path": str(self.db_path),
            "sqlite_exists": self.db_path.exists(),
            "csv_path": str(self.csv_path),
            "csv_exists": self.csv_path.exists(),
            "snapshots": snapshot_count,
            "audit_events": audit_count,
        }

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
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