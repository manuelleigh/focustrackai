from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from focustrack.models import ProductivitySnapshot


class StorageManager:
    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.data_dir / "productivity_history.csv"

    def append_snapshot(self, snapshot: ProductivitySnapshot) -> None:
        row = snapshot.to_row()
        write_header = not self.csv_path.exists()

        with self.csv_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def load_history(self, limit: int | None = None) -> pd.DataFrame:
        if not self.csv_path.exists():
            return pd.DataFrame()

        history = pd.read_csv(self.csv_path)
        if "timestamp" in history.columns:
            history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce")

        if limit is not None and not history.empty:
            history = history.tail(limit)

        return history
