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
