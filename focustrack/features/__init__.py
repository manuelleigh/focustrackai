from __future__ import annotations

import importlib
import pkgutil
from pathlib import Path
from types import ModuleType


def load_features() -> list[ModuleType]:
    """Descubre y retorna los módulos de features ordenados por su atributo ORDER."""
    package_dir = Path(__file__).parent
    modules = []
    for _, name, _ in pkgutil.iter_modules([str(package_dir)]):
        mod = importlib.import_module(f"focustrack.features.{name}")
        if hasattr(mod, "render") and hasattr(mod, "TITLE"):
            modules.append(mod)
    modules.sort(key=lambda m: getattr(m, "ORDER", 100))
    return modules
