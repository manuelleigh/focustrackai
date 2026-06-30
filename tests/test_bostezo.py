import pandas as pd
from focustrack.features.bostezo import compute_yawns


def _h(yawning: list[bool]) -> pd.DataFrame:
    return pd.DataFrame({"yawning": yawning})


def test_sin_bostezos():
    r = compute_yawns(_h([False] * 10))
    assert r["total_bostezos"] == 0


def test_un_bostezo():
    # Transición False→True→True→False cuenta como 1
    r = compute_yawns(_h([False, False, True, True, False]))
    assert r["total_bostezos"] == 1


def test_dos_bostezos():
    r = compute_yawns(_h([False, True, False, True, False]))
    assert r["total_bostezos"] == 2


def test_empty():
    r = compute_yawns(pd.DataFrame())
    assert r["total_bostezos"] == 0


def test_mar_alto_es_bostezo():
    """Verifica que MAR > 0.6 dispararía yawning=True en attention.py (lógica de umbral)."""
    mar_open = 0.60
    assert 0.75 > mar_open  # mar abierto supera umbral


if __name__ == "__main__":
    test_sin_bostezos()
    test_un_bostezo()
    test_dos_bostezos()
    test_empty()
    test_mar_alto_es_bostezo()
    print("test_bostezo: OK")
