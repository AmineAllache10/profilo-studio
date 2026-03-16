# core/io_xyz.py
from __future__ import annotations

import re
import numpy as np
import pandas as pd


_num_re = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def _extract_floats_from_line(line: str) -> list[float]:
    """
    Extrait tous les nombres flottants présents dans une ligne (robuste aux séparateurs / texte).
    """
    tokens = _num_re.findall(line)
    if not tokens:
        return []
    out: list[float] = []
    for t in tokens:
        try:
            out.append(float(t))
        except Exception:
            pass
    return out


def read_xyz_points(path: str) -> pd.DataFrame:
    """
    Lit un .xyz de manière tolérante:
    - ignore les lignes sans nombres
    - accepte un nombre de colonnes variable
    - prend les 3 premiers nombres de la ligne comme (x,y,z)

    IMPORTANT: aucune limite de points (FULL LOAD).
    """
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            vals = _extract_floats_from_line(line)
            if len(vals) >= 3:
                xs.append(vals[0])
                ys.append(vals[1])
                zs.append(vals[2])

    if len(xs) == 0:
        raise ValueError("Aucun triplet numérique (x,y,z) détecté dans le fichier.")

    return pd.DataFrame(
        {
            "x": np.array(xs, dtype=float),
            "y": np.array(ys, dtype=float),
            "z": np.array(zs, dtype=float),
        }
    )