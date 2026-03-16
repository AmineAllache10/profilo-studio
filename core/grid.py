# core/grid.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd


def _robust_step(vals: np.ndarray) -> float:
    """
    Estime un pas typique (dx ou dy) de manière robuste via la médiane
    des différences entre valeurs uniques triées.
    """
    u = np.unique(np.sort(vals))
    if u.size < 2:
        return 0.0
    diffs = np.diff(u)
    diffs = diffs[np.isfinite(diffs)]
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return 0.0
    return float(np.median(diffs))


def _round_to_step(v: np.ndarray, step: float) -> np.ndarray:
    """
    Arrondit v sur une grille de pas 'step' pour stabiliser les xyz bruités.
    """
    if step <= 0:
        return v
    return np.round(v / step) * step


@dataclass
class GridResult:
    is_grid: bool
    nx: int
    ny: int
    x_vals: np.ndarray
    y_vals: np.ndarray
    Z: np.ndarray
    missing_mask: np.ndarray
    missing_rate: float


def to_grid(df: pd.DataFrame) -> GridResult:
    """
    Convertit un DataFrame (x,y,z) en matrice Z sur une grille régulière (y en lignes, x en colonnes).
    - calcule un pas robuste dx/dy
    - arrondit x,y sur ce pas
    - pivot_table pour reconstruire Z (moyenne si doublons)
    - missing_mask = NaN dans Z
    """
    x = df["x"].to_numpy()
    y = df["y"].to_numpy()
    z = df["z"].to_numpy()

    dx = _robust_step(x)
    dy = _robust_step(y)

    # Arrondi sur pas typique
    xr = _round_to_step(x, dx if dx > 0 else 0.0)
    yr = _round_to_step(y, dy if dy > 0 else 0.0)

    x_u = np.unique(xr)
    y_u = np.unique(yr)

    nx, ny = int(x_u.size), int(y_u.size)
    expected = nx * ny
    present = int(df.shape[0])

    # Heuristique: "grille" si dimensions suffisantes et couverture correcte
    is_grid = (nx >= 5 and ny >= 5 and present >= 0.5 * expected)

    tmp = pd.DataFrame({"x": xr, "y": yr, "z": z})
    piv = tmp.pivot_table(index="y", columns="x", values="z", aggfunc="mean")
    piv = piv.reindex(index=np.sort(y_u), columns=np.sort(x_u))

    Z = piv.to_numpy(dtype=float)
    missing_mask = np.isnan(Z)
    missing_rate = float(np.mean(missing_mask)) if Z.size > 0 else 1.0

    return GridResult(
        is_grid=bool(is_grid),
        nx=nx,
        ny=ny,
        x_vals=piv.columns.to_numpy(),
        y_vals=piv.index.to_numpy(),
        Z=Z,
        missing_mask=missing_mask,
        missing_rate=missing_rate,
    )