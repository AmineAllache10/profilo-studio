# core/missing.py
from __future__ import annotations

import numpy as np
import pandas as pd


def fill_missing_simple(Z: np.ndarray) -> np.ndarray:
    """
    Fallback sans SciPy:
    - interpolation 2D par passes (axis=0 puis axis=1)
    - puis ffill/bfill pour finir de combler
    """
    df = pd.DataFrame(Z)
    df = df.interpolate(axis=0, limit_direction="both")
    df = df.interpolate(axis=1, limit_direction="both")
    df = df.ffill(axis=0).bfill(axis=0)
    df = df.ffill(axis=1).bfill(axis=1)
    return df.to_numpy(dtype=float)


def fill_missing_nearest(Z: np.ndarray) -> np.ndarray:
    """
    Remplissage par plus proche voisin (si SciPy dispo).
    Sinon fallback sur fill_missing_simple.
    """
    try:
        from scipy.ndimage import distance_transform_edt

        Z2 = Z.copy()
        mask = np.isnan(Z2)
        if not np.any(mask):
            return Z2

        # indices: array shape (ndim, ny, nx)
        indices = distance_transform_edt(mask, return_distances=False, return_indices=True)
        Z2[mask] = Z2[tuple(indices[:, mask])]
        return Z2
    except Exception:
        return fill_missing_simple(Z)


def fill_missing(Z: np.ndarray, method: str) -> np.ndarray:
    """
    method:
    - "Nearest"
    - "Interpolate"
    - autre -> copie
    """
    if method == "Nearest":
        return fill_missing_nearest(Z)
    if method == "Interpolate":
        return fill_missing_simple(Z)
    return Z.copy()