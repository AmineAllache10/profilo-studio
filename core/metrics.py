# core/metrics.py
from __future__ import annotations

import numpy as np


def compare_metrics(A: np.ndarray, B: np.ndarray) -> dict:
    """
    Compare deux matrices A et B (mêmes dimensions) avec :
    - nombre de points communs valides
    - corrélation de Pearson
    - RMSE
    - SSIM (si scikit-image est dispo)

    Les NaN sont ignorés (comparaison uniquement sur l'intersection finie).
    """
    mask = np.isfinite(A) & np.isfinite(B)

    out: dict = {}

    n_overlap = int(mask.sum())
    out["overlap_points"] = n_overlap

    if n_overlap < 10:
        out["corr"] = np.nan
        out["rmse"] = np.nan
        out["ssim"] = np.nan
        return out

    a = A[mask].ravel()
    b = B[mask].ravel()

    # Corrélation
    if a.size > 1:
        out["corr"] = float(np.corrcoef(a, b)[0, 1])
    else:
        out["corr"] = np.nan

    # RMSE
    out["rmse"] = float(np.sqrt(np.mean((a - b) ** 2)))

    # SSIM (optionnel)
    ssim_val = np.nan
    try:
        from skimage.metrics import structural_similarity as ssim

        A0 = np.where(np.isfinite(A), A, 0.0)
        B0 = np.where(np.isfinite(B), B, 0.0)

        data_min = min(A0.min(), B0.min())
        data_max = max(A0.max(), B0.max())
        data_range = float(data_max - data_min)

        if data_range > 0:
            ssim_val = float(ssim(A0, B0, data_range=data_range))
    except Exception:
        pass

    out["ssim"] = ssim_val
    return out