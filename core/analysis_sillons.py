# core/analysis_sillons.py
from __future__ import annotations

import numpy as np


def _detrend_and_surface0(profil: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Comme ta binôme :
    - enlève la pente (polyfit degré 1)
    - remet le max à 0 (surface = 0)
    """
    px = np.arange(len(profil))
    coeffs = np.polyfit(px, profil, 1)
    trend = np.polyval(coeffs, px)
    profil_det = profil - trend
    profil_det -= np.nanmax(profil_det)
    return profil_det, trend


def _resolution_um_px(x_vals: np.ndarray | None, mask: np.ndarray, nx_after_mask: int) -> float:
    if x_vals is None:
        return float("nan")
    xv = np.asarray(x_vals, dtype=float)
    if xv.size < 2:
        return float("nan")
    # si x_vals correspond à l'axe X (taille nx) -> applique le même mask que le profil
    if mask is not None and xv.size == mask.size:
        xv = xv[mask]
    if xv.size < 2 or nx_after_mask < 2:
        return float("nan")
    return float((np.nanmax(xv) - np.nanmin(xv)) / (nx_after_mask - 1))


def analyse_sillons_from_grid(
    Z: np.ndarray,
    x_vals: np.ndarray | None = None,
    ref_surface_zero: bool = True,
    depth_percentile: float = 5.0,
    period_min_px: int = 20,   # IMPORTANT : borne basse (évite les énormes plateaux)
    period_max_px: int = 300,  # IMPORTANT : borne haute
    width_ratio: float = 0.5,  # largeur_sillon = ratio * période (0.5 = demi période)
) -> dict:
    """
    Méthode alignée binôme + FFT "band-limited" (plage de période) pour éviter les périodes débiles.
    Renvoie TOUJOURS les mêmes clés (celles que ton app affiche).
    """
    if Z is None or np.size(Z) == 0:
        raise ValueError("Z vide")

    Z2 = np.array(Z, dtype=float, copy=True)

    # 1) Référence surface=0 (comme elle)
    if ref_surface_zero:
        Z2 = Z2 - np.nanmax(Z2)

    # 2) Profil moyen en X (moyenne sur Y)
    profil = np.nanmean(Z2, axis=0)

    # enlève NaN
    mask = np.isfinite(profil)
    profil = profil[mask]
    if profil.size < 50:
        raise ValueError("Profil trop court / trop de NaN")

    # 3) résolution (µm/px)
    resolution_um_px = _resolution_um_px(x_vals, mask, profil.size)

    # 4) detrend + surface=0 (comme elle)
    profil_det, trend = _detrend_and_surface0(profil)

    # 5) FFT -> période (mais on force une plage de périodes)
    profil_centre = profil_det - np.mean(profil_det)
    fft = np.fft.rfft(profil_centre)
    fft[0] = 0.0

    # fréquences en "cycles / pixel" (si pas d'unité réelle)
    freqs_px = np.fft.rfftfreq(len(profil_centre), d=1.0)

    # période en pixels = 1/f
    with np.errstate(divide="ignore", invalid="ignore"):
        periods_px = 1.0 / freqs_px

    # masque plage de périodes
    pmin = max(2, int(period_min_px))
    pmax = max(pmin + 1, int(period_max_px))

    band = np.isfinite(periods_px) & (periods_px >= pmin) & (periods_px <= pmax)
    if not np.any(band):
        raise ValueError("FFT: aucune fréquence dans la plage period_min_px/period_max_px")

    idx_band = np.where(band)[0]
    idx_peak = idx_band[int(np.argmax(np.abs(fft[idx_band])))]
    periode_px = int(round(periods_px[idx_peak]))
    periode_px = max(2, periode_px)

    # période en µm si possible
    if np.isfinite(resolution_um_px) and resolution_um_px > 0:
        periode_um = float(periode_px * resolution_um_px)
    else:
        periode_um = float("nan")

    # 6) largeur sillon
    largeur_sillon_px = int(round(width_ratio * periode_px))
    largeur_sillon_px = max(1, largeur_sillon_px)
    largeur_sillon_um = float(largeur_sillon_px * resolution_um_px) if np.isfinite(resolution_um_px) else float("nan")

    # 7) profondeur auto (comme elle)
    profondeur_um = float(abs(np.percentile(profil_det, float(depth_percentile))))

    # 8) modèle créneau
    modele = np.zeros_like(profil_det)
    toggle = False
    for start in range(0, len(modele), largeur_sillon_px):
        end = start + largeur_sillon_px
        if toggle:
            modele[start:end] = -profondeur_um
        toggle = not toggle

    # 9) alignement corrélation (comme elle)
    corr = np.correlate(
        profil_det - np.mean(profil_det),
        modele - np.mean(modele),
        mode="full",
    )
    decalage_px = int(np.argmax(corr) - (len(modele) - 1))
    modele_aligne = np.roll(modele, decalage_px)

    # 10) Ra / Rq / kurtosis
    Ra = float(np.mean(np.abs(profil_det)))
    Rq = float(np.sqrt(np.mean(profil_det**2)))
    zstd = float(np.std(profil_det))
    kurtosis = float(np.mean((profil_det - np.mean(profil_det)) ** 4) / (zstd**4)) if zstd > 0 else float("nan")

    return {
        "resolution_um_px": float(resolution_um_px),
        "periode_um": float(periode_um),
        "periode_px": int(periode_px),
        "largeur_sillon_px": int(largeur_sillon_px),
        "largeur_sillon_um": float(largeur_sillon_um),
        "profondeur_um": float(profondeur_um),
        "decalage_px": int(decalage_px),
        "Ra": float(Ra),
        "Rq": float(Rq),
        "kurtosis": float(kurtosis),
        "profil_det": profil_det,
        "modele_aligne": modele_aligne,
    }