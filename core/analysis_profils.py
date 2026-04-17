from __future__ import annotations

import numpy as np

from skfda import FDataGrid
from skfda.exploratory.depth import ModifiedBandDepth


def process_image_from_grid(Z, x_unique, y_unique) -> dict:
    """
    Analyse les profils d'une surface Z.

    Retourne un dict avec tous les arrays nécessaires pour
    construire les graphes Plotly côté app.py :
      - profiles       : (n_profils, nx) profils détrondés
      - profil_det     : (nx,) profil moyen détrendé
      - modele_aligne  : (nx,) modèle créneau aligné
      - median_fd      : (nx,) médiane fonctionnelle (Modified Band Depth)
      - lower_fd       : (nx,) borne basse de la bande centrale 50%
      - upper_fd       : (nx,) borne haute de la bande centrale 50%
      - periode_um     : float
      - profondeur_um  : float
      - resolution_um  : float (µm/px)
    """

    # ==============================
    # PREP
    # ==============================
    Z = np.array(Z, dtype=float)
    Z[~np.isfinite(Z)] = np.nan
    Z = Z - np.nanmax(Z)

    # ==============================
    # PROFILS
    # ==============================
    PROFILE_STEP = 10

    profiles_raw = np.array([Z[i, :] for i in range(0, Z.shape[0], PROFILE_STEP)])
    # Garder seulement les profils sans NaN
    profiles_raw = profiles_raw[~np.isnan(profiles_raw).any(axis=1)]

    if profiles_raw.shape[0] < 3:
        raise ValueError("Pas assez de profils valides (besoin d'au moins 3 lignes sans NaN).")

    def detrend(p: np.ndarray) -> np.ndarray:
        x = np.arange(len(p))
        coeffs = np.polyfit(x, p, 1)
        trend = np.polyval(coeffs, x)
        p = p - trend
        p -= np.max(p)
        return p

    profiles = np.array([detrend(p) for p in profiles_raw])

    # Profil moyen normalisé : max à exactement 0
    profil_det = np.mean(profiles, axis=0)
    profil_det = profil_det - np.max(profil_det)

    # ==============================
    # FFT → période + modèle
    # ==============================
    x_unique = np.asarray(x_unique, dtype=float)
    resolution = float((x_unique.max() - x_unique.min()) / (len(profil_det) - 1))

    profil_centre = profil_det - np.mean(profil_det)
    fft = np.fft.rfft(profil_centre)
    freqs = np.fft.rfftfreq(len(profil_det), d=resolution)
    fft[0] = 0

    # Band-limit pour éviter les fréquences parasites
    with np.errstate(divide="ignore", invalid="ignore"):
        periods_um = 1.0 / freqs
    band = np.isfinite(periods_um) & (periods_um >= 10) & (periods_um <= 2000)
    if not np.any(band):
        idx = int(np.argmax(np.abs(fft)))
    else:
        idx_band = np.where(band)[0]
        idx = idx_band[int(np.argmax(np.abs(fft[idx_band])))]

    periode_um = float(1.0 / freqs[idx])
    periode_px = max(2, int(round(periode_um / resolution)))
    largeur_px = max(1, periode_px // 2)

    # Profondeur : médiane des percentiles 5% par profil individuel
    # (plus robuste que le percentile global sur le profil moyen bruité)
    profondeur = float(np.median([abs(np.percentile(p, 5)) for p in profiles]))

    # ==============================
    # MODÈLE CRÉNEAU
    # ==============================
    modele = np.zeros_like(profil_det)
    toggle = False
    for i in range(0, len(modele), largeur_px):
        if toggle:
            modele[i : i + largeur_px] = -profondeur
        toggle = not toggle

    # ==============================
    # ALIGNEMENT PAR CORRÉLATION
    # ==============================
    p_norm = profil_det - np.mean(profil_det)
    m_norm = modele - np.mean(modele)
    corr = np.correlate(p_norm, m_norm, mode="full")
    shift = int(np.argmax(corr)) - (len(modele) - 1)
    modele_aligne = np.roll(modele, shift)

    # ==============================
    # FUNCTIONAL BOXPLOT (skfda)
    # ==============================
    x_grid = np.arange(profiles.shape[1])
    fd = FDataGrid(data_matrix=profiles, grid_points=x_grid)
    depth = ModifiedBandDepth()
    depths = depth(fd)

    order = np.argsort(depths)[::-1]
    sorted_profiles = profiles[order]

    median_fd = sorted_profiles[0]
    n_central = max(1, len(sorted_profiles) // 2)
    central = sorted_profiles[:n_central]
    lower_fd = np.min(central, axis=0)
    upper_fd = np.max(central, axis=0)

    # Normaliser la médiane à 0 au max
    _ref = np.max(median_fd)
    median_fd = median_fd - _ref
    lower_fd  = lower_fd  - _ref
    upper_fd  = upper_fd  - _ref

    return {
        "profiles":      profiles,
        "profil_det":    profil_det,
        "modele_aligne": modele_aligne,
        "median_fd":     median_fd,
        "lower_fd":      lower_fd,
        "upper_fd":      upper_fd,
        "periode_um":    periode_um,
        "profondeur_um": profondeur,
        "resolution_um": resolution,
    }
