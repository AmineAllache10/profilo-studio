# core/analysis_recalage.py
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d


def _detrend_surface0(profil: np.ndarray, x_loc: np.ndarray | None = None) -> np.ndarray:
    """Supprime la pente (polyfit deg 1) et remet max à 0."""
    px = x_loc if x_loc is not None else np.arange(len(profil))
    coeffs = np.polyfit(px, profil, 1)
    trend = np.polyval(coeffs, px)
    det = profil - trend
    det -= np.nanmax(det)
    return det


def _build_modele_creneau(n: int, largeur_px: int, profondeur_um: float) -> np.ndarray:
    """Créneau symétrique de période 2*largeur_px."""
    modele = np.zeros(n)
    toggle = False
    for start in range(0, n, largeur_px):
        end = start + largeur_px
        if toggle:
            modele[start:end] = -profondeur_um
        toggle = not toggle
    return modele


def _aligne_correlation(profil: np.ndarray, modele: np.ndarray) -> tuple[np.ndarray, int]:
    """Aligne le modèle sur le profil par corrélation croisée."""
    p_norm = profil - np.mean(profil)
    m_norm = modele - np.mean(modele)
    corr = np.correlate(p_norm, m_norm, mode="full")
    decalage = int(np.argmax(corr) - (len(modele) - 1))
    return np.roll(modele, decalage), decalage


def analyse_recalage(
    Z: np.ndarray,
    x_vals: np.ndarray | None = None,
    ref_surface_zero: bool = True,
    depth_percentile: float = 5.0,
    period_min_px: int = 20,
    period_max_px: int = 300,
    width_ratio: float = 0.5,
) -> dict:
    """
    Pipeline complète de recalage :
      1. Référence Z = 0 (max → 0)
      2. Profil moyen
      3. Calibration spatiale (résolution µm/px)
      4. Suppression de pente + surface=0
      5. FFT → période dominante (band-limited)
      6. Modèle créneau
      7. Alignement par corrélation
      8. Ra / Rq / Kurtosis (profil moyen)
      9. Kurtosis ligne par ligne
     10. Erreur L2 ligne par ligne (absolue + normalisée)
     11. Erreurs relatives Ra / Rq / Kurtosis (profil vs modèle)
    """
    if Z is None or np.size(Z) == 0:
        raise ValueError("Z vide")

    Z2 = np.array(Z, dtype=float, copy=True)
    ny, nx = Z2.shape

    # ---- 1. Référence surface = 0
    if ref_surface_zero:
        Z2 = Z2 - np.nanmax(Z2)

    # ---- 2. Profil moyen
    profil_moyen = np.nanmean(Z2, axis=0)

    # ---- 3. Résolution µm/px
    resolution_um_px: float = float("nan")
    if x_vals is not None:
        xv = np.asarray(x_vals, dtype=float)
        mask_x = np.isfinite(profil_moyen)
        if xv.size == mask_x.size:
            xv = xv[mask_x]
        if xv.size >= 2:
            n_valid = int(mask_x.sum())
            resolution_um_px = float((xv.max() - xv.min()) / max(n_valid - 1, 1))
    if not np.isfinite(resolution_um_px):
        # fallback : 1 pixel = 1 µm
        resolution_um_px = 1.0

    # ---- enlève NaN du profil moyen
    mask_valid = np.isfinite(profil_moyen)
    profil = profil_moyen[mask_valid]
    if profil.size < 50:
        raise ValueError("Profil moyen trop court / trop de NaN")

    # ---- 4. Detrend + surface=0
    profil_det = _detrend_surface0(profil)

    # ---- 5. FFT → période
    profil_centre = profil_det - np.mean(profil_det)
    fft = np.fft.rfft(profil_centre)
    fft[0] = 0.0
    freqs_px = np.fft.rfftfreq(len(profil_centre), d=1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        periods_px = 1.0 / freqs_px

    pmin = max(2, int(period_min_px))
    pmax = max(pmin + 1, int(period_max_px))
    band = np.isfinite(periods_px) & (periods_px >= pmin) & (periods_px <= pmax)
    if not np.any(band):
        raise ValueError("FFT: aucune fréquence dans la plage period_min_px/period_max_px")

    idx_band = np.where(band)[0]
    idx_peak = idx_band[int(np.argmax(np.abs(fft[idx_band])))]
    periode_px = int(round(periods_px[idx_peak]))
    periode_px = max(2, periode_px)
    periode_um = float(periode_px * resolution_um_px)

    # ---- 6. Largeur sillon + profondeur
    largeur_sillon_px = max(1, int(round(width_ratio * periode_px)))
    largeur_sillon_um = float(largeur_sillon_px * resolution_um_px)
    profondeur_um = float(abs(np.percentile(profil_det, float(depth_percentile))))

    # ---- 7. Modèle créneau + alignement
    modele = _build_modele_creneau(len(profil_det), largeur_sillon_px, profondeur_um)
    modele_aligne, decalage_px = _aligne_correlation(profil_det, modele)

    # ---- 8. Ra / Rq / Kurtosis excès profil moyen (aligné script analyse.py)
    def _kurtosis_excess(v: np.ndarray) -> float:
        m = float(np.mean(v))
        s = float(np.std(v))
        if s < 1e-12:
            return float("nan")
        return abs(float(np.mean((v - m) ** 4) / s ** 4) - 3.0)

    Ra = float(np.mean(np.abs(profil_det)))
    Rq = float(np.sqrt(np.mean(profil_det ** 2)))
    kurtosis_profil = _kurtosis_excess(profil_det)

    # ---- 8b. Métriques modèle aligné
    Ra_modele = float(np.mean(np.abs(modele_aligne)))
    Rq_modele = float(np.sqrt(np.mean(modele_aligne ** 2)))
    kurtosis_modele = _kurtosis_excess(modele_aligne)

    # ---- 8c. L2 global (profil moyen vs modele aligné)
    L2_global = float(np.sqrt(np.sum((profil_det - modele_aligne) ** 2) * resolution_um_px))

    # ---- 9. Kurtosis ligne par ligne
    x_px = np.arange(nx)
    kurtosis_lignes = []

    for i in range(ny):
        ligne = Z2[i, :]
        if np.all(np.isnan(ligne)):
            continue
        mask_l = ~np.isnan(ligne)
        ligne_v = ligne[mask_l]
        x_loc = x_px[mask_l]
        if len(ligne_v) < 10:
            continue
        ligne_det = _detrend_surface0(ligne_v, x_loc)
        std_l = np.std(ligne_det)
        if std_l < 1e-12:
            continue
        kurt = float(np.mean((ligne_det - np.mean(ligne_det)) ** 4) / std_l ** 4)
        kurtosis_lignes.append(kurt)

    kurtosis_lignes_arr = np.array(kurtosis_lignes) if kurtosis_lignes else np.array([float("nan")])
    kurtosis_moyenne = float(np.nanmean(kurtosis_lignes_arr))
    kurtosis_std = float(np.nanstd(kurtosis_lignes_arr))

    # ---- 10. Erreur L2 ligne par ligne
    erreurs_L2 = []
    L2_normalises = []
    amplitude_modele = float(np.max(modele_aligne) - np.min(modele_aligne))

    for i in range(ny):
        ligne = Z2[i, :]
        if np.all(np.isnan(ligne)):
            continue
        mask_l = ~np.isnan(ligne)
        ligne_v = ligne[mask_l]
        x_loc = x_px[mask_l]
        if len(ligne_v) < 10:
            continue

        ligne_det = _detrend_surface0(ligne_v, x_loc)

        # alignement modèle sur cette ligne
        m_aligne_l, _ = _aligne_correlation(ligne_det, modele)

        # interpolation pour correspondre à la taille de la ligne
        interp_fn = interp1d(
            np.arange(len(m_aligne_l)), m_aligne_l, kind="linear", fill_value="extrapolate"
        )
        m_resized = interp_fn(np.linspace(0, len(m_aligne_l) - 1, len(ligne_det)))

        diff = ligne_det - m_resized

        # L2 absolue
        erreur_L2 = float(np.sqrt(np.sum(diff ** 2) * resolution_um_px))
        erreurs_L2.append(erreur_L2)

        # L2 normalisée
        if amplitude_modele > 1e-6:
            L2_norm = float(np.sqrt(np.mean((diff / amplitude_modele) ** 2)))
            L2_normalises.append(L2_norm)

    erreurs_L2_arr = np.array(erreurs_L2) if erreurs_L2 else np.array([float("nan")])
    L2_normalises_arr = np.array(L2_normalises) if L2_normalises else np.array([float("nan")])

    erreur_L2_moyenne = float(np.nanmean(erreurs_L2_arr))
    erreur_L2_std = float(np.nanstd(erreurs_L2_arr))
    L2_norm_moyenne = float(np.nanmean(L2_normalises_arr))
    L2_norm_std = float(np.nanstd(L2_normalises_arr))

    # ---- 11. Erreurs relatives (profil vs modèle)
    def _err_pct(a: float, b: float) -> float:
        if abs(b) < 1e-12:
            return float("nan")
        return float(abs((a - b) / b) * 100)

    erreur_Ra_pct = _err_pct(Ra, Ra_modele)
    erreur_Rq_pct = _err_pct(Rq, Rq_modele)
    # dK = difference absolue (aligné script analyse.py)
    erreur_kurtosis_pct = _err_pct(kurtosis_moyenne, kurtosis_modele)
    dK = abs(kurtosis_moyenne - kurtosis_modele)

    return {
        # calibration
        "resolution_um_px": resolution_um_px,
        # période / sillons
        "periode_um": periode_um,
        "periode_px": periode_px,
        "largeur_sillon_px": largeur_sillon_px,
        "largeur_sillon_um": largeur_sillon_um,
        "profondeur_um": profondeur_um,
        "decalage_px": decalage_px,
        # métriques profil moyen
        "Ra": Ra,
        "Rq": Rq,
        "kurtosis_profil": kurtosis_profil,
        # métriques modèle
        "Ra_modele": Ra_modele,
        "Rq_modele": Rq_modele,
        "kurtosis_modele": kurtosis_modele,
        # kurtosis ligne par ligne
        "kurtosis_moyenne": kurtosis_moyenne,
        "kurtosis_std": kurtosis_std,
        "n_lignes": len(kurtosis_lignes),
        # L2
        "erreur_L2_moyenne": erreur_L2_moyenne,
        "erreur_L2_std": erreur_L2_std,
        "L2_norm_moyenne": L2_norm_moyenne,
        "L2_norm_std": L2_norm_std,
        # erreurs relatives %
        "erreur_Ra_pct": erreur_Ra_pct,
        "erreur_Rq_pct": erreur_Rq_pct,
        "erreur_kurtosis_pct": erreur_kurtosis_pct,
        "dK": dK,
        "L2_global": L2_global,
        # arrays pour plot
        "profil_det": profil_det,
        "modele_aligne": modele_aligne,
        "erreurs_L2_arr": erreurs_L2_arr,
        "L2_normalises_arr": L2_normalises_arr,
        "kurtosis_lignes_arr": kurtosis_lignes_arr,
    }