# viz/plots.py
from __future__ import annotations
import numpy as np
from matplotlib.figure import Figure



def fig_heatmap(Z: np.ndarray, title: str = "", show_colorbar: bool = True) -> Figure:
    fig = Figure(figsize=(6.5, 4.5), dpi=120)
    ax = fig.add_subplot(111)
    im = ax.imshow(Z, origin="lower", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    if show_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig


def fig_mask(mask: np.ndarray, title: str = "") -> Figure:
    fig = Figure(figsize=(6.5, 4.5), dpi=120)
    ax = fig.add_subplot(111)
    ax.imshow(mask.astype(float), origin="lower", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("x index")
    ax.set_ylabel("y index")
    fig.tight_layout()
    return fig


def fig_profile_mean(Z: np.ndarray, title: str = "") -> Figure:
    prof = np.nanmean(Z, axis=1)
    fig = Figure(figsize=(6.5, 3.6), dpi=120)
    ax = fig.add_subplot(111)
    ax.plot(prof)
    ax.set_title(title)
    ax.set_xlabel("y index")
    ax.set_ylabel("mean(z)")
    fig.tight_layout()
    return fig





def fig_profiles_sample(Z: np.ndarray, n_lines: int = 10, axis: int = 0, title: str = "") -> Figure:
    """
    Plot quelques profils individuels.
    axis=0 -> lignes (y), donc profils en x
    axis=1 -> colonnes (x), donc profils en y
    """
    fig = Figure(figsize=(10, 4), dpi=120)
    axp = fig.add_subplot(111)

    if Z.size == 0:
        axp.set_title("Z vide")
        return fig

    if axis == 0:
        ny = Z.shape[0]
        idx = np.linspace(0, ny - 1, min(n_lines, ny), dtype=int)
        for i in idx:
            axp.plot(Z[i, :], label=f"Ligne {i}")
        axp.set_xlabel("index X")
        axp.set_ylabel("z")
    else:
        nx = Z.shape[1]
        idx = np.linspace(0, nx - 1, min(n_lines, nx), dtype=int)
        for j in idx:
            axp.plot(Z[:, j], label=f"Col {j}")
        axp.set_xlabel("index Y")
        axp.set_ylabel("z")

    axp.set_title(title or "Profils individuels (échantillonnés)")
    axp.grid(True)
    if len(idx) <= 12:
        axp.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


def fig_profile_band_mean(Z: np.ndarray, axis: int = 0, title: str = "", ref_to_zero: bool = False) -> Figure:
    """
    Plot mean + bande min-max.
    axis=0 -> agrège sur y => profil en x
    axis=1 -> agrège sur x => profil en y
    ref_to_zero: si True, soustrait le max (surface=0)
    """
    fig = Figure(figsize=(10, 4), dpi=120)
    axp = fig.add_subplot(111)

    if Z.size == 0:
        axp.set_title("Z vide")
        return fig

    Z2 = Z.copy()
    if ref_to_zero:
        Z2 = Z2 - np.nanmax(Z2)

    prof_mean = np.nanmean(Z2, axis=axis)
    prof_min = np.nanmin(Z2, axis=axis)
    prof_max = np.nanmax(Z2, axis=axis)

    x = np.arange(len(prof_mean))
    axp.fill_between(x, prof_min, prof_max, alpha=0.3, label="Dispersion min–max")
    axp.plot(x, prof_mean, linewidth=2, label="Profil moyen")

    axp.set_title(title or "Profil moyen + dispersion")
    axp.set_xlabel("index")
    axp.set_ylabel("z")
    axp.grid(True)
    axp.legend()
    fig.tight_layout()
    return fig