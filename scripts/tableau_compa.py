import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.interpolate import interp1d

# =============================
# FONCTIONS UTILES
# =============================
def Ra_fun(v):
    return float(np.mean(np.abs(v)))

def Rq_fun(v):
    return float(np.sqrt(np.mean(v**2)))

def kurtosis_excess(v):
    m = np.mean(v)
    s = np.std(v)
    if s == 0:
        return np.nan
    return abs(float(np.mean((v - m)**4) / (s**4) - 3))

def L2_curve(a, b, dx):
    return float(np.sqrt(np.sum((a - b)**2) * dx))

# =============================
# FONCTION PRINCIPALE
# =============================
def analyse_xyz(filepath):

    # === lecture fichier
    data = []
    with open(filepath, "r") as f:
        for line in f:
            try:
                vals = [float(v) for v in line.split()]
                if len(vals) >= 3:
                    data.append(vals[:3])
            except:
                continue

    data = np.array(data)
    x, y, z = data[:, 0], data[:, 1], data[:, 2]

    # === grille
    x_unique = np.unique(x)
    y_unique = np.unique(y)
    nx, ny = len(x_unique), len(y_unique)

    Z = np.full((ny, nx), np.nan)
    x_index = {val: i for i, val in enumerate(x_unique)}
    y_index = {val: i for i, val in enumerate(y_unique)}

    for xi, yi, zi in zip(x, y, z):
        Z[y_index[yi], x_index[xi]] = zi

    # === calibration
    largeur_image_um = x_unique.max() - x_unique.min()
    resolution_um_px = largeur_image_um / (nx - 1)

    Z_ref = Z - np.nanmax(Z)

    # === profil moyen
    profil_moyen = np.nanmean(Z_ref, axis=0)
    px = np.arange(len(profil_moyen))

    coeffs = np.polyfit(px, profil_moyen, 1)
    profil_det = profil_moyen - np.polyval(coeffs, px)
    profil_det -= np.max(profil_det)

    # === FFT
    profil_centre = profil_det - np.mean(profil_det)
    fft = np.fft.rfft(profil_centre)
    freqs = np.fft.rfftfreq(len(profil_centre), d=resolution_um_px)
    fft[0] = 0

    idx_max = np.argmax(np.abs(fft))
    periode_um = 1 / freqs[idx_max]
    periode_px = int(round(periode_um / resolution_um_px))

    largeur_sillon_px = periode_px // 2
    largeur_sillon_um = largeur_sillon_px * resolution_um_px

    profondeur_sillon_um = abs(np.percentile(profil_det, 5))

    # === modèle créneau
    profil_modele = np.zeros_like(profil_det)
    toggle = False
    for start in range(0, len(profil_modele), largeur_sillon_px):
        if toggle:
            profil_modele[start:start+largeur_sillon_px] = -profondeur_sillon_um
        toggle = not toggle

    # === alignement
    corr = np.correlate(profil_det - np.mean(profil_det),
                        profil_modele - np.mean(profil_modele), mode='full')
    decalage = np.argmax(corr) - (len(profil_modele) - 1)
    profil_modele_aligne = np.roll(profil_modele, decalage)

    # =============================
    # PARAMÈTRES BLEU / ORANGE
    # =============================
    Ra_bleu = Ra_fun(profil_det)
    Rq_bleu = Rq_fun(profil_det)
    K_bleu = kurtosis_excess(profil_det)

    Ra_orange = Ra_fun(profil_modele_aligne)
    Rq_orange = Rq_fun(profil_modele_aligne)
    K_orange = kurtosis_excess(profil_modele_aligne)

    # =============================
    # L2 GLOBAL
    # =============================
    L2 = L2_curve(profil_det, profil_modele_aligne, resolution_um_px)

    # =============================
    # L2 NORMALISÉ STABLE (ligne par ligne)
    # =============================
    L2_norm_list = []
    x_px = np.arange(nx)

    for i in range(ny):
        ligne = Z_ref[i, :]
        if np.all(np.isnan(ligne)):
            continue

        mask = ~np.isnan(ligne)
        ligne = ligne[mask]
        x_loc = x_px[mask]

        if len(ligne) < 10:
            continue

        coeffs = np.polyfit(x_loc, ligne, 1)
        ligne_det = ligne - np.polyval(coeffs, x_loc)
        ligne_det -= np.max(ligne_det)

        corr = np.correlate(ligne_det - np.mean(ligne_det),
                            profil_modele - np.mean(profil_modele), mode='full')
        decalage = np.argmax(corr) - (len(profil_modele) - 1)
        modele_aligne = np.roll(profil_modele, decalage)

        interp_modele = interp1d(np.arange(len(modele_aligne)), modele_aligne)
        modele_resized = interp_modele(np.linspace(0, len(modele_aligne)-1, len(ligne_det)))

        diff = ligne_det - modele_resized

        amp = np.max(modele_resized) - np.min(modele_resized)
        if amp < 1e-6:
            continue

        L2_norm_list.append(np.sqrt(np.mean((diff/amp)**2)))

    L2_norm_stable = np.mean(L2_norm_list) if len(L2_norm_list) > 0 else np.nan

    # =============================
    # ERREURS RELATIVES
    # =============================
    dRa = abs((Ra_bleu - Ra_orange) / Ra_orange) if Ra_orange != 0 else np.nan
    dRq = abs((Rq_bleu - Rq_orange) / Rq_orange) if Rq_orange != 0 else np.nan
    dK = abs(K_bleu - K_orange)

    # =============================
    # RETURN
    # =============================
    return {
        "dossier": os.path.basename(os.path.dirname(filepath)),
        "fichier": os.path.basename(filepath),

        # bleu
        "Ra_bleu": Ra_bleu,
        "Rq_bleu": Rq_bleu,
        "K_bleu": K_bleu,

        # orange
        "Ra_orange": Ra_orange,
        "Rq_orange": Rq_orange,
        "K_orange": K_orange,

        # erreurs
        "L2": L2,
        "L2_norm_stable": L2_norm_stable,
        "dRa": dRa,
        "dRq": dRq,
        "dK": dK,

        # géométrie
        "periode_um": periode_um,
        "largeur_sillon_um": largeur_sillon_um,
        "profondeur_um": profondeur_sillon_um
    }

# =============================
# TRAITEMENT DOSSIER (RÉCURSIF)
# =============================
dossier = "C:/Users/Admin/Documents/Projet_Ski/profilo/data/PMMA"

resultats = []

for root, _, files in os.walk(dossier):
    for fichier in files:
        if fichier.lower().endswith(".xyz"):
            path = os.path.join(root, fichier)
            print("Analyse :", path)
            try:
                res = analyse_xyz(path)
                resultats.append(res)
            except Exception as e:
                print("Erreur sur", fichier, ":", e)

# =============================
# EXPORT CSV
# =============================
df = pd.DataFrame(resultats)

if df.empty:
    print("Aucun fichier analysé.")
else:
    df.to_csv(
        "resultats_profils.csv",
        mode="w",   # 🔥 IMPORTANT : écrase au lieu d'empiler
        header=True,
        index=False
    )

    print("\nCSV généré avec", len(df), "fichiers")