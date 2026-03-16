import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
#"pas = 1000 et decalage = 800 profondeur moyenne = 4,90
# =============================
# 1. PARAMÈTRES EXPÉRIMENTAUX
# =============================
largeur_mm = 47  # largeur physique de l'image (mm)
largeur_um = largeur_mm * 1000  # conversion en µm

# =============================
# 2. PARAMÈTRES MANUELS
# =============================
pas_sillon_um = float(input("Entrez le pas des sillons (µm) : "))
decalage_horizontal_um = float(input("Entrez la longueur à ajouter au début (µm) : "))

# =============================
# 3. LECTURE DU FICHIER .XYZ
# =============================
data = []
with open("C:/Users/anaid/Documents/M1/projet_semmelle_ski/PMMA-I/ImX0Y0.xyz", "r") as f:
    for line in f:
        try:
            vals = [float(v) for v in line.split()]
            if len(vals) >= 3:
                data.append(vals[:3])
        except ValueError:
            continue

data = np.array(data)
x, y, z = data[:,0], data[:,1], data[:,2]

# Reconstruction grille
x_unique = np.unique(x)
y_unique = np.unique(y)
nx, ny = len(x_unique), len(y_unique)

Z = np.full((ny, nx), np.nan)
x_index = {val:i for i,val in enumerate(x_unique)}
y_index = {val:i for i,val in enumerate(y_unique)}

for xi, yi, zi in zip(x, y, z):
    Z[y_index[yi], x_index[xi]] = zi

# =============================
# 4. PROFIL CENTRAL
# =============================
profil = Z[ny//2, :]
mask = ~np.isnan(profil)
profil = profil[mask]
px = np.arange(len(profil))

# Résolution spatiale en µm/pixel
resolution_um_px = largeur_um / nx

# =============================
# 5. SUPPRESSION DE LA PENTE
# =============================
coeffs = np.polyfit(px, profil, 1)
tendance = np.polyval(coeffs, px)
profil_det = profil - tendance

# Profondeur moyenne à utiliser pour le modèle
profondeur_mesuree_um = profil_det.max() - profil_det.min()

# =============================
# 6. MODÈLE CRÉNEAU RÉGULIER
# =============================
pas_sillon_px = max(int(round(pas_sillon_um / resolution_um_px)), 1)
profil_modele = np.zeros_like(profil_det)

toggle = False
for start in range(0, len(profil_modele), pas_sillon_px):
    end = start + pas_sillon_px
    if toggle:
        profil_modele[start:end] = -profondeur_mesuree_um
    toggle = not toggle

profil_modele = profil_modele[:len(profil_det)]

# =============================
# 7. DÉCALAGE HORIZONTAL
# =============================
decalage_px = int(round(decalage_horizontal_um / resolution_um_px))
profil_modele_decale = np.zeros_like(profil_det)
profil_modele_decale[decalage_px:] = profil_modele[:len(profil_modele) - decalage_px]


# =============================
# Alignement vertical : pic le plus haut à 0
# =============================
profil_det -= profil_det.max()  # pic le plus haut à 0



# =============================
# 8. AFFICHAGE
# =============================
plt.figure(figsize=(10,4))
plt.plot(profil_det, label="Profil mesuré")
plt.plot(profil_modele_decale, '--', label=f"Modèle créneau aligné")
plt.xlabel("Pixel")
plt.ylabel("Profondeur (µm)")
plt.title("Profil mesuré vs modèle créneau régulier")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# =============================
# 9. RÉSULTATS
# =============================
print("===== RÉSULTATS =====")
print(f"Pas des sillons entré (µm)      : {pas_sillon_um}")
print(f"Décalage horizontal (µm)        : {decalage_horizontal_um}")
print(f"Profondeur moyenne du profil (µm): {profondeur_mesuree_um:.2f}")
