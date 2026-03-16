import numpy as np
import matplotlib.pyplot as plt

# =============================
# 1. Lecture du fichier .XYZ
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
# 2. Profil central
# =============================
profil = Z[ny//2, :]
mask = ~np.isnan(profil)
profil = profil[mask]
px = np.arange(len(profil))

# Suppression de la pente
coeffs = np.polyfit(px, profil, 1)
tendance = np.polyval(coeffs, px)
profil_det = profil - tendance

# Décaler le profil pour que le pic le plus haut soit à 0
profil_det -= profil_det.max()

# =============================
# 3. Affichage
# =============================
plt.figure(figsize=(10,4))
plt.plot(profil_det, color='blue', label="Profil mesuré")
plt.xlabel("Pixel")
plt.ylabel("Profondeur (µm)")
plt.title("Profil central détendu")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

