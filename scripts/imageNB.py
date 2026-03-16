import numpy as np
import matplotlib.pyplot as plt

def generer_sillons(
    taille=1000,
    pas_mm=0.3,
    profondeur_mm=0.025,
    largeur_mm=47,
    noir=0,
    blanc=255
):
    """
    Génère une image N&B de sillons type diamant (créneaux)

    Parameters
    ----------
    taille : int
        Taille de l'image (taille x taille)
    pas_mm : float
        Pas des sillons en mm
    profondeur_mm : float
            Profondeur des sillons (information physique, utile plus tard
    largeur_mm : float
        Largeur physique de l'image en mm
    noir, blanc : int
        Valeurs de niveaux de gris

    Returns
    -------
    image : ndarray (taille x taille)
    profil_1d : ndarray (taille)
    """
    
    # Conversion mm → pixels
    resolution = largeur_mm / taille  # mm / pixel
    pas_px = pas_mm / resolution

    # Axe x
    x = np.arange(taille)

    # Créneau 1D
    profil_1d = ((x // pas_px) % 2)
    
    # Convention : 1 = creux (noir), 0 = plat (blanc)
    profil_1d = np.where(profil_1d == 1, noir, blanc)

    # Image 2D par répétition
    image = np.tile(profil_1d, (taille, 1))

    return image, profil_1d


image, profil = generer_sillons(
    pas_mm=2,
    profondeur_mm=0.25
)

plt.figure(figsize=(10,3))
plt.imshow(image, cmap='gray')
plt.title("Image N&B – sillons type diamant")
plt.axis('off')
plt.show()

plt.figure(figsize=(10,3))
plt.plot(profil, linewidth=2)
plt.title("Profil 1D – créneaux")
plt.xlabel("Pixel")
plt.ylabel("Niveau de gris")
plt.grid(True)
plt.show()


profondeur_mm = 0.025

profil_mm = np.where(profil == 0, -profondeur_mm, 0)
plt.plot(profil_mm)
plt.ylabel("Profondeur (mm)")
plt.grid()
plt.show()



