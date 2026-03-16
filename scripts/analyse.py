import os
import numpy as np
import matplotlib.pyplot as plt

from parse_xyz import lire_xyz, creer_grille
from analyse_structures import frequence_sillons
from classification import classifier_structure

def sauvegarder_image(z, outpath):
    plt.figure(figsize=(7, 5))
    plt.imshow(z, cmap='viridis')
    plt.colorbar(label="Hauteur (µm)")
    plt.title(os.path.basename(outpath))
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()

def main():

    racine = os.path.dirname(os.path.abspath(__file__))
    dossier_xyz = os.path.join(racine, "..", "data_raw")
    dossier_out = os.path.join(racine, "..", "classification")

    os.makedirs(dossier_out, exist_ok=True)

    for cat in ["CAT1_bandes_fines", "CAT2_bandes_moyennes", "CAT3_bandes_larges"]:
        os.makedirs(os.path.join(dossier_out, cat), exist_ok=True)

    for root, dirs, files in os.walk(dossier_xyz):
        for f in files:
            if not f.endswith(".xyz"):
                continue

            chemin = os.path.join(root, f)
            print("\nTraitement :", chemin)

            try:
                pts = lire_xyz(chemin)
                z = creer_grille(pts)

                freq = frequence_sillons(z)
                categorie = classifier_structure(freq)

                out_path = os.path.join(
                    dossier_out, categorie, f.replace(".xyz", ".png")
                )
                sauvegarder_image(z, out_path)

                print(" → CLASSÉ :", categorie)

            except Exception as e:
                print("Erreur sur", f, ":", e)

    print("\nClassification terminée !")

if __name__ == "__main__":
    main()
