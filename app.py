# app.py
# Streamlit "Studio Profilo" V2 (modulaire) : inventaire, visionneuse, manquants, comparaison, rapport

import os
import io
import zipfile
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from core.grid import GridResult, to_grid
from core.missing import fill_missing
from core.metrics import compare_metrics
from core.analysis_sillons import analyse_sillons_from_grid
from core.inventory import (
    build_inventory_local,
    build_inventory_drive,
    clear_inventory_cache,
    read_xyz_points_cached,
)

from core.io_drive import get_drive_service, download_drive_file_to_temp
from core.io_xyz import read_xyz_points

from viz.plots import fig_heatmap, fig_mask
from viz.plots import fig_profiles_sample, fig_profile_band_mean


# -----------------------------
# Config Streamlit
# -----------------------------
st.set_page_config(page_title="Profilo Studio", layout="wide")

DRIVE_FOLDER_ID = "1PBtZj_Uc927MybfWwWP-kdHULsgXpCwF"
service = get_drive_service(dict(st.secrets["gcp"]))



def load_grid_or_scatter(row: pd.Series) -> tuple[pd.DataFrame, GridResult]:
    source = str(row.get("source", "local"))

    if source == "drive":
        file_id = str(row["file_id"])
        tmp_path = download_drive_file_to_temp(service, file_id, suffix=".xyz")
        try:
            df = read_xyz_points(tmp_path)
            grid = to_grid(df)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass
        return df, grid

    path = str(row["chemin"])
    df = read_xyz_points_cached(path)
    grid = to_grid(df)
    return df, grid


# -----------------------------
# UI
# -----------------------------
st.title("Profilo Studio")

with st.sidebar:
    st.header("Dataset")

    source_mode = st.selectbox(
        "Source des données",
        options=["Google Drive", "Local"],
        index=0,
    )

    root = ""
    if source_mode == "Local":
        root = st.text_input("Chemin dossier data", value="data")

    cbtn1, cbtn2 = st.columns(2)
    with cbtn1:
        do_scan = st.button("Scanner")
    with cbtn2:
        reset_cache = st.button("Reset cache")

    if reset_cache:
        clear_inventory_cache()
        st.session_state.inventory = None
        st.success("Cache inventaire supprimé.")

    st.divider()
    st.header("Sélection")
    st.caption("Les filtres s'appliquent à l'inventaire.")

if "inventory" not in st.session_state:
    st.session_state.inventory = None

if do_scan:
    if source_mode == "Local":
        if not os.path.exists(root):
            st.error("Chemin invalide: le dossier n'existe pas.")
        else:
            with st.spinner("Inventaire local..."):
                inv = build_inventory_local(root)
            st.session_state.inventory = inv
    else:
        with st.spinner("Inventaire Google Drive..."):
            inv = build_inventory_drive(service, DRIVE_FOLDER_ID)
        st.session_state.inventory = inv
        
inv = st.session_state.inventory
if inv is None:
    st.info("Renseigne le dossier puis clique Scanner.")
    st.stop()

# Filtres
with st.sidebar:
    grid_filter = st.multiselect(
        "Type",
        options=["Grille", "Hors-grille"],
        default=["Grille", "Hors-grille"],
    )
    only_missing = st.checkbox("Afficher seulement fichiers avec manquants > 0", value=False)
    search = st.text_input("Recherche (nom contient)", value="")

f = inv.copy()
if grid_filter != ["Grille", "Hors-grille"]:
    if "Grille" in grid_filter and "Hors-grille" not in grid_filter:
        f = f[f["is_grid"] == True]
    elif "Hors-grille" in grid_filter and "Grille" not in grid_filter:
        f = f[f["is_grid"] == False]

if only_missing:
    f = f[(f["missing_rate"].fillna(0.0) > 0.0)]
if search.strip():
    f = f[f["fichier"].str.contains(search.strip(), case=False, na=False)]

# Compteurs
colA, colB, colC, colD = st.columns(4)
colA.metric("Fichiers (filtrés)", int(f.shape[0]))
colB.metric("Fichiers en grille", int((f["is_grid"] == True).sum()))
colC.metric("Fichiers hors-grille", int((f["is_grid"] == False).sum()))
colD.metric("Avec manquants (>0)", int((f["missing_rate"].fillna(0.0) > 0.0).sum()))

tabs = st.tabs(
    [
        "Inventaire",
        "Visionneuse",
        "Données manquantes",
        "Comparer",
        "Analyse profils",
        "Analyse sillons",
        "Rapport",
        "Dictionnaire",
    ]
)

# -----------------------------
# Tab 1: Inventaire
# -----------------------------
with tabs[0]:
    st.subheader("Inventaire")
    st.dataframe(
        f.sort_values(["missing_rate"], ascending=[False]),
        use_container_width=True,
        height=520,
    )

    csv_bytes = f.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Exporter inventaire CSV",
        data=csv_bytes,
        file_name="inventaire_profilo.csv",
        mime="text/csv",
    )

# -----------------------------
# Sélections globales
# -----------------------------
filtered_inv = f.reset_index(drop=True)

with st.sidebar:
    if filtered_inv.shape[0] == 0:
        st.warning("Aucun fichier dans le filtre.")
        sel_row = None
        selA_row = None
        selB_row = None
    else:
        indices = list(range(filtered_inv.shape[0]))

        sel_idx = st.selectbox(
            "Fichier actif (Visionneuse/Manquants)",
            options=indices,
            index=0,
            format_func=lambda i: filtered_inv.iloc[i]["chemin"],
        )
        sel_row = filtered_inv.iloc[sel_idx]

        st.caption("Comparer")

        selA_idx = st.selectbox(
            "Fichier A",
            options=indices,
            index=0,
            key="selA",
            format_func=lambda i: filtered_inv.iloc[i]["chemin"],
        )
        selA_row = filtered_inv.iloc[selA_idx]

        selB_idx = st.selectbox(
            "Fichier B",
            options=indices,
            index=min(1, len(indices) - 1),
            key="selB",
            format_func=lambda i: filtered_inv.iloc[i]["chemin"],
        )
        selB_row = filtered_inv.iloc[selB_idx]

# -----------------------------
# Tab 2: Visionneuse
# -----------------------------
with tabs[1]:
    st.subheader("Visionneuse")
    st.info("""
            Visualisation :

            - Les données (.xyz) sont converties en grille 2D
            - Si grille → heatmap de la surface
            - Sinon → nuage de points

            Méthode standard de reconstruction et visualisation de surface
            """)

    if sel_row is None:
        st.info("Aucun fichier sélectionné.")
    else:
        left, right = st.columns([2, 1])

        with st.spinner("Lecture complète et construction de la grille..."):
            dfp, gridp = load_grid_or_scatter(sel_row)

        with left:
            if gridp.is_grid:
                st.caption("Comparaison entre la surface réelle mesurée et un modèle théorique de structure diamant")

                col1, col2 = st.columns(2)

                with col1:
                    st.pyplot(fig_heatmap(gridp.Z, title="Semelle mesurée"))

                with col2:
                    Z_model = np.sign(np.sin(np.linspace(0, 20, gridp.Z.shape[1])))
                    Z_model = np.tile(Z_model, (gridp.Z.shape[0], 1))
                    st.pyplot(fig_heatmap(Z_model, title="Modèle diamant (théorique)"))
            else:
                fig = Figure(figsize=(6.5, 4.5), dpi=120)
                ax = fig.add_subplot(111)
                n = dfp.shape[0]
                sample = dfp.sample(200000, random_state=0) if n > 200000 else dfp
                ax.scatter(sample["x"], sample["y"], s=1)
                ax.set_title("Nuage de points (x,y) - preview (échantillonné)")
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                fig.tight_layout()
                st.pyplot(fig)

        with right:
            st.write("**Fichier**")
            st.code(str(sel_row["chemin"]), language="text")

            st.write("**Résumé des indicateurs**")

            c1, c2 = st.columns(2)
            c1.metric("Points", f"{dfp.shape[0]:,}")
            c2.metric("Grille", "Oui" if gridp.is_grid else "Non")

            c3, c4 = st.columns(2)
            c3.metric("Dimensions", f"{gridp.nx} x {gridp.ny}")
            c4.metric("Manquants", f"{gridp.missing_rate*100:.2f}%")

            if gridp.is_grid:
                c5, c6 = st.columns(2)
                c5.metric("Z min", f"{float(np.nanmin(gridp.Z)):.1f}")
                c6.metric("Z max", f"{float(np.nanmax(gridp.Z)):.1f}")
            else:
                z = dfp["z"].to_numpy()
                c5, c6 = st.columns(2)
                c5.metric("Z min", f"{float(np.nanmin(z)):.1f}")
                c6.metric("Z max", f"{float(np.nanmax(z)):.1f}")

            if gridp.missing_rate > 0.05:
                st.warning("Beaucoup de données manquantes → analyse à interpréter avec prudence")
            else:
                st.success("Surface exploitable")

            if gridp.is_grid:
                img_buf = io.BytesIO()
                fig_out = fig_heatmap(gridp.Z, title="Surface (Z)")
                fig_out.savefig(img_buf, format="png")
                st.download_button(
                    "Exporter image PNG",
                    data=img_buf.getvalue(),
                    file_name="surface.png",
                    mime="image/png",
                )

# -----------------------------
# Tab 3: Données manquantes
# -----------------------------
with tabs[2]:
    st.subheader("Données manquantes")
    st.info("""
            Méthodologie :
            
            Les données manquantes correspondent aux zones où aucune valeur de hauteur z n’est disponible
            après reconstruction de la grille.

            Deux méthodes de remplissage sont proposées :

            - Nearest Neighbor (plus proche voisin) :
            méthode classique consistant à attribuer à une case manquante la valeur du point valide
            le plus proche. Cette méthode est rapide, mais elle peut produire un rendu moins lisse.

            - Interpolation :
            méthode classique d’estimation utilisant les valeurs voisines pour reconstruire les zones
            manquantes. Elle donne en général une surface plus continue, mais peut lisser certaines
            structures fines.

            Ces deux approches sont des méthodes standards en traitement de données spatiales.
            """)
    if sel_row is None:
        st.stop()

    with st.spinner("Lecture complète..."):
        dfm, gridm = load_grid_or_scatter(sel_row)

    if not gridm.is_grid:
        st.warning("Ce fichier ne ressemble pas à une grille régulière. Module manquants limité.")
        st.stop()

    st.write(f"missing_rate = {gridm.missing_rate:.6f}")

    c1, c2 = st.columns(2)
    with c1:
        st.pyplot(fig_mask(gridm.missing_mask, title="Masque manquants (1=manquant)"))
    with c2:
        st.pyplot(fig_heatmap(gridm.Z, title="Surface AVANT (avec trous)"))

    st.divider()
    st.write("**Remplissage des trous** (optionnel, affichage avant/après)")
    method = st.selectbox("Méthode", options=["Nearest", "Interpolate"], index=0)
    do_fill = st.checkbox("Appliquer remplissage", value=False)

    if do_fill:
        Z_filled = fill_missing(gridm.Z, method=method)
        d1, d2, d3 = st.columns(3)
        with d1:
            st.pyplot(fig_heatmap(gridm.Z, title="AVANT"))
        with d2:
            st.pyplot(fig_heatmap(Z_filled, title="APRES"))
        with d3:
            diff = np.abs(Z_filled - np.where(np.isfinite(gridm.Z), gridm.Z, Z_filled))
            st.pyplot(fig_heatmap(diff, title="DIFF (abs)"))
    else:
        st.info("Active 'Appliquer remplissage' pour voir l'APRES.")

# -----------------------------
# Tab 4: Comparer
# -----------------------------
with tabs[3]:
    st.subheader("Comparer")
    st.info("""
            Comparaison :

            - Comparaison point à point des deux surfaces
            - Carte |A - B| pour visualiser les écarts
            - Calcul de métriques globales

            Méthodes standards d’analyse de différence entre surfaces
            """)
    if selA_row is None or selB_row is None:
        st.stop()

    with st.spinner("Lecture complète A..."):
        _, gA = load_grid_or_scatter(selA_row)
    with st.spinner("Lecture complète B..."):
        _, gB = load_grid_or_scatter(selB_row)

    if not (gA.is_grid and gB.is_grid):
        st.warning("Comparaison complète prévue pour deux grilles.")
        st.stop()

    ny = min(gA.Z.shape[0], gB.Z.shape[0])
    nx = min(gA.Z.shape[1], gB.Z.shape[1])
    A = gA.Z[:ny, :nx]
    B = gB.Z[:ny, :nx]

    mcol, icol = st.columns([1, 2])
    with mcol:
        st.write("**Fichier A**")
        st.code(str(selA_row["chemin"]), language="text")
        st.write("**Fichier B**")
        st.code(str(selB_row["chemin"]), language="text")

        fill_cmp = st.checkbox("Remplir trous avant comparaison", value=False)
        method_cmp = st.selectbox(
            "Méthode remplissage",
            options=["Nearest", "Interpolate"],
            index=0,
            key="cmp_fill_method",
        )

        A2, B2 = A.copy(), B.copy()
        if fill_cmp:
            A2 = fill_missing(A2, method_cmp)
            B2 = fill_missing(B2, method_cmp)

        metrics = compare_metrics(A2, B2)
        st.write("**Métriques**")
        st.write(metrics)

    with icol:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.pyplot(fig_heatmap(A2, title="A"))
        with c2:
            st.pyplot(fig_heatmap(B2, title="B"))
        with c3:
            st.pyplot(fig_heatmap(np.abs(A2 - B2), title="|A-B|"))

# -----------------------------
# Tab 5: Analyse profils
# -----------------------------
with tabs[4]:
    st.subheader("Analyse profils (1D)")
    st.info("""
            Analyse des profils :
            
            - Extraction de profils 1D depuis la surface 2D
            - Calcul d’un profil moyen
            - Analyse de la dispersion autour du profil
            
            Méthodes classiques d’analyse de surface
            
            Réduction :
            Surface 2D → profils 1D
            --""")
    if sel_row is None:
        st.stop()

    with st.spinner("Lecture complète..."):
        dfP, gP = load_grid_or_scatter(sel_row)

    if not gP.is_grid:
        st.warning("Analyse profils disponible seulement si fichier en grille.")
        st.stop()

    cA, cB = st.columns([1, 2])
    with cA:
        axis = st.selectbox(
            "Direction des profils",
            options=["Profils en X (lignes Y)", "Profils en Y (colonnes X)"],
            index=0,
        )
        axis_id = 0 if axis.startswith("Profils en X") else 1
        n_lines = st.slider("Nombre de profils affichés", 3, 30, 10, 1)
        ref0 = st.checkbox("Référence surface = 0 (soustraire max)", value=True)

    Zuse = gP.Z.copy()
    if ref0:
        Zuse = Zuse - np.nanmax(Zuse)

    with cB:
        st.pyplot(fig_profiles_sample(Zuse, n_lines=n_lines, axis=axis_id, title="Profils individuels"))
        st.pyplot(fig_profile_band_mean(Zuse, axis=axis_id, title="Profil moyen + dispersion", ref_to_zero=False))

# -----------------------------
# Tab 6: Analyse sillons
# -----------------------------
with tabs[5]:
    st.subheader("Analyse sillons (FFT + modèle créneau)")
    st.info("""
            Analyse des sillons :

            Méthodes :
            - FFT : détecte la fréquence dominante → période des sillons
            - Modèle diamant (créneau) : modèle simplifié pour représenter la structure
            - Alignement : comparaison entre profil réel et modèle

            Réduction :
            Surface 2D → profil moyen (1D) → analyse fréquentielle
            """)

    if sel_row is None:
        st.info("Aucun fichier sélectionné.")
    else:
        with st.spinner("Lecture complète..."):
            dfS, gS = load_grid_or_scatter(sel_row)

        if not gS.is_grid:
            st.warning("Analyse sillons disponible seulement si fichier en grille.")
        else:
            left, right = st.columns([1, 2])

            with left:
                ref0 = st.checkbox("Référence surface = 0", value=True, key="s_ref0")
                depth_pct = st.slider("Percentile profondeur (ex: 5%)", 1.0, 20.0, 5.0, 0.5)

                if st.button("Lancer analyse", key="run_sillons"):
                    try:
                        res = analyse_sillons_from_grid(
                            gS.Z,
                            x_vals=gS.x_vals,
                            ref_surface_zero=ref0,
                            depth_percentile=float(depth_pct),
                        )
                        st.session_state["sillons_res"] = res
                    except Exception as e:
                        st.error(str(e))

            res = st.session_state.get("sillons_res", None)

            if res is None:
                st.info("Clique sur 'Lancer analyse'.")
            else:
                with left:
                    st.write("**Résultats**")
                    st.write(
                        {
                            "resolution_um_px": res["resolution_um_px"],
                            "periode_um": res["periode_um"],
                            "periode_px": res["periode_px"],
                            "largeur_sillon_px": res["largeur_sillon_px"],
                            "profondeur_um": res["profondeur_um"],
                            "decalage_px": res["decalage_px"],
                            "Ra": res["Ra"],
                            "Rq": res["Rq"],
                            "kurtosis": res["kurtosis"],
                        }
                    )

                with right:
                    fig = Figure(figsize=(10, 4), dpi=120)
                    ax = fig.add_subplot(111)
                    ax.plot(res["profil_det"], label="Semelle mesurée")
                    ax.plot(res["modele_aligne"], linestyle="--", label="Modèle diamant")
                    ax.set_title("Profil vs modèle")
                    ax.set_xlabel("index X")
                    ax.set_ylabel("profondeur (µm)")
                    ax.grid(True)
                    ax.legend()
                    fig.tight_layout()
                    st.pyplot(fig)
# -----------------------------
# Tab 7: Rapport
# -----------------------------
with tabs[6]:
    st.subheader("Rapport (exports)")

    if sel_row is None:
        st.info("Aucun fichier sélectionné.")
    else:
        do_fill_rep = st.checkbox("Inclure surface après remplissage trous", value=False, key="rep_fill")
        method_rep = st.selectbox(
            "Méthode remplissage",
            options=["Nearest", "Interpolate"],
            index=0,
            key="rep_fill_method",
        )

        if st.button("Générer ZIP rapport"):
            with st.spinner("Génération..."):
                with tempfile.TemporaryDirectory() as td:
                    f.to_csv(os.path.join(td, "inventaire_filtre.csv"), index=False)

                    dfR, gR = load_grid_or_scatter(sel_row)
                    base = os.path.splitext(os.path.basename(str(sel_row["chemin"])))[0]

                    if gR.is_grid:
                        statsR = pd.DataFrame(
                            [
                                {
                                    "fichier": os.path.basename(str(sel_row["chemin"])),
                                    "chemin": str(sel_row["chemin"]),
                                    "n_points": int(dfR.shape[0]),
                                    "is_grid": bool(gR.is_grid),
                                    "nx": int(gR.nx),
                                    "ny": int(gR.ny),
                                    "missing_rate": float(gR.missing_rate),
                                    "z_min": float(np.nanmin(gR.Z)),
                                    "z_max": float(np.nanmax(gR.Z)),
                                    "z_mean": float(np.nanmean(gR.Z)),
                                    "z_std": float(np.nanstd(gR.Z)),
                                }
                            ]
                        )
                        statsR.to_csv(os.path.join(td, f"{base}_stats.csv"), index=False)

                        fig1 = fig_heatmap(gR.Z, title=f"{base} - surface")
                        fig1.savefig(os.path.join(td, f"{base}_surface.png"), format="png")

                        fig2 = fig_mask(gR.missing_mask, title=f"{base} - masque manquants")
                        fig2.savefig(os.path.join(td, f"{base}_missing_mask.png"), format="png")

                        if do_fill_rep:
                            Zf = fill_missing(gR.Z, method_rep)
                            fig3 = fig_heatmap(Zf, title=f"{base} - surface apres fill ({method_rep})")
                            fig3.savefig(os.path.join(td, f"{base}_surface_filled.png"), format="png")

                    zip_path = os.path.join(td, "rapport_profilo.zip")
                    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                        for dirpath, _, filenames in os.walk(td):
                            for fn in filenames:
                                if fn.endswith(".zip"):
                                    continue
                                full = os.path.join(dirpath, fn)
                                rel = os.path.relpath(full, td)
                                zf.write(full, arcname=rel)

                    with open(zip_path, "rb") as fzip:
                        st.download_button(
                            "Télécharger rapport_profilo.zip",
                            data=fzip.read(),
                            file_name="rapport_profilo.zip",
                            mime="application/zip",
                        )


with tabs[7]:
    st.subheader("Dictionnaire")

    st.write("- **XYZ** : fichier contenant des points (x, y, z)")
    st.write("- **Grille** : représentation 2D de la surface")
    st.write("- **NaN** : valeur manquante")
    st.write("- **Interpolation** : estimation des valeurs manquantes")
    st.write("- **FFT** : analyse des fréquences d’un signal")
    st.write("- **Sillons** : rainures de la semelle")
    st.write("- **Diamant** : modèle théorique des sillons")







