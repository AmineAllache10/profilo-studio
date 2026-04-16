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
from core.analysis_sillons import analyse_sillons_from_grid
from core.analysis_recalage import analyse_recalage
from core.inventory import (
    build_inventory_local,
    build_inventory_drive,
    clear_inventory_cache,
    load_inventory_from_cache,
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

# -----------------------------
# Session state init
# -----------------------------
if "inventory" not in st.session_state:
    st.session_state.inventory = None
if "source_mode" not in st.session_state:
    st.session_state.source_mode = "Google Drive"
if "local_root" not in st.session_state:
    st.session_state.local_root = ""

with st.sidebar:
    st.header("Dataset")

    source_mode = st.selectbox(
        "Source des données",
        options=["Google Drive", "Local"],
        index=0,
        key="source_mode_select",
    )
    st.session_state.source_mode = source_mode

    # -----------------------------
    # Mode Local : explorateur de dossiers
    # -----------------------------
    if source_mode == "Local":
        st.caption("Navigue jusqu'au dossier contenant tes fichiers .xyz")

        # Zone de saisie du chemin
        typed_path = st.text_input(
            "Chemin du dossier",
            value=st.session_state.local_root or os.path.expanduser("~"),
            key="local_path_input",
        )

        # Autocomplétion : liste les sous-dossiers du chemin saisi
        if typed_path and os.path.isdir(typed_path):
            try:
                subdirs = sorted([
                    d for d in os.listdir(typed_path)
                    if os.path.isdir(os.path.join(typed_path, d))
                    and not d.startswith(".")
                ])
                if subdirs:
                    chosen = st.selectbox(
                        "Sous-dossiers disponibles",
                        options=["(rester ici)"] + subdirs,
                        index=0,
                        key="subdir_select",
                    )
                    if chosen != "(rester ici)":
                        typed_path = os.path.join(typed_path, chosen)
                        st.session_state.local_root = typed_path

                # Compter les .xyz dans le dossier choisi
                n_xyz = sum(
                    1 for _, _, fnames in os.walk(typed_path)
                    for fn in fnames if fn.lower().endswith(".xyz")
                )
                st.caption(f"`{typed_path}`  —  **{n_xyz} fichiers .xyz** trouvés")
                root = typed_path
                st.session_state.local_root = typed_path
            except PermissionError:
                st.warning("Accès refusé à ce dossier.")
                root = typed_path
        else:
            st.warning("Chemin invalide ou inaccessible.")
            root = typed_path
    else:
        root = ""

    st.divider()

    cbtn1, cbtn2 = st.columns(2)
    with cbtn1:
        do_scan = st.button("Scanner", use_container_width=True)
    with cbtn2:
        reset_cache = st.button("Reset cache", use_container_width=True)

    if reset_cache:
        clear_inventory_cache()
        st.session_state.inventory = None
        st.success("Cache supprimé.")

    st.divider()
    st.header("Sélection")
    st.caption("Les filtres s'appliquent à l'inventaire.")

# -----------------------------
# Chargement auto du cache au démarrage
# -----------------------------
if st.session_state.inventory is None:
    cached = load_inventory_from_cache(source_mode.lower().replace(" ", "_") if source_mode != "Google Drive" else "drive")
    if cached is not None:
        st.session_state.inventory = cached

# -----------------------------
# Scan manuel
# -----------------------------
if do_scan:
    if source_mode == "Local":
        if not root or not os.path.exists(root):
            st.error("Chemin invalide : le dossier n'existe pas.")
        else:
            progress_bar = st.progress(0, text="Préparation du scan...")

            def local_progress(i, total, fname):
                if total > 0:
                    pct = int((i / total) * 100)
                    progress_bar.progress(pct, text=f"Scan ({i}/{total}) : {fname}")

            inv = build_inventory_local(root, progress_cb=local_progress)
            progress_bar.progress(100, text="Scan termine")
            st.session_state.inventory = inv
    else:
        progress_bar = st.progress(0, text="Connexion Google Drive...")

        def drive_progress(i, total, fname):
            if total > 0:
                pct = int((i / total) * 100)
                progress_bar.progress(pct, text=f"Téléchargement ({i}/{total}) : {fname}")

        inv = build_inventory_drive(service, DRIVE_FOLDER_ID, progress_cb=drive_progress)
        progress_bar.progress(100, text="Scan termine")
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
        "Recalage",
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
    else:
        indices = list(range(filtered_inv.shape[0]))

        sel_idx = st.selectbox(
            "Fichier actif (Visionneuse/Manquants)",
            options=indices,
            index=0,
            format_func=lambda i: filtered_inv.iloc[i]["chemin"],
        )
        sel_row = filtered_inv.iloc[sel_idx]



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
                    # Modèle théorique PMMA : créneau à la même résolution que Z
                    ny_g, nx_g = gridp.Z.shape

                    # Période via FFT sur profil moyen (même logique que recalage)
                    profil_tmp = np.nanmean(gridp.Z, axis=0)
                    mask_tmp = np.isfinite(profil_tmp)
                    profil_tmp = profil_tmp[mask_tmp]

                    if profil_tmp.size > 10:
                        p = profil_tmp - np.mean(profil_tmp)
                        fft_tmp = np.fft.rfft(p)
                        fft_tmp[0] = 0
                        freqs_tmp = np.fft.rfftfreq(len(p), d=1.0)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            periods_tmp = 1.0 / freqs_tmp
                        band = np.isfinite(periods_tmp) & (periods_tmp >= 10) & (periods_tmp <= 300)
                        if np.any(band):
                            idx_b = np.where(band)[0]
                            idx_peak = idx_b[int(np.argmax(np.abs(fft_tmp[idx_b])))]
                            periode_px = max(4, int(round(periods_tmp[idx_peak])))
                        else:
                            periode_px = nx_g // 6
                    else:
                        periode_px = nx_g // 6

                    largeur_px = max(2, periode_px // 2)

                    # Créneau binaire : même shape que Z (ny_g x nx_g)
                    # valeurs dans le même range que Z pour comparaison valable
                    z_min = float(np.nanmin(gridp.Z))
                    z_max = float(np.nanmax(gridp.Z))

                    ligne_modele = np.zeros(nx_g, dtype=float)
                    toggle = False
                    for start in range(0, nx_g, largeur_px):
                        end = min(start + largeur_px, nx_g)
                        ligne_modele[start:end] = z_max if not toggle else z_min
                        toggle = not toggle

                    # Même grille que Z : motif répété sur toutes les lignes Y
                    Z_modele = np.tile(ligne_modele, (ny_g, 1))

                    st.pyplot(fig_heatmap(Z_modele, title="Modèle théorique (sillons PMMA)"))
                    st.caption(f"Période détectée : {periode_px} px · Même échelle couleur que la mesure")
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
# Tab 4: Recalage
# -----------------------------
with tabs[3]:
    st.subheader("Recalage")
    st.info("""
        Pipeline complète de recalage sur le fichier actif :

        1. Référence surface = 0 (max → 0)
        2. Calibration spatiale (résolution µm/px)
        3. Profil moyen + suppression de pente
        4. FFT → détection de la période dominante
        5. Modèle créneau (diamant) + alignement par corrélation
        6. Métriques globales : Ra, Rq, Kurtosis (profil vs modèle)
    """)

    if sel_row is None:
        st.info("Aucun fichier sélectionné.")
    else:
        path = sel_row["chemin"]
        if "PMMA" not in path:
            st.error("Cette analyse est disponible uniquement pour les profils linéaires (PMMA).")
            st.stop()

        rec_col, _ = st.columns([1, 3])
        with rec_col:
            ref0_rec = st.checkbox("Référence surface = 0", value=True, key="rec_ref0")
            depth_pct_rec = st.slider("Percentile profondeur", 1.0, 20.0, 5.0, 0.5, key="rec_depth")
            period_min_rec = st.number_input("Période min (px)", value=20, min_value=2, key="rec_pmin")
            period_max_rec = st.number_input("Période max (px)", value=300, min_value=3, key="rec_pmax")
            run_rec = st.button("Lancer recalage", key="run_recalage")

        if run_rec:
            try:
                with st.spinner("Lecture + recalage en cours..."):
                    _, g_rec = load_grid_or_scatter(sel_row)

                    if not g_rec.is_grid:
                        st.warning("Recalage disponible uniquement pour les fichiers en grille.")
                        st.stop()

                    res_rec = analyse_recalage(
                        g_rec.Z,
                        x_vals=g_rec.x_vals,
                        ref_surface_zero=ref0_rec,
                        depth_percentile=float(depth_pct_rec),
                        period_min_px=int(period_min_rec),
                        period_max_px=int(period_max_rec),
                    )

                st.session_state["recalage_res"] = res_rec

            except Exception as e:
                st.error(str(e))

        res_rec = st.session_state.get("recalage_res", None)

        if res_rec is None:
            st.info("Clique sur 'Lancer recalage'.")
        else:
            st.subheader("Paramètres détectés")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Résolution (µm/px)", f"{res_rec['resolution_um_px']:.3f}")
            col2.metric("Période (µm)", f"{res_rec['periode_um']:.1f}")
            col3.metric("Largeur sillon (µm)", f"{res_rec['largeur_sillon_um']:.1f}")
            col4.metric("Profondeur (µm)", f"{res_rec['profondeur_um']:.2f}")

            st.subheader("Profil moyen vs modèle créneau")

            profil_plot = res_rec["profil_det"].copy().astype(float)
            modele_plot = res_rec["modele_aligne"].copy().astype(float)

            valeurs_valides = np.concatenate([
                profil_plot[np.isfinite(profil_plot)],
                modele_plot[np.isfinite(modele_plot)],
            ])

            if valeurs_valides.size > 0:
                p1 = float(np.percentile(valeurs_valides, 1))
                p99 = float(np.percentile(valeurs_valides, 99))
                marge = abs(p99 - p1) * 0.15
                y_min_plot = p1 - marge
                y_max_plot = p99 + marge
                profil_plot[profil_plot < y_min_plot] = np.nan
            else:
                y_min_plot, y_max_plot = None, None

            fig_pvm = Figure(figsize=(11, 4), dpi=120)
            ax_pvm = fig_pvm.add_subplot(111)

            ax_pvm.plot(
                profil_plot,
                label=f"Profil mesure (Ra={res_rec['Ra']:.3f}, Rq={res_rec['Rq']:.3f}, K={res_rec['kurtosis_profil']:.2f})",
            )

            ax_pvm.plot(
                modele_plot, "--", color="orange",
                label=f"Modele (Ra={res_rec['Ra_modele']:.3f}, Rq={res_rec['Rq_modele']:.3f}, K={res_rec['kurtosis_modele']:.2f})",
            )

            if y_min_plot is not None:
                ax_pvm.set_ylim(y_min_plot, y_max_plot)

            ax_pvm.set_xlabel("Index X")
            ax_pvm.set_ylabel("Profondeur (µm)")
            ax_pvm.set_title("Profil moyen vs modele aligne")
            ax_pvm.grid(True, alpha=0.4)
            ax_pvm.legend(loc="lower right")
            fig_pvm.tight_layout()

            st.pyplot(fig_pvm)

            st.subheader("Métriques : profil vs modèle")
            mc1, mc2, mc3 = st.columns(3)

            mc1.metric(
                "Ra mesuré / modèle (µm)",
                f"{res_rec['Ra']:.3f} / {res_rec['Ra_modele']:.3f}",
                delta=f"{res_rec['erreur_Ra_pct']:.1f} % d'écart",
            )

            mc2.metric(
                "Rq mesuré / modèle (µm)",
                f"{res_rec['Rq']:.3f} / {res_rec['Rq_modele']:.3f}",
                delta=f"{res_rec['erreur_Rq_pct']:.1f} % d'écart",
            )

            mc3.metric(
                "Kurtosis moy / modèle",
                f"{res_rec['kurtosis_moyenne']:.2f} / {res_rec['kurtosis_modele']:.2f}",
                delta=f"{res_rec['erreur_kurtosis_pct']:.1f} % d'écart",
            )

            st.divider()

            col_tab, col_exp = st.columns([1.2, 1])

            with col_tab:
                st.subheader("Recapitulatif des metriques")

                recap = {
                    "fichier": str(sel_row["chemin"]),
                    "Ra_mesure": res_rec["Ra"],
                    "Rq_mesure": res_rec["Rq"],
                    "K_mesure": res_rec["kurtosis_profil"],
                    "Ra_modele": res_rec["Ra_modele"],
                    "Rq_modele": res_rec["Rq_modele"],
                    "K_modele": res_rec["kurtosis_modele"],
                    "dRa_pct": res_rec["erreur_Ra_pct"],
                    "dRq_pct": res_rec["erreur_Rq_pct"],
                    "dK_abs": res_rec["dK"],
                    "L2_global": res_rec["L2_global"],
                    "periode_um": res_rec["periode_um"],
                    "largeur_sillon_um": res_rec["largeur_sillon_um"],
                    "profondeur_um": res_rec["profondeur_um"],
                    "resolution_um_px": res_rec["resolution_um_px"],
                }

                df_recap = pd.DataFrame([recap])
                st.dataframe(df_recap.T.rename(columns={0: "Valeur"}), use_container_width=True)

                csv_bytes = df_recap.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Télécharger CSV",
                    data=csv_bytes,
                    file_name="recalage_resultats.csv",
                    mime="text/csv",
                )

            with col_exp:
                st.subheader("Formules")

                st.markdown("Ra")
                st.latex(r"Ra = \frac{1}{n} \sum |z_i|")
                st.caption("Rugosité moyenne")

                st.divider()

                st.markdown("Rq")
                st.latex(r"Rq = \sqrt{\frac{1}{n} \sum z_i^2}")
                st.caption("Sensibilité aux pics")

                st.divider()

                st.markdown("K")
                st.latex(r"K = \frac{1}{n} \sum \left(\frac{z_i - \bar{z}}{\sigma}\right)^4")
                st.caption("Forme des structures")

                st.divider()

                st.markdown("L2")
                st.latex(r"L2 = \sqrt{\sum (z_{mes} - z_{mod})^2}")
                st.caption("Écart global")

                st.divider()

                st.markdown("Erreur %")
                st.latex(r"\left|\frac{X_{mes} - X_{mod}}{X_{mod}}\right| \times 100")
                st.caption("Écart relatif")




# -----------------------------
# Tab 5: Analyse profils
# -----------------------------
with tabs[4]:
    st.subheader("Analyse des profils (1D)")

    st.info("""
Réduction de dimension :

La surface 2D (Z) est transformée en profils 1D pour simplifier l’analyse.

Méthodologie :
- Extraction de plusieurs lignes de la grille
- Visualisation des profils individuels
- Calcul d’un profil moyen
- Analyse de la dispersion (min / max)
""")

    if sel_row is None:
        st.info("Aucun fichier sélectionné.")
    else:
        with st.spinner("Lecture complète..."):
            dfP, gP = load_grid_or_scatter(sel_row)

        if not gP.is_grid:
            st.warning("Analyse profils disponible seulement si fichier en grille.")
        else:
            # -----------------------------
            # Paramètres utilisateur
            # -----------------------------
            colA, colB = st.columns(2)
            with colA:
                n_lines = st.slider("Nombre de profils affichés", 3, 30, 10)
            with colB:
                ref0 = st.checkbox("Centrer la surface (max = 0)", value=True)

            Z = gP.Z.copy()

            if ref0:
                Z = Z - np.nanmax(Z)

            ny, nx = Z.shape
            indices = np.linspace(0, ny - 1, n_lines, dtype=int)

            # -----------------------------
            # 1. Profils individuels
            # -----------------------------
            fig1 = Figure(figsize=(10, 4), dpi=120)
            ax1 = fig1.add_subplot(111)

            for i in indices:
                ax1.plot(Z[i, :], alpha=0.7)

            ax1.set_title("Profils individuels")
            ax1.set_xlabel("Position X")
            ax1.set_ylabel("Profondeur (µm)")
            ax1.grid(True)

            st.pyplot(fig1)

            # -----------------------------
            # 2. Profil moyen + dispersion
            # -----------------------------
            profil_moyen = np.nanmean(Z, axis=0)
            profil_min = np.nanmin(Z, axis=0)
            profil_max = np.nanmax(Z, axis=0)

            fig2 = Figure(figsize=(10, 4), dpi=120)
            ax2 = fig2.add_subplot(111)

            ax2.fill_between(
                range(nx),
                profil_min,
                profil_max,
                alpha=0.3,
                label="Dispersion (min–max)"
            )
            ax2.plot(profil_moyen, linewidth=2, label="Profil moyen")

            ax2.set_title("Profil moyen et dispersion")
            ax2.set_xlabel("Position X")
            ax2.set_ylabel("Profondeur (µm)")
            ax2.legend()
            ax2.grid(True)

            st.pyplot(fig2)

            st.caption("Reduction : surface 2D vers profils 1D pour analyse simplifiee")

            st.divider()

            # -----------------------------
            # 3. Classification et visualisation CSV
            # -----------------------------
            st.subheader("Classification des sillons")

            CSV_PATH = "resultats_profils.csv"

            if not os.path.exists(CSV_PATH):
                st.info("Aucun resultat disponible. Lance le recalage sur plusieurs fichiers et exporte via l'onglet Recalage.")
            else:
                df_csv = pd.read_csv(CSV_PATH).dropna(subset=["largeur_sillon_um", "profondeur_um"])

                if df_csv.empty:
                    st.warning("Le CSV est vide ou ne contient pas les colonnes attendues.")
                else:
                    def classify_sillons(row):
                        largeur = row["largeur_sillon_um"]
                        profondeur = row["profondeur_um"]
                        if largeur < 30:
                            return "Type 1 : fin peu profond"
                        elif largeur > 60:
                            if profondeur < 11:
                                return "Type 4 : large intermediaire"
                            else:
                                return "Type 3 : large profond"
                        else:
                            return "Type 2 : moyen profond"

                    df_csv["groupe"] = df_csv.apply(classify_sillons, axis=1)

                    MARKERS = {
                        "Type 1 : fin peu profond": "^",
                        "Type 2 : moyen profond": "s",
                        "Type 3 : large profond": "o",
                        "Type 4 : large intermediaire": "D",
                    }
                    COLORS = {
                        "Type 1 : fin peu profond": "#4C72B0",
                        "Type 2 : moyen profond": "#DD8452",
                        "Type 3 : large profond": "#55A868",
                        "Type 4 : large intermediaire": "#C44E52",
                    }

                    st.write(f"Fichiers charges : **{len(df_csv)}** — Groupes : **{df_csv['groupe'].nunique()}**")
                    st.dataframe(df_csv[["fichier", "largeur_sillon_um", "profondeur_um", "groupe"]].sort_values("groupe"), use_container_width=True)

                    # ---- Ra ideal vs reel
                    st.markdown("**Ra : ideal vs reel**")
                    fig_ra = Figure(figsize=(6, 4), dpi=110)
                    ax_ra = fig_ra.add_subplot(111)
                    for grp in df_csv["groupe"].unique():
                        sub = df_csv[df_csv["groupe"] == grp]
                        ax_ra.scatter(sub["Ra_modele"], sub["Ra_mesure"],
                                      label=grp, marker=MARKERS.get(grp, "o"),
                                      color=COLORS.get(grp, "gray"))
                    mn = min(df_csv["Ra_modele"].min(), df_csv["Ra_mesure"].min())
                    mx = max(df_csv["Ra_modele"].max(), df_csv["Ra_mesure"].max())
                    ax_ra.plot([mn, mx], [mn, mx], "r--", label="Parfait")
                    ax_ra.set_xlabel("Ra ideal (modele)")
                    ax_ra.set_ylabel("Ra reel (mesure)")
                    ax_ra.set_title("Comparaison Ra")
                    ax_ra.legend(fontsize=7)
                    ax_ra.grid(True, alpha=0.4)
                    fig_ra.tight_layout()
                    st.pyplot(fig_ra)

                    # ---- Rq ideal vs reel
                    st.markdown("**Rq : ideal vs reel**")
                    fig_rq = Figure(figsize=(6, 4), dpi=110)
                    ax_rq = fig_rq.add_subplot(111)
                    for grp in df_csv["groupe"].unique():
                        sub = df_csv[df_csv["groupe"] == grp]
                        ax_rq.scatter(sub["Rq_modele"], sub["Rq_mesure"],
                                      label=grp, marker=MARKERS.get(grp, "o"),
                                      color=COLORS.get(grp, "gray"))
                    mn = min(df_csv["Rq_modele"].min(), df_csv["Rq_mesure"].min())
                    mx = max(df_csv["Rq_modele"].max(), df_csv["Rq_mesure"].max())
                    ax_rq.plot([mn, mx], [mn, mx], "r--", label="Parfait")
                    ax_rq.set_xlabel("Rq ideal (modele)")
                    ax_rq.set_ylabel("Rq reel (mesure)")
                    ax_rq.set_title("Comparaison Rq")
                    ax_rq.legend(fontsize=7)
                    ax_rq.grid(True, alpha=0.4)
                    fig_rq.tight_layout()
                    st.pyplot(fig_rq)

                    # ---- Kurtosis reel vs K ideal
                    st.markdown("**Kurtosis reel vs modele ideal**")
                    fig_k = Figure(figsize=(8, 4), dpi=110)
                    ax_k = fig_k.add_subplot(111)
                    K_ideal = float(df_csv["K_modele"].iloc[0]) if "K_modele" in df_csv.columns else None
                    for grp in df_csv["groupe"].unique():
                        sub = df_csv[df_csv["groupe"] == grp]
                        ax_k.scatter(sub["largeur_sillon_um"], sub["K_mesure"],
                                     label=grp, marker=MARKERS.get(grp, "o"),
                                     color=COLORS.get(grp, "gray"))
                    if K_ideal is not None:
                        ax_k.axhline(y=K_ideal, linestyle="--", color="red", label=f"K ideal = {K_ideal:.2f}")
                    ax_k.set_xlabel("Largeur sillon (µm)")
                    ax_k.set_ylabel("Kurtosis mesure")
                    ax_k.set_title("Ecart kurtosis reel vs modele ideal")
                    ax_k.legend(fontsize=7)
                    ax_k.grid(True, alpha=0.4)
                    fig_k.tight_layout()
                    st.pyplot(fig_k)

                    # ---- Geometrie vs L2 (colormap)
                    st.markdown("**Geometrie des sillons vs erreur L2**")
                    fig_l2g = Figure(figsize=(8, 4), dpi=110)
                    ax_l2g = fig_l2g.add_subplot(111)
                    for grp in df_csv["groupe"].unique():
                        sub = df_csv[df_csv["groupe"] == grp]
                        sc = ax_l2g.scatter(sub["largeur_sillon_um"], sub["profondeur_um"],
                                            c=sub["L2_global"], marker=MARKERS.get(grp, "o"),
                                            label=grp, cmap="YlOrRd", vmin=df_csv["L2_global"].min(),
                                            vmax=df_csv["L2_global"].max())
                    fig_l2g.colorbar(sc, ax=ax_l2g, label="L2 global")
                    ax_l2g.set_xlabel("Largeur (µm)")
                    ax_l2g.set_ylabel("Profondeur (µm)")
                    ax_l2g.set_title("Geometrie des sillons vs erreur L2")
                    ax_l2g.legend(fontsize=7)
                    ax_l2g.grid(True, alpha=0.4)
                    fig_l2g.tight_layout()
                    st.pyplot(fig_l2g)

                    # ---- Profondeur vs L2
                    st.markdown("**Impact profondeur sur L2**")
                    fig_d = Figure(figsize=(6, 4), dpi=110)
                    ax_d = fig_d.add_subplot(111)
                    for grp in df_csv["groupe"].unique():
                        sub = df_csv[df_csv["groupe"] == grp]
                        ax_d.scatter(sub["profondeur_um"], sub["L2_global"],
                                     label=grp, marker=MARKERS.get(grp, "o"),
                                     color=COLORS.get(grp, "gray"))
                    ax_d.set_xlabel("Profondeur (µm)")
                    ax_d.set_ylabel("L2 global")
                    ax_d.set_title("Impact de la profondeur sur L2")
                    ax_d.legend(fontsize=7)
                    ax_d.grid(True, alpha=0.4)
                    fig_d.tight_layout()
                    st.pyplot(fig_d)

                    # ---- Top kurtosis
                    st.markdown("**Top 5 kurtosis (usure outil)**")
                    top_k = df_csv.sort_values("K_mesure", ascending=False).head(5)
                    st.dataframe(top_k[["fichier", "K_mesure", "largeur_sillon_um", "profondeur_um"]], use_container_width=True)

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