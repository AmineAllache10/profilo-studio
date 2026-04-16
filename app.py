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
        "Analyse globale",
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
                import plotly.express as _px_viz
                st.caption("Comparaison entre la surface réelle mesurée et un modèle théorique de structure diamant")
                st.caption(" Zoom : sélectionner une zone · Double-clic pour réinitialiser")

                col1, col2 = st.columns(2)

                with col1:
                    _fig_mes = _px_viz.imshow(gridp.Z, origin="lower", color_continuous_scale="viridis",
                                              labels={"color": "Z (µm)"}, title="Semelle mesurée", aspect="auto")
                    _fig_mes.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(_fig_mes, use_container_width=True)

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

                    _fig_mod = _px_viz.imshow(Z_modele, origin="lower", color_continuous_scale="viridis",
                                               labels={"color": "Z (µm)"}, title="Modèle théorique (sillons PMMA)", aspect="auto")
                    _fig_mod.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
                    st.plotly_chart(_fig_mod, use_container_width=True)
                    st.caption(f"Période détectée : {periode_px} px · Même échelle couleur que la mesure")
            else:
                import plotly.express as _px_scatter
                n = dfp.shape[0]
                sample = dfp.sample(200000, random_state=0) if n > 200000 else dfp
                _fig_sc = _px_scatter.scatter(sample, x="x", y="y", render_mode="webgl",
                                              title="Nuage de points (x,y) — échantillonné",
                                              labels={"x": "x", "y": "y"})
                _fig_sc.update_traces(marker=dict(size=2))
                _fig_sc.update_layout(height=420)
                st.plotly_chart(_fig_sc, use_container_width=True)

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
    st.caption(" Zoom : sélectionner une zone · Double-clic pour réinitialiser")

    import plotly.express as _px_miss
    c1, c2 = st.columns(2)
    with c1:
        _fig_mask = _px_miss.imshow(gridm.missing_mask.astype(float), origin="lower",
                                    color_continuous_scale="reds", aspect="auto",
                                    title="Masque manquants (1 = manquant)",
                                    labels={"color": "Manquant"})
        _fig_mask.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(_fig_mask, use_container_width=True)
    with c2:
        _fig_avant = _px_miss.imshow(gridm.Z, origin="lower", color_continuous_scale="viridis",
                                     aspect="auto", title="Surface AVANT (avec trous)",
                                     labels={"color": "Z (µm)"})
        _fig_avant.update_layout(height=360, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(_fig_avant, use_container_width=True)

    st.divider()
    st.write("**Remplissage des trous** (optionnel, affichage avant/après)")
    method = st.selectbox("Méthode", options=["Nearest", "Interpolate"], index=0)
    do_fill = st.checkbox("Appliquer remplissage", value=False)

    if do_fill:
        Z_filled = fill_missing(gridm.Z, method=method)
        d1, d2, d3 = st.columns(3)
        with d1:
            _f1 = _px_miss.imshow(gridm.Z, origin="lower", color_continuous_scale="viridis", aspect="auto", title="AVANT", labels={"color": "Z"})
            _f1.update_layout(height=320, margin=dict(l=5, r=5, t=35, b=5))
            st.plotly_chart(_f1, use_container_width=True)
        with d2:
            _f2 = _px_miss.imshow(Z_filled, origin="lower", color_continuous_scale="viridis", aspect="auto", title="APRÈS", labels={"color": "Z"})
            _f2.update_layout(height=320, margin=dict(l=5, r=5, t=35, b=5))
            st.plotly_chart(_f2, use_container_width=True)
        with d3:
            diff = np.abs(Z_filled - np.where(np.isfinite(gridm.Z), gridm.Z, Z_filled))
            _f3 = _px_miss.imshow(diff, origin="lower", color_continuous_scale="hot", aspect="auto", title="DIFF (abs)", labels={"color": "Δ"})
            _f3.update_layout(height=320, margin=dict(l=5, r=5, t=35, b=5))
            st.plotly_chart(_f3, use_container_width=True)
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
        path = str(sel_row["chemin"])

        if "PMMA" not in path:
            st.error("Cette analyse est disponible uniquement pour les profils linéaires (PMMA).")
        else:
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
                        else:
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

# -----------------------------
# Tab 6: Analyse globale des sillons
# -----------------------------
with tabs[5]:
    import plotly.express as px
    import plotly.graph_objects as go

    st.subheader("Analyse globale des sillons")
    st.info("Tous les graphes sont interactifs : zoom, pan, survol pour les détails du fichier.")

    csv_path = "resultats_profils.csv"

    if not os.path.exists(csv_path):
        st.warning("Aucun résultat disponible. Lance le recalage sur plusieurs fichiers et exporte via l'onglet Recalage.")
    else:
        df = pd.read_csv(csv_path)
        df = df.rename(columns={
            "Ra_bleu": "Ra_mesure",
            "Rq_bleu": "Rq_mesure",
            "K_bleu": "K_mesure",
            "Ra_orange": "Ra_modele",
            "Rq_orange": "Rq_modele",
            "K_orange": "K_modele",
            "L2": "L2_global"
        })
        df = df.dropna(subset=["largeur_sillon_um", "profondeur_um"])

        if df.empty:
            st.warning("Le CSV est vide ou ne contient pas les colonnes attendues.")
        else:
            # -------------------------
            # CLASSIFICATION
            # -------------------------
            def classify_sillons(row):
                largeur = row["largeur_sillon_um"]
                profondeur = row["profondeur_um"]
                if largeur < 30:
                    return "Type 1 : fin peu profond"
                elif largeur > 60:
                    return "Type 3 : large profond" if profondeur > 11 else "Type 4 : large intermédiaire"
                else:
                    return "Type 2 : moyen profond"

            df["groupe"] = df.apply(classify_sillons, axis=1)
            df["fichier_court"] = df["fichier"].apply(lambda p: os.path.basename(str(p)))

            COLOR_MAP = {
                "Type 1 : fin peu profond":     "#4C72B0",
                "Type 2 : moyen profond":       "#DD8452",
                "Type 3 : large profond":       "#55A868",
                "Type 4 : large intermédiaire": "#C44E52",
            }
            SYMBOL_MAP = {
                "Type 1 : fin peu profond":     "triangle-up",
                "Type 2 : moyen profond":       "square",
                "Type 3 : large profond":       "circle",
                "Type 4 : large intermédiaire": "diamond",
            }

            st.write(f"Fichiers chargés : **{len(df)}** — Groupes : **{df['groupe'].nunique()}**")
            st.dataframe(
                df[["fichier_court", "largeur_sillon_um", "profondeur_um", "Ra_mesure", "Rq_mesure", "K_mesure", "L2_global", "groupe"]]
                .sort_values("groupe")
                .rename(columns={"fichier_court": "fichier"}),
                use_container_width=True,
                height=220,
            )

            PLOTLY_H = 420
            HOVER_BASE = "<b>%{customdata[0]}</b><br>Groupe : %{customdata[1]}<br>"

            # -------------------------
            # 1. Ra idéal vs réel
            # -------------------------
            st.subheader("Ra : idéal vs réel")
            fig_ra = go.Figure()
            for grp, grp_df in df.groupby("groupe"):
                fig_ra.add_trace(go.Scatter(
                    x=grp_df["Ra_modele"], y=grp_df["Ra_mesure"],
                    mode="markers",
                    name=grp,
                    marker=dict(color=COLOR_MAP.get(grp), symbol=SYMBOL_MAP.get(grp), size=10, line=dict(width=1, color="white")),
                    customdata=grp_df[["fichier_court", "groupe", "Ra_modele", "Ra_mesure", "largeur_sillon_um", "profondeur_um"]].values,
                    hovertemplate=(
                        HOVER_BASE +
                        "Ra modèle : %{customdata[2]:.4f} µm<br>"
                        "Ra mesuré : %{customdata[3]:.4f} µm<br>"
                        "Largeur : %{customdata[4]:.1f} µm · Profondeur : %{customdata[5]:.2f} µm"
                        "<extra></extra>"
                    ),
                ))
            mn = min(df["Ra_modele"].min(), df["Ra_mesure"].min())
            mx = max(df["Ra_modele"].max(), df["Ra_mesure"].max())
            fig_ra.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", line=dict(dash="dash", color="red"), name="Parfait", hoverinfo="skip"))
            fig_ra.update_layout(height=PLOTLY_H, xaxis_title="Ra idéal (modèle) µm", yaxis_title="Ra réel (mesuré) µm", legend_title="Groupe")
            st.plotly_chart(fig_ra, use_container_width=True)

            # -------------------------
            # 2. Rq idéal vs réel
            # -------------------------
            st.subheader("Rq : idéal vs réel")
            fig_rq = go.Figure()
            for grp, grp_df in df.groupby("groupe"):
                fig_rq.add_trace(go.Scatter(
                    x=grp_df["Rq_modele"], y=grp_df["Rq_mesure"],
                    mode="markers",
                    name=grp,
                    marker=dict(color=COLOR_MAP.get(grp), symbol=SYMBOL_MAP.get(grp), size=10, line=dict(width=1, color="white")),
                    customdata=grp_df[["fichier_court", "groupe", "Rq_modele", "Rq_mesure", "largeur_sillon_um", "profondeur_um"]].values,
                    hovertemplate=(
                        HOVER_BASE +
                        "Rq modèle : %{customdata[2]:.4f} µm<br>"
                        "Rq mesuré : %{customdata[3]:.4f} µm<br>"
                        "Largeur : %{customdata[4]:.1f} µm · Profondeur : %{customdata[5]:.2f} µm"
                        "<extra></extra>"
                    ),
                ))
            mn = min(df["Rq_modele"].min(), df["Rq_mesure"].min())
            mx = max(df["Rq_modele"].max(), df["Rq_mesure"].max())
            fig_rq.add_trace(go.Scatter(x=[mn, mx], y=[mn, mx], mode="lines", line=dict(dash="dash", color="red"), name="Parfait", hoverinfo="skip"))
            fig_rq.update_layout(height=PLOTLY_H, xaxis_title="Rq idéal (modèle) µm", yaxis_title="Rq réel (mesuré) µm", legend_title="Groupe")
            st.plotly_chart(fig_rq, use_container_width=True)

            # -------------------------
            # 3. Kurtosis vs largeur
            # -------------------------
            st.subheader("Kurtosis réel vs largeur de sillon")
            fig_k = go.Figure()
            K_ideal = float(df["K_modele"].iloc[0]) if "K_modele" in df.columns else None
            for grp, grp_df in df.groupby("groupe"):
                fig_k.add_trace(go.Scatter(
                    x=grp_df["largeur_sillon_um"], y=grp_df["K_mesure"],
                    mode="markers",
                    name=grp,
                    marker=dict(color=COLOR_MAP.get(grp), symbol=SYMBOL_MAP.get(grp), size=10, line=dict(width=1, color="white")),
                    customdata=grp_df[["fichier_court", "groupe", "K_mesure", "K_modele", "largeur_sillon_um", "profondeur_um"]].values,
                    hovertemplate=(
                        HOVER_BASE +
                        "K mesuré : %{customdata[2]:.3f}<br>"
                        "K modèle : %{customdata[3]:.3f}<br>"
                        "Largeur : %{customdata[4]:.1f} µm · Profondeur : %{customdata[5]:.2f} µm"
                        "<extra></extra>"
                    ),
                ))
            if K_ideal is not None:
                fig_k.add_hline(y=K_ideal, line_dash="dash", line_color="red", annotation_text=f"K idéal = {K_ideal:.2f}")
            fig_k.update_layout(height=PLOTLY_H, xaxis_title="Largeur sillon (µm)", yaxis_title="Kurtosis mesuré", legend_title="Groupe")
            st.plotly_chart(fig_k, use_container_width=True)

            # -------------------------
            # 4 & 5. L2 vs géométrie (2 graphes côte à côte)
            # -------------------------
            st.subheader("Erreur L2 vs géométrie")
            col_l2a, col_l2b = st.columns(2)

            with col_l2a:
                fig_l2larg = go.Figure()
                for grp, grp_df in df.groupby("groupe"):
                    fig_l2larg.add_trace(go.Scatter(
                        x=grp_df["largeur_sillon_um"], y=grp_df["L2_global"],
                        mode="markers", name=grp,
                        marker=dict(color=COLOR_MAP.get(grp), symbol=SYMBOL_MAP.get(grp), size=10, line=dict(width=1, color="white")),
                        customdata=grp_df[["fichier_court", "groupe", "largeur_sillon_um", "L2_global", "profondeur_um"]].values,
                        hovertemplate=(
                            HOVER_BASE +
                            "Largeur : %{customdata[2]:.1f} µm<br>"
                            "L2 : %{customdata[3]:.4f}<br>"
                            "Profondeur : %{customdata[4]:.2f} µm"
                            "<extra></extra>"
                        ),
                    ))
                fig_l2larg.update_layout(height=PLOTLY_H, xaxis_title="Largeur (µm)", yaxis_title="L2 global", legend_title="Groupe", showlegend=False)
                st.plotly_chart(fig_l2larg, use_container_width=True)

            with col_l2b:
                fig_l2prof = go.Figure()
                for grp, grp_df in df.groupby("groupe"):
                    fig_l2prof.add_trace(go.Scatter(
                        x=grp_df["profondeur_um"], y=grp_df["L2_global"],
                        mode="markers", name=grp,
                        marker=dict(color=COLOR_MAP.get(grp), symbol=SYMBOL_MAP.get(grp), size=10, line=dict(width=1, color="white")),
                        customdata=grp_df[["fichier_court", "groupe", "profondeur_um", "L2_global", "largeur_sillon_um"]].values,
                        hovertemplate=(
                            HOVER_BASE +
                            "Profondeur : %{customdata[2]:.2f} µm<br>"
                            "L2 : %{customdata[3]:.4f}<br>"
                            "Largeur : %{customdata[4]:.1f} µm"
                            "<extra></extra>"
                        ),
                    ))
                fig_l2prof.update_layout(height=PLOTLY_H, xaxis_title="Profondeur (µm)", yaxis_title="L2 global", legend_title="Groupe")
                st.plotly_chart(fig_l2prof, use_container_width=True)

            # -------------------------
            # 6. Carte Largeur × Profondeur colorée L2
            # -------------------------
            st.subheader("Carte géométrie — couleur = L2")
            fig_map = px.scatter(
                df, x="largeur_sillon_um", y="profondeur_um", color="L2_global",
                symbol="groupe", symbol_map={g: SYMBOL_MAP.get(g, "circle") for g in df["groupe"].unique()},
                color_continuous_scale="YlOrRd",
                hover_data={
                    "fichier_court": True,
                    "groupe": True,
                    "largeur_sillon_um": ":.1f",
                    "profondeur_um": ":.2f",
                    "L2_global": ":.4f",
                    "Ra_mesure": ":.4f",
                    "Rq_mesure": ":.4f",
                    "K_mesure": ":.3f",
                },
                labels={
                    "largeur_sillon_um": "Largeur (µm)",
                    "profondeur_um": "Profondeur (µm)",
                    "L2_global": "L2",
                    "fichier_court": "Fichier",
                    "groupe": "Groupe",
                },
            )
            fig_map.update_traces(marker=dict(size=12, line=dict(width=1, color="white")))
            fig_map.update_layout(height=480)
            st.plotly_chart(fig_map, use_container_width=True)

            # -------------------------
            # 7. Top kurtosis
            # -------------------------
            st.subheader("Top 5 kurtosis (usure outil)")
            top_k = df.sort_values("K_mesure", ascending=False).head(5)
            st.dataframe(
                top_k[["fichier_court", "K_mesure", "largeur_sillon_um", "profondeur_um", "L2_global", "groupe"]]
                .rename(columns={"fichier_court": "fichier"})
                .reset_index(drop=True),
                use_container_width=True,
            )
    
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