# app.py
# Streamlit "Studio Profilo" V2 (modulaire) : inventaire, visionneuse, manquants, comparaison, rapport
# Dépendances: streamlit numpy pandas matplotlib
# Optionnel: scipy (fill nearest), scikit-image (SSIM)
# Cache disque inventaire: nécessite parquet -> pip install pyarrow (ou fastparquet)

import os
import io
import zipfile
import tempfile

import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.figure import Figure

from core.io_xyz import read_xyz_points
from core.grid import GridResult, to_grid
from core.missing import fill_missing
from core.metrics import compare_metrics
from viz.plots import fig_heatmap, fig_mask, fig_profile_mean
from core.analysis_sillons import analyse_sillons_from_grid
from viz.plots import fig_profiles_sample, fig_profile_band_mean


# -----------------------------
# Config Streamlit
# -----------------------------
st.set_page_config(page_title="Profilo Studio V1", layout="wide")


# -----------------------------
# Helpers: scan fichiers
# -----------------------------
def find_xyz_files(root_dir: str) -> list[str]:
    xyz_files: list[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".xyz"):
                xyz_files.append(os.path.join(dirpath, fn))
    xyz_files.sort()
    return xyz_files


# -----------------------------
# Cache Streamlit (lecture xyz)
# -----------------------------
@st.cache_data(show_spinner=False)
def read_xyz_points_cached(path: str) -> pd.DataFrame:
    # FULL READ
    return read_xyz_points(path)


def load_grid_or_scatter(path: str) -> tuple[pd.DataFrame, GridResult]:
    df = read_xyz_points_cached(path)
    grid = to_grid(df)  # pivot_table inside -> FULL
    return df, grid


# -----------------------------
# Cache disque inventaire (évite rescans)
# -----------------------------
CACHE_DIR = ".cache_profilo"
CACHE_FILE = os.path.join(CACHE_DIR, "inventory_cache.parquet")


def _file_sig(path: str) -> tuple[int, int]:
    """Signature rapide d’un fichier (mtime_ns, size) pour détecter les modifs."""
    stt = os.stat(path)
    return (int(stt.st_mtime_ns), int(stt.st_size))


def _load_cache_df() -> pd.DataFrame:
    if os.path.exists(CACHE_FILE):
        try:
            return pd.read_parquet(CACHE_FILE)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _save_cache_df(df: pd.DataFrame) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = CACHE_FILE + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, CACHE_FILE)


def _clear_cache() -> None:
    if os.path.exists(CACHE_FILE):
        try:
            os.remove(CACHE_FILE)
        except Exception:
            pass


# -----------------------------
# Inventaire (FULL READ + to_grid) mais incrémental grâce au cache disque
# -----------------------------
@st.cache_data(show_spinner=False)
def build_inventory(root_dir: str) -> pd.DataFrame:
    files = find_xyz_files(root_dir)

    cache_df = _load_cache_df()
    if not cache_df.empty:
        cache_df = cache_df.drop_duplicates(subset=["chemin"], keep="last")
        cache_map = {row["chemin"]: row for _, row in cache_df.iterrows()}
    else:
        cache_map = {}

    rows: list[dict] = []
    updated_any = False

    for p in files:
        base = os.path.basename(p)
        mtime_ns, size = _file_sig(p)

        # 1) si déjà en cache et pas modifié -> on reprend direct
        if p in cache_map:
            r = cache_map[p]
            if int(r.get("mtime_ns", -1)) == mtime_ns and int(r.get("size", -1)) == size:
                rows.append(dict(r))
                continue

        # 2) sinon: FULL READ + to_grid() (donc pivot_table) comme tu veux
        df = read_xyz_points_cached(p)  # FULL
        grid = to_grid(df)              # FULL (pivot_table)

        row = {
            "fichier": base,
            "chemin": p,
            "mtime_ns": mtime_ns,
            "size": size,
            "n_points": int(df.shape[0]),
            "is_grid": bool(grid.is_grid),
            "nx": int(grid.nx),
            "ny": int(grid.ny),
            "missing_rate": float(grid.missing_rate),
            "z_min": float(np.nanmin(grid.Z)) if grid.Z.size else np.nan,
            "z_max": float(np.nanmax(grid.Z)) if grid.Z.size else np.nan,
            "z_std": float(np.nanstd(grid.Z)) if grid.Z.size else np.nan,
        }
        rows.append(row)
        cache_map[p] = row
        updated_any = True

    out = pd.DataFrame(rows)

    # 3) sauvegarde cache disque
    if updated_any:
        _save_cache_df(pd.DataFrame(list(cache_map.values())))

    return out


# -----------------------------
# UI
# -----------------------------
st.title("Profilo Studio V1")

with st.sidebar:
    st.header("Dataset")
    root = st.text_input("Chemin dossier data", value="data")

    cbtn1, cbtn2 = st.columns(2)
    with cbtn1:
        do_scan = st.button("Scanner")
    with cbtn2:
        reset_cache = st.button("Reset cache")

    if reset_cache:
        _clear_cache()
        st.cache_data.clear()
        st.session_state.inventory = None
        st.success("Cache inventaire supprimé.")

    st.divider()
    st.header("Sélection")
    st.caption("Les filtres s'appliquent à l'inventaire.")

if "inventory" not in st.session_state:
    st.session_state.inventory = None

if do_scan:
    if not os.path.exists(root):
        st.error("Chemin invalide: le dossier n'existe pas.")
    else:
        with st.spinner("Inventaire "):
            inv = build_inventory(root)
        st.session_state.inventory = inv

inv = st.session_state.inventory
if inv is None:
    st.info("Renseigne le dossier puis clique Scanner.")
    st.stop()

# Filtres (plus de OK/ERREUR)
with st.sidebar:
    grid_filter = st.multiselect("Type", options=["Grille", "Hors-grille"], default=["Grille", "Hors-grille"])
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

# Compteurs top
colA, colB, colC, colD = st.columns(4)
colA.metric("Fichiers (filtrés)", int(f.shape[0]))
colB.metric("Fichiers en grille", int((f["is_grid"] == True).sum()))
colC.metric("Fichiers hors-grille", int((f["is_grid"] == False).sum()))
colD.metric("Avec manquants (>0)", int((f["missing_rate"].fillna(0.0) > 0.0).sum()))

tabs = st.tabs(["Inventaire", "Visionneuse", "Données manquantes", "Comparer", "Analyse profils", "Analyse sillons", "Rapport"])

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
file_options = f["chemin"].tolist()

with st.sidebar:
    if len(file_options) == 0:
        st.warning("Aucun fichier dans le filtre.")
        sel_path = None
        selA = None
        selB = None
    else:
        sel_path = st.selectbox("Fichier actif (Visionneuse/Manquants)", options=file_options, index=0)
        st.caption("Comparer")
        selA = st.selectbox("Fichier A", options=file_options, index=0, key="selA")
        selB = st.selectbox("Fichier B", options=file_options, index=min(1, len(file_options) - 1), key="selB")


# -----------------------------
# Tab 2: Visionneuse
# -----------------------------
with tabs[1]:
    st.subheader("Visionneuse")
    if sel_path is None:
        st.stop()

    left, right = st.columns([2, 1])

    with st.spinner("Lecture complète et construction de la grille..."):
        dfp, gridp = load_grid_or_scatter(sel_path)

    with left:
        if gridp.is_grid:
            st.pyplot(fig_heatmap(gridp.Z, title="Surface (grille, Z)"))
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
        st.code(sel_path, language="text")
        st.write("**Résumé**")
        st.write(f"n_points: {dfp.shape[0]}")
        st.write(f"is_grid: {gridp.is_grid}")
        st.write(f"nx, ny: {gridp.nx}, {gridp.ny}")
        st.write(f"missing_rate: {gridp.missing_rate:.6f}")

        if gridp.is_grid:
            st.write(f"z_min: {float(np.nanmin(gridp.Z)):.6g}")
            st.write(f"z_max: {float(np.nanmax(gridp.Z)):.6g}")
            st.write(f"z_std: {float(np.nanstd(gridp.Z)):.6g}")
            st.write(f"z_mean: {float(np.nanmean(gridp.Z)):.6g}")
        else:
            z = dfp["z"].to_numpy()
            st.write(f"z_min: {float(np.nanmin(z)):.6g}")
            st.write(f"z_max: {float(np.nanmax(z)):.6g}")
            st.write(f"z_std: {float(np.nanstd(z)):.6g}")
            st.write(f"z_mean: {float(np.nanmean(z)):.6g}")

        if gridp.is_grid:
            img_buf = io.BytesIO()
            fig_out = fig_heatmap(gridp.Z, title="Surface (Z)")
            fig_out.savefig(img_buf, format="png")
            st.download_button("Exporter image PNG", data=img_buf.getvalue(), file_name="surface.png", mime="image/png")


# -----------------------------
# Tab 3: Données manquantes
# -----------------------------
with tabs[2]:
    st.subheader("Données manquantes")
    if sel_path is None:
        st.stop()

    with st.spinner("Lecture complète..."):
        dfm, gridm = load_grid_or_scatter(sel_path)

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
    if selA is None or selB is None:
        st.stop()

    with st.spinner("Lecture complète A..."):
        _, gA = load_grid_or_scatter(selA)
    with st.spinner("Lecture complète B..."):
        _, gB = load_grid_or_scatter(selB)

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
        st.code(selA, language="text")
        st.write("**Fichier B**")
        st.code(selB, language="text")

        fill_cmp = st.checkbox("Remplir trous avant comparaison", value=False)
        method_cmp = st.selectbox("Méthode remplissage", options=["Nearest", "Interpolate"], index=0, key="cmp_fill_method")

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


with tabs[4]:
    st.subheader("Analyse profils (1D)")
    if sel_path is None:
        st.stop()

    with st.spinner("Lecture complète..."):
        dfP, gP = load_grid_or_scatter(sel_path)

    if not gP.is_grid:
        st.warning("Analyse profils disponible seulement si fichier en grille.")
        st.stop()

    cA, cB = st.columns([1, 2])
    with cA:
        axis = st.selectbox("Direction des profils", options=["Profils en X (lignes Y)", "Profils en Y (colonnes X)"], index=0)
        axis_id = 0 if axis.startswith("Profils en X") else 1
        n_lines = st.slider("Nombre de profils affichés", 3, 30, 10, 1)
        ref0 = st.checkbox("Référence surface = 0 (soustraire max)", value=True)

    Zuse = gP.Z.copy()
    if ref0:
        Zuse = Zuse - np.nanmax(Zuse)

    with cB:
        st.pyplot(fig_profiles_sample(Zuse, n_lines=n_lines, axis=axis_id, title="Profils individuels"))
        st.pyplot(fig_profile_band_mean(Zuse, axis=axis_id, title="Profil moyen + dispersion", ref_to_zero=False))



with tabs[5]:
    st.subheader("Analyse sillons (FFT + modèle créneau)")
    if sel_path is None:
        st.stop()

    with st.spinner("Lecture complète..."):
        dfS, gS = load_grid_or_scatter(sel_path)

    if not gS.is_grid:
        st.warning("Analyse sillons disponible seulement si fichier en grille.")
        st.stop()

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
        st.stop()

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
        ax.plot(res["profil_det"], label="Profil moyen (detrend + surface=0)")
        ax.plot(res["modele_aligne"], linestyle="--", label="Modèle créneau aligné")
        ax.set_title("Profil vs modèle")
        ax.set_xlabel("index X")
        ax.set_ylabel("profondeur (µm)")

        ax.grid(True)
        ax.legend()
        fig.tight_layout()
        st.pyplot(fig)


# -----------------------------
# Tab 5: Rapport
# -----------------------------
with tabs[6]:
    st.subheader("Rapport (exports)")

    if sel_path is None:
        st.stop()

    do_fill_rep = st.checkbox("Inclure surface après remplissage trous", value=False, key="rep_fill")
    method_rep = st.selectbox("Méthode remplissage", options=["Nearest", "Interpolate"], index=0, key="rep_fill_method")

    if st.button("Générer ZIP rapport"):
        with st.spinner("Génération..."):
            with tempfile.TemporaryDirectory() as td:
                f.to_csv(os.path.join(td, "inventaire_filtre.csv"), index=False)

                dfR, gR = load_grid_or_scatter(sel_path)
                base = os.path.splitext(os.path.basename(sel_path))[0]

                if gR.is_grid:
                    statsR = pd.DataFrame(
                        [{
                            "fichier": os.path.basename(sel_path),
                            "chemin": sel_path,
                            "n_points": int(dfR.shape[0]),
                            "is_grid": bool(gR.is_grid),
                            "nx": int(gR.nx),
                            "ny": int(gR.ny),
                            "missing_rate": float(gR.missing_rate),
                            "z_min": float(np.nanmin(gR.Z)),
                            "z_max": float(np.nanmax(gR.Z)),
                            "z_mean": float(np.nanmean(gR.Z)),
                            "z_std": float(np.nanstd(gR.Z)),
                        }]
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