from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from core.io_xyz import read_xyz_points
from core.grid import to_grid
from core.io_drive import list_xyz_files_in_folder, download_drive_file_to_temp


# -----------------------------
# Cache disque inventaire
# -----------------------------
CACHE_DIR = ".cache_profilo"
CACHE_FILE_LOCAL = os.path.join(CACHE_DIR, "inventory_cache_local.parquet")
CACHE_FILE_DRIVE = os.path.join(CACHE_DIR, "inventory_cache_drive.parquet")


def find_xyz_files(root_dir: str) -> list[str]:
    xyz_files: list[str] = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(".xyz"):
                xyz_files.append(os.path.join(dirpath, fn))
    xyz_files.sort()
    return xyz_files


def _file_sig(path: str) -> tuple[int, int]:
    stt = os.stat(path)
    return (int(stt.st_mtime_ns), int(stt.st_size))


def _load_cache_df(cache_file: str) -> pd.DataFrame:
    if os.path.exists(cache_file):
        try:
            return pd.read_parquet(cache_file)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def _save_cache_df(df: pd.DataFrame, cache_file: str) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = cache_file + ".tmp"
    df.to_parquet(tmp, index=False)
    os.replace(tmp, cache_file)


def clear_inventory_cache() -> None:
    """Supprime les fichiers parquet de cache disque."""
    for cache_file in [CACHE_FILE_LOCAL, CACHE_FILE_DRIVE]:
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
            except Exception:
                pass
    # On vide aussi le cache mémoire Streamlit des lectures xyz
    st.cache_data.clear()


def load_inventory_from_cache(source: str) -> pd.DataFrame | None:
    """
    Charge l'inventaire depuis le cache parquet persistant.
    Retourne None si le cache n'existe pas.
    Appelé au démarrage pour éviter tout rescan.
    """
    cache_file = CACHE_FILE_DRIVE if source == "drive" else CACHE_FILE_LOCAL
    df = _load_cache_df(cache_file)
    if df.empty:
        return None
    return df


@st.cache_data(show_spinner=False)
def read_xyz_points_cached(path: str) -> pd.DataFrame:
    return read_xyz_points(path)


def build_inventory_local(root_dir: str, progress_cb=None) -> pd.DataFrame:
    """
    Scan incrémental local :
    - Charge le cache parquet existant
    - Ne retraite que les fichiers nouveaux ou modifiés
    - Sauvegarde le cache mis à jour
    - progress_cb(current, total, filename) pour afficher la progression
    """
    files = find_xyz_files(root_dir)

    cache_df = _load_cache_df(CACHE_FILE_LOCAL)
    if not cache_df.empty:
        cache_df = cache_df.drop_duplicates(subset=["chemin"], keep="last")
        cache_map = {row["chemin"]: row for _, row in cache_df.iterrows()}
    else:
        cache_map = {}

    rows: list[dict] = []
    updated_any = False
    to_process = []

    # Identifier quels fichiers nécessitent un traitement
    for p in files:
        mtime_ns, size = _file_sig(p)
        if p in cache_map:
            r = cache_map[p]
            if int(r.get("mtime_ns", -1)) == mtime_ns and int(r.get("size", -1)) == size:
                rows.append(dict(r))
                continue
        to_process.append((p, mtime_ns, size))

    total_new = len(to_process)

    for i, (p, mtime_ns, size) in enumerate(to_process):
        base = os.path.basename(p)
        if progress_cb:
            progress_cb(i, total_new, base)

        df = read_xyz_points_cached(p)
        grid = to_grid(df)

        row = {
            "source": "local",
            "fichier": base,
            "chemin": p,
            "file_id": "",
            "mtime_ns": mtime_ns,
            "modifiedTime": "",
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

    out = pd.DataFrame(rows) if rows else pd.DataFrame()

    if updated_any:
        _save_cache_df(pd.DataFrame(list(cache_map.values())), CACHE_FILE_LOCAL)

    return out


def build_inventory_drive(_service: Any, folder_id: str, progress_cb=None) -> pd.DataFrame:
    """
    Scan incrémental Google Drive :
    - liste tous les .xyz
    - réutilise le cache pour les fichiers inchangés
    - retraite seulement nouveaux / modifiés
    - progress_cb(current, total, filename) pour afficher la progression
    """
    files = list_xyz_files_in_folder(_service, folder_id)

    cache_df = _load_cache_df(CACHE_FILE_DRIVE)
    if not cache_df.empty:
        cache_df = cache_df.drop_duplicates(subset=["file_id"], keep="last")
        cache_map = {row["file_id"]: row for _, row in cache_df.iterrows()}
    else:
        cache_map = {}

    rows: list[dict] = []
    updated_any = False
    current_ids = set()
    to_process = []

    for f in files:
        file_id = f["id"]
        current_ids.add(file_id)
        modified_time = f.get("modifiedTime", "")
        size = int(f.get("size", 0))

        if file_id in cache_map:
            r = cache_map[file_id]
            if str(r.get("modifiedTime", "")) == modified_time and int(r.get("size", -1)) == size:
                rows.append(dict(r))
                continue
        to_process.append(f)

    total_new = len(to_process)

    for i, f in enumerate(to_process):
        file_id = f["id"]
        drive_path = f["path"]
        base = f["name"]
        modified_time = f.get("modifiedTime", "")
        size = int(f.get("size", 0))

        if progress_cb:
            progress_cb(i, total_new, base)

        tmp_path = download_drive_file_to_temp(_service, file_id, suffix=".xyz")
        try:
            df = read_xyz_points(tmp_path)
            grid = to_grid(df)
        finally:
            try:
                os.remove(tmp_path)
            except Exception:
                pass

        row = {
            "source": "drive",
            "fichier": base,
            "chemin": drive_path,
            "file_id": file_id,
            "mtime_ns": -1,
            "modifiedTime": modified_time,
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
        cache_map[file_id] = row
        updated_any = True

    # Supprimer du cache les fichiers supprimés sur Drive
    cache_map = {k: v for k, v in cache_map.items() if k in current_ids}

    out = pd.DataFrame(rows) if rows else pd.DataFrame()

    if updated_any or len(cache_map) != len(_load_cache_df(CACHE_FILE_DRIVE)):
        _save_cache_df(pd.DataFrame(list(cache_map.values())), CACHE_FILE_DRIVE)

    return out