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
    for cache_file in [CACHE_FILE_LOCAL, CACHE_FILE_DRIVE]:
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
            except Exception:
                pass
    st.cache_data.clear()


@st.cache_data(show_spinner=False)
def read_xyz_points_cached(path: str) -> pd.DataFrame:
    return read_xyz_points(path)


@st.cache_data(show_spinner=False)
def build_inventory_local(root_dir: str) -> pd.DataFrame:
    files = find_xyz_files(root_dir)

    cache_df = _load_cache_df(CACHE_FILE_LOCAL)
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

        if p in cache_map:
            r = cache_map[p]
            if int(r.get("mtime_ns", -1)) == mtime_ns and int(r.get("size", -1)) == size:
                rows.append(dict(r))
                continue

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

    out = pd.DataFrame(rows)

    if updated_any:
        _save_cache_df(pd.DataFrame(list(cache_map.values())), CACHE_FILE_LOCAL)

    return out


@st.cache_data(show_spinner=False)
def build_inventory_drive(_service: Any, folder_id: str) -> pd.DataFrame:
    """
    Scan incrémental Google Drive :
    - liste tous les .xyz
    - réutilise le cache pour les fichiers inchangés
    - retraite seulement nouveaux / modifiés
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

    for f in files:
        file_id = f["id"]
        current_ids.add(file_id)

        drive_path = f["path"]
        base = f["name"]
        modified_time = f.get("modifiedTime", "")
        size = int(f.get("size", 0))

        if file_id in cache_map:
            r = cache_map[file_id]
            if str(r.get("modifiedTime", "")) == modified_time and int(r.get("size", -1)) == size:
                rows.append(dict(r))
                continue

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

    # on supprime du cache les fichiers qui n'existent plus sur Drive
    cache_map = {k: v for k, v in cache_map.items() if k in current_ids}

    out = pd.DataFrame(rows)

    if updated_any or len(cache_map) != len(_load_cache_df(CACHE_FILE_DRIVE)):
        _save_cache_df(pd.DataFrame(list(cache_map.values())), CACHE_FILE_DRIVE)

    return out