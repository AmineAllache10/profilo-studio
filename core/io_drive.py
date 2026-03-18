from __future__ import annotations

import io
import tempfile
from typing import Any

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload


DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]


def get_drive_service(service_account_info: dict[str, Any]):
    credentials = service_account.Credentials.from_service_account_info(
        service_account_info,
        scopes=DRIVE_SCOPES,
    )
    return build("drive", "v3", credentials=credentials)


def list_xyz_files_in_folder(service, folder_id: str) -> list[dict]:
    """
    Liste récursivement tous les fichiers .xyz contenus dans un dossier Drive.
    Retourne une liste de dict avec au minimum:
    - id
    - name
    - modifiedTime
    - size
    - parents
    """
    results: list[dict] = []

    def _walk(current_folder_id: str, prefix: str = ""):
        page_token = None
        while True:
            resp = (
                service.files()
                .list(
                    q=f"'{current_folder_id}' in parents and trashed = false",
                    fields="nextPageToken, files(id, name, mimeType, modifiedTime, size, parents)",
                    pageToken=page_token,
                    pageSize=1000,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )

            files = resp.get("files", [])
            for f in files:
                mime_type = f.get("mimeType", "")
                name = f.get("name", "")
                if mime_type == "application/vnd.google-apps.folder":
                    _walk(f["id"], prefix=f"{prefix}{name}/")
                else:
                    if name.lower().endswith(".xyz"):
                        results.append(
                            {
                                "id": f["id"],
                                "name": name,
                                "path": f"{prefix}{name}",
                                "modifiedTime": f.get("modifiedTime", ""),
                                "size": int(f.get("size", 0)) if f.get("size") else 0,
                                "parents": f.get("parents", []),
                            }
                        )

            page_token = resp.get("nextPageToken", None)
            if page_token is None:
                break

    _walk(folder_id)
    results.sort(key=lambda d: d["path"].lower())
    return results


def download_drive_file_to_temp(service, file_id: str, suffix: str = ".xyz") -> str:
    """
    Télécharge un fichier Drive dans un fichier temporaire local
    et retourne le chemin local.
    """
    request = service.files().get_media(fileId=file_id, supportsAllDrives=True)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    fh = io.FileIO(tmp.name, "wb")
    downloader = MediaIoBaseDownload(fh, request)

    done = False
    while not done:
        _, done = downloader.next_chunk()

    fh.close()
    return tmp.name