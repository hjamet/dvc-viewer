"""
Google Drive auto-discovery logic for DVC remote configuration.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


logger = logging.getLogger(__name__)


def _get_drive_service(creds_data: str):
    """Create a Google Drive API service client from service account credentials JSON."""
    try:
        creds_dict = json.loads(creds_data)
        creds = service_account.Credentials.from_service_account_info(
            creds_dict,
            scopes=["https://www.googleapis.com/auth/drive"]
        )
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return service
    except Exception as e:
        logger.error(f"Failed to authenticate with Google Drive using the provided credentials: {e}")
        return None


def _find_folder(service, name: str, parent_id: Optional[str] = None) -> Optional[str]:
    """Find a folder by name, optionally within a specific parent folder."""
    query = f"name = '{name}' and mimeType = 'application/vnd.google-apps.folder' and trashed = false"
    if parent_id:
        query += f" and '{parent_id}' in parents"

    try:
        results = service.files().list(
            q=query,
            spaces="drive",
            fields="files(id, name)"
        ).execute()

        items = results.get("files", [])
        if not items:
            return None

        # Return the first matching folder
        return items[0]["id"]
    except HttpError as error:
        logger.error(f"An error occurred while searching for folder '{name}': {error}")
        return None


def _create_folder(service, name: str, parent_id: Optional[str] = None) -> Optional[str]:
    """Create a new folder."""
    file_metadata = {
        "name": name,
        "mimeType": "application/vnd.google-apps.folder"
    }
    if parent_id:
        file_metadata["parents"] = [parent_id]

    try:
        folder = service.files().create(
            body=file_metadata,
            fields="id"
        ).execute()
        return folder.get("id")
    except HttpError as error:
        logger.error(f"An error occurred while creating folder '{name}': {error}")
        return None


def discover_or_create_dvc_folder(creds_data: str, project_dir: Path) -> Optional[str]:
    """
    Discover or create the DVC Google Drive remote folder automatically.

    1. Looks for a root folder named "DVC" (or the value of DVC_GDRIVE_WORKSPACE_NAME).
    2. If the root folder is found, looks for a sub-folder matching the project directory name.
    3. If the sub-folder doesn't exist, it creates it.
    4. Returns the ID of the sub-folder.
    """
    service = _get_drive_service(creds_data)
    if not service:
        print("❌ Could not initialize Google Drive API client.")
        return None

    workspace_name = os.environ.get("DVC_GDRIVE_WORKSPACE_NAME", "DVC")
    project_name = project_dir.name

    print(f"🔍 Looking for Google Drive workspace folder: '{workspace_name}'...")
    root_folder_id = _find_folder(service, workspace_name)

    if not root_folder_id:
        print(f"❌ Could not find a folder named '{workspace_name}' accessible by the Service Account.")
        print(f"   Please make sure to create a folder named '{workspace_name}' in Google Drive")
        print("   and share it with the email address of your Service Account.")
        return None

    print(f"✅ Found workspace folder '{workspace_name}' (ID: {root_folder_id})")
    print(f"🔍 Looking for project sub-folder: '{project_name}'...")

    project_folder_id = _find_folder(service, project_name, parent_id=root_folder_id)

    if project_folder_id:
        print(f"✅ Found project folder '{project_name}' (ID: {project_folder_id})")
        return project_folder_id

    print(f"📁 Project folder '{project_name}' not found. Creating it...")
    project_folder_id = _create_folder(service, project_name, parent_id=root_folder_id)

    if project_folder_id:
        print(f"✅ Created project folder '{project_name}' (ID: {project_folder_id})")
        return project_folder_id
    else:
        print(f"❌ Failed to create project folder '{project_name}' inside '{workspace_name}'.")
        print("   Please check if the Service Account has 'Editor' or 'Writer' permissions on the workspace folder.")
        return None
