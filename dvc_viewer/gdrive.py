import json
import os
from pathlib import Path
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import logging

# Suppress googleapiclient warning about file_cache
logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)

def convert_to_oauth2client(creds_data: dict, token_data: dict) -> dict:
    """Convert modern google-auth token to legacy oauth2client format for pydrive2."""
    client_id = creds_data.get("installed", {}).get("client_id") or creds_data.get("web", {}).get("client_id", "")
    client_secret = creds_data.get("installed", {}).get("client_secret") or creds_data.get("web", {}).get("client_secret", "")

    return {
        "access_token": token_data.get("token"),
        "client_id": client_id,
        "client_secret": client_secret,
        "refresh_token": token_data.get("refresh_token"),
        "token_expiry": "2038-01-01T00:00:00Z", # Dummy future date to force refresh reliance or just valid
        "token_uri": token_data.get("token_uri", "https://oauth2.googleapis.com/token"),
        "user_agent": None,
        "revoke_uri": "https://oauth2.googleapis.com/revoke",
        "id_token": None,
        "id_token_jwt": None,
        "token_response": {
            "access_token": token_data.get("token"),
            "expires_in": 3599,
            "refresh_token": token_data.get("refresh_token"),
            "scope": " ".join(token_data.get("scopes", ["https://www.googleapis.com/auth/drive.file"])),
            "token_type": "Bearer"
        },
        "scopes": token_data.get("scopes", ["https://www.googleapis.com/auth/drive.file"]),
        "token_info_uri": "https://oauth2.googleapis.com/tokeninfo",
        "invalid": False,
        "_class": "OAuth2Credentials",
        "_module": "oauth2client.client"
    }

def setup_gdrive_workspace(project_dir: Path, creds_str: str, token_str: str) -> str | None:
    """
    Sets up the GDrive workspace:
    1. Finds or creates the root DVC folder.
    2. Finds or creates the repository subfolder.
    Returns the repository folder ID.
    """
    try:
        creds_dict = json.loads(creds_str)
        token_dict = json.loads(token_str)

        client_id = creds_dict.get("installed", {}).get("client_id") or creds_dict.get("web", {}).get("client_id", "")
        client_secret = creds_dict.get("installed", {}).get("client_secret") or creds_dict.get("web", {}).get("client_secret", "")
        token_uri = creds_dict.get("installed", {}).get("token_uri") or creds_dict.get("web", {}).get("token_uri", "https://oauth2.googleapis.com/token")

        creds = Credentials(
            token=token_dict.get("token"),
            refresh_token=token_dict.get("refresh_token"),
            token_uri=token_dict.get("token_uri") or token_uri,
            client_id=token_dict.get("client_id") or client_id,
            client_secret=token_dict.get("client_secret") or client_secret,
            scopes=token_dict.get("scopes", ["https://www.googleapis.com/auth/drive.file"])
        )
        service = build('drive', 'v3', credentials=creds)

        root_folder_name = os.environ.get("DVC_GDRIVE_WORKSPACE_NAME", "DVC")
        repo_folder_name = project_dir.name

        # 1. Find or create root folder
        query_root = f"name='{root_folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results_root = service.files().list(q=query_root, spaces='drive', fields='files(id, name)').execute()
        files_root = results_root.get('files', [])

        root_folder_id = None
        if files_root:
            root_folder_id = files_root[0]['id']
            print(f"🎯 Found root Google Drive folder '{root_folder_name}' with ID: {root_folder_id}")
        else:
            print(f"📁 Creating root Google Drive folder '{root_folder_name}'...")
            file_metadata = {
                'name': root_folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = service.files().create(body=file_metadata, fields='id').execute()
            root_folder_id = folder.get('id')
            print(f"✅ Created root folder with ID: {root_folder_id}")

        # 2. Find or create repository subfolder
        query_repo = f"name='{repo_folder_name}' and '{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results_repo = service.files().list(q=query_repo, spaces='drive', fields='files(id, name)').execute()
        files_repo = results_repo.get('files', [])

        repo_folder_id = None
        if files_repo:
            repo_folder_id = files_repo[0]['id']
            print(f"🎯 Found repository Google Drive folder '{repo_folder_name}' with ID: {repo_folder_id}")
        else:
            print(f"📁 Creating repository Google Drive folder '{repo_folder_name}' under '{root_folder_name}'...")
            file_metadata = {
                'name': repo_folder_name,
                'parents': [root_folder_id],
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = service.files().create(body=file_metadata, fields='id').execute()
            repo_folder_id = folder.get('id')
            print(f"✅ Created repository folder with ID: {repo_folder_id}")

        return repo_folder_id

    except Exception as e:
        print(f"❌ Failed to setup Google Drive workspace: {e}")

    return None
