import json
import os
import sys
from pathlib import Path
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

def verify_gdrive_sync():
    """Verify that the DVC folder exists at the root and contains the repo folder."""
    creds_str = os.environ.get("DVC_GDRIVE_CREDENTIALS")
    token_str = os.environ.get("DVC_GDRIVE_TOKEN")

    if not creds_str or not token_str:
        print("⏭️ DVC_GDRIVE_CREDENTIALS or DVC_GDRIVE_TOKEN missing, skipping verification.")
        sys.exit(0)

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
        repo_folder_name = "dvc-viewer" # Hardcoded repo name for this specific check

        # 1. Verify root folder exists at the root
        query_root = f"name='{root_folder_name}' and 'root' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results_root = service.files().list(q=query_root, spaces='drive', fields='files(id, name)').execute()
        files_root = results_root.get('files', [])

        if not files_root:
            print(f"❌ Root Google Drive folder '{root_folder_name}' not found at the root level.")
            sys.exit(1)

        root_folder_id = files_root[0]['id']
        print(f"✅ Found root folder with ID: {root_folder_id}")

        # 2. Verify repository subfolder exists
        query_repo = f"name='{repo_folder_name}' and '{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results_repo = service.files().list(q=query_repo, spaces='drive', fields='files(id, name)').execute()
        files_repo = results_repo.get('files', [])

        if not files_repo:
            print(f"❌ Repository Google Drive folder '{repo_folder_name}' not found inside root folder.")
            sys.exit(1)

        print(f"✅ Found repository folder with ID: {files_repo[0]['id']}")
        sys.exit(0)

    except Exception as e:
        print(f"❌ Verification failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_gdrive_sync()
