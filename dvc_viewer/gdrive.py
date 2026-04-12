import json
import os
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import logging

# Suppress googleapiclient warning about file_cache
logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)

def discover_dvc_folder(token_data: str) -> str | None:
    """
    Scans Google Drive for a folder named exactly 'DVC' (or DVC_GDRIVE_WORKSPACE_NAME).
    Creates it if it doesn't exist.
    Returns the Folder ID.
    """
    try:
        token_dict = json.loads(token_data)
        creds = Credentials.from_authorized_user_info(token_dict, scopes=['https://www.googleapis.com/auth/drive.file'])

        service = build('drive', 'v3', credentials=creds)

        folder_name = os.environ.get("DVC_GDRIVE_WORKSPACE_NAME", "DVC")

        # Search for folders with the exact name, not trashed
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = service.files().list(q=query, spaces='drive', fields='files(id, name, parents)').execute()
        files = results.get('files', [])

        if len(files) > 0:
            print(f"🎯 Discovered Google Drive folder '{folder_name}' with ID: {files[0]['id']}")
            return files[0]['id']
        else:
            print(f"✨ Creating Google Drive folder '{folder_name}' at the root...")
            file_metadata = {
                'name': folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = service.files().create(body=file_metadata, fields='id').execute()
            folder_id = folder.get('id')
            print(f"✅ Folder '{folder_name}' created successfully with ID: {folder_id}")
            return folder_id

    except Exception as e:
        print(f"❌ Failed to discover or create Google Drive folder: {e}")

    return None
