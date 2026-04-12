import json
import os
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import logging

# Suppress googleapiclient warning about file_cache
logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)

def discover_dvc_folder(token_data: str, project_name: str) -> str | None:
    """
    Scans Google Drive for a root folder named exactly 'DVC' (or DVC_GDRIVE_WORKSPACE_NAME),
    and a subfolder named after the project_name. Creates them if they don't exist.
    Returns the ID of the project subfolder.
    """
    try:
        token_dict = json.loads(token_data)
        creds = Credentials.from_authorized_user_info(token_dict, scopes=['https://www.googleapis.com/auth/drive.file'])

        service = build('drive', 'v3', credentials=creds)

        root_folder_name = os.environ.get("DVC_GDRIVE_WORKSPACE_NAME", "DVC")

        # 1. Search for root folder
        query = f"name='{root_folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false and 'root' in parents"
        results = service.files().list(q=query, spaces='drive', fields='files(id, name, parents)').execute()
        files = results.get('files', [])

        if len(files) > 0:
            root_folder_id = files[0]['id']
            print(f"🎯 Discovered root Google Drive folder '{root_folder_name}' with ID: {root_folder_id}")
        else:
            print(f"✨ Creating root Google Drive folder '{root_folder_name}'...")
            file_metadata = {
                'name': root_folder_name,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = service.files().create(body=file_metadata, fields='id').execute()
            root_folder_id = folder.get('id')
            print(f"✅ Root folder '{root_folder_name}' created successfully with ID: {root_folder_id}")

        # 2. Search for project subfolder inside root folder
        query = f"name='{project_name}' and mimeType='application/vnd.google-apps.folder' and '{root_folder_id}' in parents and trashed=false"
        results = service.files().list(q=query, spaces='drive', fields='files(id, name, parents)').execute()
        subfiles = results.get('files', [])

        if len(subfiles) > 0:
            subfolder_id = subfiles[0]['id']
            print(f"🎯 Discovered project subfolder '{project_name}' inside '{root_folder_name}' with ID: {subfolder_id}")
            return subfolder_id
        else:
            print(f"✨ Creating project subfolder '{project_name}' inside '{root_folder_name}'...")
            file_metadata = {
                'name': project_name,
                'parents': [root_folder_id],
                'mimeType': 'application/vnd.google-apps.folder'
            }
            folder = service.files().create(body=file_metadata, fields='id').execute()
            subfolder_id = folder.get('id')
            print(f"✅ Project subfolder '{project_name}' created successfully with ID: {subfolder_id}")
            return subfolder_id

    except Exception as e:
        print(f"❌ Failed to discover or create Google Drive folder structure: {e}")

    return None
