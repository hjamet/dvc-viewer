import json
import os
from google.oauth2 import service_account
from googleapiclient.discovery import build
import logging

# Suppress googleapiclient warning about file_cache
logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)

def discover_dvc_folder(creds_data: str) -> str | None:
    """
    Scans Google Drive for a folder named exactly 'DVC' (or DVC_GDRIVE_WORKSPACE_NAME).
    Returns the Folder ID if exactly one is found, otherwise asks the user or returns None.
    """
    try:
        creds_dict = json.loads(creds_data)

        # Check if this is a service account or an OAuth2 installed app credentials
        if creds_dict.get("type") == "service_account":
            creds = service_account.Credentials.from_service_account_info(
                creds_dict, scopes=['https://www.googleapis.com/auth/drive']
            )
        elif "installed" in creds_dict or "web" in creds_dict:
            print("⚠️ OAuth2 client credentials detected. DVC folder auto-discovery requires a Service Account.")
            return None
        else:
            print("⚠️ Unknown Google credentials format.")
            return None

        service = build('drive', 'v3', credentials=creds)

        folder_name = os.environ.get("DVC_GDRIVE_WORKSPACE_NAME", "DVC")

        # Search for folders with the exact name, not trashed
        query = f"name='{folder_name}' and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results = service.files().list(q=query, spaces='drive', fields='files(id, name, parents)').execute()
        files = results.get('files', [])

        if len(files) == 1:
            print(f"🎯 Discovered Google Drive folder '{folder_name}' with ID: {files[0]['id']}")
            return files[0]['id']
        elif len(files) > 1:
            print(f"⚠️ Multiple folders named '{folder_name}' found.")
            for f in files:
                print(f" - ID: {f['id']}, Name: {f['name']}")
            print("Please specify the exact DVC_GDRIVE_FOLDER_ID.")
            try:
                folder_id = input("Enter Folder ID: ").strip()
                if folder_id:
                    return folder_id
            except EOFError:
                pass
        else:
            print(f"⚠️ No folder named '{folder_name}' found in the authenticated Drive.")
            print("Please create the folder, share it with the service account, and specify DVC_GDRIVE_FOLDER_ID.")
            try:
                folder_id = input("Enter Folder ID manually (or press Enter to skip): ").strip()
                if folder_id:
                    return folder_id
            except EOFError:
                pass

    except Exception as e:
        print(f"❌ Failed to discover Google Drive folder: {e}")

    return None
