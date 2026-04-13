import os
import sys
import json
import subprocess
from pathlib import Path
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
import logging

logging.getLogger("googleapiclient.discovery_cache").setLevel(logging.ERROR)

def setup_and_push():
    """Sets up the DVC Google Drive remote and pushes the fake_output.txt"""
    print("⚙️ Setting up DVC Google Drive remote...")
    try:
        from dvc_viewer.cli import _setup_gdrive_sync
        from dvc_viewer.dvc_client import resolve_dvc_bin

        project_dir = Path.cwd()
        _setup_gdrive_sync(project_dir)
        dvc_bin = resolve_dvc_bin(project_dir)

        print("☁️ Pushing 'fake_output.txt' to Google Drive...")
        result = subprocess.run([dvc_bin, "push", "fake_output.txt"], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Failed to push data using '{dvc_bin} push fake_output.txt'.")
            print(f"Stdout: {result.stdout}")
            print(f"Stderr: {result.stderr}")
            sys.exit(1)
        print("✅ Push completed successfully.")
    except Exception as e:
        print(f"❌ Error during DVC remote setup or push: {e}")
        sys.exit(1)

def verify_gdrive():
    creds_str = os.environ.get("DVC_GDRIVE_CREDENTIALS")
    token_str = os.environ.get("DVC_GDRIVE_TOKEN")

    if not creds_str or not token_str:
        print("❌ DVC_GDRIVE_CREDENTIALS and/or DVC_GDRIVE_TOKEN environment variables are missing.")
        print("   Cannot verify Google Drive sync.")
        sys.exit(1)

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

        # Determine the repo folder name, default to the current working directory name
        # But if running inside github actions or cluster-ci, it's typically the repo name
        # Using pathlib to get the directory name:
        repo_folder_name = Path.cwd().name

        print(f"🔍 Searching for root folder '{root_folder_name}'...")
        query_root = f"name='{root_folder_name}' and 'root' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results_root = service.files().list(q=query_root, spaces='drive', fields='files(id, name)').execute()
        files_root = results_root.get('files', [])

        if not files_root:
            print(f"❌ Root folder '{root_folder_name}' not found on Google Drive.")
            sys.exit(1)

        root_folder_id = files_root[0]['id']
        print(f"✅ Found root folder '{root_folder_name}' (ID: {root_folder_id})")

        print(f"🔍 Searching for repository subfolder '{repo_folder_name}' inside '{root_folder_name}'...")
        query_repo = f"name='{repo_folder_name}' and '{root_folder_id}' in parents and mimeType='application/vnd.google-apps.folder' and trashed=false"
        results_repo = service.files().list(q=query_repo, spaces='drive', fields='files(id, name)').execute()
        files_repo = results_repo.get('files', [])

        if not files_repo:
            print(f"❌ Repository folder '{repo_folder_name}' not found on Google Drive inside '{root_folder_name}'.")
            sys.exit(1)

        repo_folder_id = files_repo[0]['id']
        print(f"✅ Found repository folder '{repo_folder_name}' (ID: {repo_folder_id})")

        # Now verify if any files exist in the repository folder (or its subfolders)
        # Note: DVC creates a nested structure under this folder, so we just check if it's non-empty.
        query_files = f"'{repo_folder_id}' in parents and trashed=false"
        results_files = service.files().list(q=query_files, spaces='drive', fields='files(id, name)').execute()
        files_in_repo = results_files.get('files', [])

        if not files_in_repo:
            print(f"❌ No files found inside '{repo_folder_name}' on Google Drive.")
            print("   Expected DVC to have synchronized some data (like 'fake_output.txt').")
            sys.exit(1)

        print(f"✅ Found {len(files_in_repo)} items directly inside '{repo_folder_name}'. Sync appears successful!")
        sys.exit(0)

    except Exception as e:
        print(f"❌ An error occurred while verifying Google Drive sync: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_and_push()
    verify_gdrive()
