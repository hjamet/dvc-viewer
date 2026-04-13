import os
import json
import uuid
import subprocess
from pathlib import Path
import pytest
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build

def test_gdrive_sync(tmp_path):
    creds_str = os.environ.get("DVC_GDRIVE_CREDENTIALS")
    token_str = os.environ.get("DVC_GDRIVE_TOKEN")

    if not creds_str or not token_str:
        pytest.skip("GDrive credentials not provided")

    # Change to temp dir
    os.chdir(tmp_path)

    # Initialize Git
    subprocess.run(["git", "init"], check=True)

    # Initialize DVC
    subprocess.run(["dvc", "init"], check=True)

    # Step A: Setup GDrive & Push data
    # Create a unique file to verify
    test_id = str(uuid.uuid4())
    test_file = tmp_path / "data.txt"
    test_file.write_text(f"test_data_{test_id}")

    # Track with DVC
    subprocess.run(["dvc", "add", "data.txt"], check=True)

    # We need to simulate the CLI setting up the gdrive remote
    # Write gdrive credentials in the legacy oauth2client format as done by dvc_viewer
    from dvc_viewer.gdrive import convert_to_oauth2client, setup_gdrive_workspace
    creds_dict = json.loads(creds_str)
    token_dict = json.loads(token_str)

    # We want a unique folder for tests so we don't pollute the real drive
    os.environ["DVC_GDRIVE_WORKSPACE_NAME"] = "DVC_CI_TEST_WORKSPACE"

    # The repo_folder_name is determined by the project_dir.name in dvc_viewer
    # Since we use tmp_path, it's something like pytest-of-runner...
    folder_id = setup_gdrive_workspace(tmp_path, creds_str, token_str)
    assert folder_id is not None, "Failed to create/find GDrive workspace"

    viewer_dir = tmp_path / ".dvc-viewer"
    viewer_dir.mkdir(parents=True, exist_ok=True)
    legacy_token = convert_to_oauth2client(creds_dict, token_dict)
    creds_file = viewer_dir / "gdrive_token.json"
    creds_file.write_text(json.dumps(legacy_token))

    # Add DVC remote
    subprocess.run(["dvc", "remote", "add", "-d", "-f", "gdrive_remote", f"gdrive://{folder_id}"], check=True)
    subprocess.run(["dvc", "remote", "modify", "--local", "gdrive_remote", "gdrive_use_service_account", "false"], check=True)
    subprocess.run(["dvc", "remote", "modify", "--local", "gdrive_remote", "gdrive_user_credentials_file", str(creds_file)], check=True)

    # Push data
    push_res = subprocess.run(["dvc", "push"], capture_output=True, text=True)
    assert push_res.returncode == 0, f"DVC push failed: {push_res.stderr}"


    # Step B: Verify with GDrive API
    creds = Credentials(
        token=token_dict.get("token"),
        refresh_token=token_dict.get("refresh_token"),
        token_uri=token_dict.get("token_uri"),
        client_id=token_dict.get("client_id"),
        client_secret=token_dict.get("client_secret"),
        scopes=token_dict.get("scopes", ["https://www.googleapis.com/auth/drive.file"])
    )
    service = build('drive', 'v3', credentials=creds)

    # Get DVC tracked file hash
    with open("data.txt.dvc", "r") as f:
        import yaml
        dvc_data = yaml.safe_load(f)
        file_hash = dvc_data["outs"][0]["md5"]

    # In DVC gdrive remote, files are typically stored in folders based on their hash (e.g. hash[:2]/hash[2:])
    # For a simple verification, we just check that the folder is not empty
    query = f"'{folder_id}' in parents and trashed=false"
    results = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
    files = results.get('files', [])

    # Step C: Validate & Cleanup
    assert len(files) > 0, "No files found in the DVC GDrive remote folder"

    # Cleanup the test workspace
    query_root = f"name='DVC_CI_TEST_WORKSPACE' and mimeType='application/vnd.google-apps.folder' and trashed=false"
    results_root = service.files().list(q=query_root, spaces='drive', fields='files(id)').execute()
    files_root = results_root.get('files', [])

    if files_root:
        root_id = files_root[0]['id']
        service.files().delete(fileId=root_id).execute()
