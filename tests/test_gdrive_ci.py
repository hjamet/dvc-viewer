import os
from pathlib import Path
from dvc_viewer.gdrive import setup_gdrive_workspace

def test_setup_gdrive_workspace_creates_folders(tmp_path: Path):
    """
    Integration test to verify that setup_gdrive_workspace correctly communicates
    with the real Google Drive API and creates/finds the root 'DVC' folder
    and the repository subfolder.
    """
    creds_str = os.environ.get("DVC_GDRIVE_CREDENTIALS")
    token_str = os.environ.get("DVC_GDRIVE_TOKEN")

    if not creds_str or not token_str:
        print("❌ DVC_GDRIVE_CREDENTIALS and DVC_GDRIVE_TOKEN must be set to run Google Drive integration tests.")
        import pytest
        pytest.skip("Missing DVC_GDRIVE_CREDENTIALS or DVC_GDRIVE_TOKEN")

    # Use a unique name for the mock project directory
    project_dir = tmp_path / "test_dvc_viewer_repo_12345"
    project_dir.mkdir()

    # Call the function
    repo_folder_id = setup_gdrive_workspace(project_dir, creds_str, token_str)

    # Assert that a folder ID was returned (meaning it was found or created)
    assert repo_folder_id is not None
    assert isinstance(repo_folder_id, str)
    assert len(repo_folder_id) > 0

    # We call it twice to ensure the "find" logic works and doesn't crash
    repo_folder_id_second = setup_gdrive_workspace(project_dir, creds_str, token_str)

    # It should return the exact same folder ID the second time
    assert repo_folder_id == repo_folder_id_second
