"""Tests for Google Drive auto-discovery logic."""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from dvc_viewer.gdrive import _get_drive_service, _find_folder, _create_folder, discover_or_create_dvc_folder

@pytest.fixture
def fake_creds_data():
    return json.dumps({
        "type": "service_account",
        "project_id": "fake_project",
        "private_key_id": "fake_key_id",
        "private_key": "fake_private_key",
        "client_email": "fake@fake.com",
        "client_id": "12345",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/fake@fake.com"
    })

def test_find_folder():
    mock_service = MagicMock()
    mock_service.files.return_value.list.return_value.execute.return_value = {
        "files": [{"id": "folder_123", "name": "DVC"}]
    }

    folder_id = _find_folder(mock_service, "DVC")
    assert folder_id == "folder_123"

    mock_service.files.return_value.list.assert_called_once()
    kwargs = mock_service.files.return_value.list.call_args[1]
    assert "name = 'DVC'" in kwargs["q"]

def test_find_folder_not_found():
    mock_service = MagicMock()
    mock_service.files.return_value.list.return_value.execute.return_value = {}

    folder_id = _find_folder(mock_service, "DVC")
    assert folder_id is None

def test_create_folder():
    mock_service = MagicMock()
    mock_service.files.return_value.create.return_value.execute.return_value = {"id": "new_folder_456"}

    folder_id = _create_folder(mock_service, "my_project", parent_id="parent_123")
    assert folder_id == "new_folder_456"

    mock_service.files.return_value.create.assert_called_once()
    kwargs = mock_service.files.return_value.create.call_args[1]
    assert kwargs["body"]["name"] == "my_project"
    assert kwargs["body"]["parents"] == ["parent_123"]

@patch("dvc_viewer.gdrive._get_drive_service")
@patch("dvc_viewer.gdrive._find_folder")
@patch("dvc_viewer.gdrive._create_folder")
def test_discover_or_create_dvc_folder_existing(mock_create, mock_find, mock_get_service, fake_creds_data, tmp_path):
    # Mocking service
    mock_service = MagicMock()
    mock_get_service.return_value = mock_service

    # Root folder exists, project folder exists
    mock_find.side_effect = ["root_folder_id", "project_folder_id"]

    folder_id = discover_or_create_dvc_folder(fake_creds_data, tmp_path)

    assert folder_id == "project_folder_id"
    assert mock_find.call_count == 2
    mock_create.assert_not_called()

@patch("dvc_viewer.gdrive._get_drive_service")
@patch("dvc_viewer.gdrive._find_folder")
@patch("dvc_viewer.gdrive._create_folder")
def test_discover_or_create_dvc_folder_create_new(mock_create, mock_find, mock_get_service, fake_creds_data, tmp_path):
    mock_service = MagicMock()
    mock_get_service.return_value = mock_service

    # Root folder exists, project folder DOES NOT exist
    mock_find.side_effect = ["root_folder_id", None]
    mock_create.return_value = "new_project_folder_id"

    folder_id = discover_or_create_dvc_folder(fake_creds_data, tmp_path)

    assert folder_id == "new_project_folder_id"
    assert mock_find.call_count == 2
    mock_create.assert_called_once()

@patch("dvc_viewer.gdrive._get_drive_service")
@patch("dvc_viewer.gdrive._find_folder")
@patch("dvc_viewer.gdrive._create_folder")
def test_discover_or_create_dvc_folder_no_root(mock_create, mock_find, mock_get_service, fake_creds_data, tmp_path):
    mock_service = MagicMock()
    mock_get_service.return_value = mock_service

    # Root folder DOES NOT exist
    mock_find.return_value = None

    folder_id = discover_or_create_dvc_folder(fake_creds_data, tmp_path)

    assert folder_id is None
    assert mock_find.call_count == 1
    mock_create.assert_not_called()
