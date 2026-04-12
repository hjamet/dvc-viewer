"""
CLI entry point for dvc-viewer.

Detects dvc.yaml, sets up the project directory, and starts the web server.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


import json
import subprocess

def _setup_gdrive_sync(project_dir: Path) -> None:
    """Configure DVC remote if GDrive environment variables are present."""
    creds_data = os.environ.get("DVC_GDRIVE_CREDENTIALS")
    token_data = os.environ.get("DVC_GDRIVE_TOKEN")
    folder_id = os.environ.get("DVC_GDRIVE_FOLDER_ID")

    if not creds_data or not token_data:
        return

    if not folder_id:
        print("🔍 Searching for Google Drive DVC workspace...")
        try:
            from .gdrive import discover_dvc_folder
            folder_id = discover_dvc_folder(token_data, project_dir.name)
            if folder_id:
                os.environ["DVC_GDRIVE_FOLDER_ID"] = folder_id
            else:
                print("❌ Could not determine or create Google Drive Folder ID. Auto-Sync disabled.")
                return
        except ImportError:
            print("⚠️ Google API client libraries not installed. Please install them to use Drive auto-discovery.")
            return

    print("☁️  Configuring Google Drive Auto-Sync...")
    try:
        creds_json = json.loads(creds_data)
        token_json = json.loads(token_data)

        # Convert the modern google-auth-oauthlib format into the legacy oauth2client format
        # required by pydrive2 and DVC's internal `gdrive_user_credentials_file`.
        legacy_token = {
            "access_token": token_json.get("token"),
            "client_id": token_json.get("client_id"),
            "client_secret": token_json.get("client_secret"),
            "refresh_token": token_json.get("refresh_token"),
            "token_expiry": token_json.get("expiry"),
            "token_uri": token_json.get("token_uri"),
            "user_agent": None,
            "revoke_uri": "https://oauth2.googleapis.com/revoke",
            "id_token": None,
            "id_token_jwt": None,
            "token_response": {
                "access_token": token_json.get("token"),
                "expires_in": 3599,
                "refresh_token": token_json.get("refresh_token"),
                "scope": "https://www.googleapis.com/auth/drive.file",
                "token_type": "Bearer"
            },
            "scopes": token_json.get("scopes"),
            "token_info_uri": "https://oauth2.googleapis.com/tokeninfo",
            "invalid": False,
            "_class": "OAuth2Credentials",
            "_module": "oauth2client.client"
        }

        # Write the JSON locally inside .dvc-viewer for DVC to use
        viewer_dir = project_dir / ".dvc-viewer"
        viewer_dir.mkdir(parents=True, exist_ok=True)
        creds_file = viewer_dir / "gdrive_credentials.json"
        token_file = viewer_dir / "gdrive_token.json"

        creds_file.write_text(creds_data, encoding="utf-8")
        token_file.write_text(json.dumps(legacy_token), encoding="utf-8")

        # Add to .gitignore
        gitignore = project_dir / ".gitignore"
        ignore_line = ".dvc-viewer/\n"
        if gitignore.exists():
            git_content = gitignore.read_text(encoding="utf-8")
            if ".dvc-viewer" not in git_content:
                gitignore.write_text(git_content + "\n" + ignore_line, encoding="utf-8")
        else:
            gitignore.write_text(ignore_line, encoding="utf-8")

    except json.JSONDecodeError:
        print("❌ Invalid JSON in DVC_GDRIVE_CREDENTIALS or DVC_GDRIVE_TOKEN")
        return

    from .dvc_client import resolve_dvc_bin
    dvc_bin = resolve_dvc_bin(project_dir)

    # 1. Add remote
    subprocess.run([dvc_bin, "remote", "add", "-d", "-f", "gdrive_remote", f"gdrive://{folder_id}"],
                   cwd=str(project_dir), capture_output=True)

    # 2. Configure OAuth Desktop locally
    app_type = "installed" if "installed" in creds_json else "web"
    client_id = creds_json.get(app_type, {}).get("client_id")
    client_secret = creds_json.get(app_type, {}).get("client_secret")

    if not client_id or not client_secret:
        print("❌ Invalid OAuth credentials format.")
        return

    subprocess.run([dvc_bin, "remote", "modify", "--local", "gdrive_remote", "gdrive_use_service_account", "false"], cwd=str(project_dir), capture_output=True)
    subprocess.run([dvc_bin, "remote", "modify", "--local", "gdrive_remote", "gdrive_client_id", client_id], cwd=str(project_dir), capture_output=True)
    subprocess.run([dvc_bin, "remote", "modify", "--local", "gdrive_remote", "gdrive_client_secret", client_secret], cwd=str(project_dir), capture_output=True)
    subprocess.run([dvc_bin, "remote", "modify", "--local", "gdrive_remote", "gdrive_user_credentials_file", str(token_file)], cwd=str(project_dir), capture_output=True)

    print("✅ Google Drive remote 'gdrive_remote' configured as default with OAuth.")

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="dvc-viewer",
        description="🔍 Interactive web visualization for DVC pipelines",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8686,
        help="Port to serve the web interface on (default: 8686)",
    )
    # Subcommand for internal hashing
    subparsers = parser.add_subparsers(dest="command", help="Subcommands")
    hash_parser = subparsers.add_parser("hash", help="Compute hashes internal command")

    args = parser.parse_args()

    # Handle hash command
    if args.command == "hash":
        project_dir = Path.cwd()
        from .updater import update_dvc_yaml
        # Just run the update logic (which computes hashes)
        update_dvc_yaml(project_dir)
        sys.exit(0)

    project_dir = Path.cwd()
    dvc_yaml = project_dir / "dvc.yaml"

    if not dvc_yaml.exists():
        print("❌ No dvc.yaml found in the current directory.", file=sys.stderr)
        print("   Run this command from inside a DVC project.", file=sys.stderr)
        sys.exit(1)

    # 1. Setup Auto-Sync Drive Configuration if available
    _setup_gdrive_sync(project_dir)

    # 2. Run the auto-updater to ensure hashes and dvc.yaml are consistent
    from .updater import update_dvc_yaml
    update_dvc_yaml(project_dir)

    # Set project dir for the server to pick up
    os.environ["DVC_VIEWER_PROJECT_DIR"] = str(project_dir)

    print(f"🔍 DVC Viewer — reading pipeline from {dvc_yaml}")
    print(f"🌐 Starting server at http://localhost:{args.port}")
    print("   Press Ctrl+C to stop.\n")

    import uvicorn
    uvicorn.run(
        "dvc_viewer.server:app",
        host="0.0.0.0",
        port=args.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
