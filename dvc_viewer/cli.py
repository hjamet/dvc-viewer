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
import ast

def _parse_json_str(s: str) -> dict:
    """Robust JSON parser that falls back to ast.literal_eval for python dict strings."""
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return ast.literal_eval(s)

def _setup_gdrive_sync(project_dir: Path) -> None:
    """Configure DVC remote if GDrive environment variables are present."""
    creds_data = os.environ.get("DVC_GDRIVE_CREDENTIALS")
    token_data = os.environ.get("DVC_GDRIVE_TOKEN")

    if not creds_data or not token_data:
        return

    from .dvc_client import resolve_dvc_bin, install_dvc_gdrive
    dvc_bin = resolve_dvc_bin(project_dir)

    install_dvc_gdrive(dvc_bin)

    # Check if remote already exists
    remote_list = subprocess.run([dvc_bin, "remote", "list"], cwd=str(project_dir), capture_output=True, text=True)
    remote_exists = False
    existing_folder_id = None
    if remote_list.returncode == 0:
        for line in remote_list.stdout.splitlines():
            if line.startswith("gdrive_remote\t"):
                remote_exists = True
                parts = line.split("\t")
                if len(parts) > 1 and parts[1].startswith("gdrive://"):
                    existing_folder_id = parts[1][9:]
                break

    folder_id = existing_folder_id

    if not remote_exists:
        print("🔍 Searching for Google Drive DVC workspace...")
        try:
            from .gdrive import setup_gdrive_workspace, convert_to_oauth2client
            folder_id = setup_gdrive_workspace(project_dir, creds_data, token_data)
            if not folder_id:
                print("❌ Could not determine Google Drive Folder ID. Auto-Sync disabled.")
                return
        except ImportError:
            print("⚠️ Google API client libraries not installed. Please install them to use Drive auto-discovery.")
            return

    if folder_id:
        os.environ["DVC_GDRIVE_FOLDER_ID"] = folder_id

    print("☁️  Configuring Google Drive Auto-Sync...")

    try:
        # Parse JSONs
        creds_dict = _parse_json_str(creds_data)
        token_dict = _parse_json_str(token_data)

        # We need the credentials file to be persistent across git commands and DVC runs.
        # We will write it inside the .dvc-viewer config dir, but add it to .gitignore
        viewer_dir = project_dir / ".dvc-viewer"
        viewer_dir.mkdir(parents=True, exist_ok=True)

        from .gdrive import convert_to_oauth2client
        # Write legacy oauth2client token for pydrive2
        legacy_token = convert_to_oauth2client(creds_dict, token_dict)
        creds_file = viewer_dir / "gdrive_token.json"
        creds_file.write_text(json.dumps(legacy_token, indent=2), encoding="utf-8")

        # Ensure .dvc-viewer is in .gitignore to prevent accidental commits
        gitignore = project_dir / ".gitignore"
        ignore_line = ".dvc-viewer/\n"
        if gitignore.exists():
            content = gitignore.read_text(encoding="utf-8")
            if ".dvc-viewer" not in content:
                gitignore.write_text(content + "\n" + ignore_line, encoding="utf-8")
        else:
            gitignore.write_text(ignore_line, encoding="utf-8")

    except (json.JSONDecodeError, ValueError, SyntaxError):
        print("❌ Invalid JSON in DVC_GDRIVE_CREDENTIALS or DVC_GDRIVE_TOKEN")
        return

    # 1. Add remote if it doesn't exist
    if not remote_exists and folder_id:
        subprocess.run([dvc_bin, "remote", "add", "-d", "-f", "gdrive_remote", f"gdrive://{folder_id}"],
                       cwd=str(project_dir), capture_output=True)

    # 2. Configure OAuth token locally
    subprocess.run([dvc_bin, "remote", "modify", "--local", "gdrive_remote", "gdrive_use_service_account", "false"],
                   cwd=str(project_dir), capture_output=True)
    subprocess.run([dvc_bin, "remote", "modify", "--local", "gdrive_remote", "gdrive_user_credentials_file", str(creds_file)],
                   cwd=str(project_dir), capture_output=True)

    print("✅ Google Drive remote 'gdrive_remote' configured as default.")

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
