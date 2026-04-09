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
    creds_data = os.environ.get("DVC_GDRIVE_CREDENTIALS_DATA")
    folder_id = os.environ.get("DVC_GDRIVE_FOLDER_ID")

    if not creds_data or not folder_id:
        return

    print("☁️  Configuring Google Drive Auto-Sync...")
    # Use a temp file for credentials to avoid committing it
    import tempfile

    try:
        # Check if JSON is valid
        json.loads(creds_data)

        # Create a temp directory for the credentials file
        temp_dir = Path(tempfile.mkdtemp(prefix="dvc-viewer-"))
        creds_file = temp_dir / "gdrive_credentials.json"
        creds_file.write_text(creds_data, encoding="utf-8")

        # Register an exit handler to cleanup the temp directory if possible
        import atexit
        def cleanup_creds():
            try:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
        atexit.register(cleanup_creds)

    except json.JSONDecodeError:
        print("❌ Invalid JSON in DVC_GDRIVE_CREDENTIALS_DATA")
        return

    from .dvc_client import resolve_dvc_bin
    dvc_bin = resolve_dvc_bin(project_dir)

    # 1. Add remote
    subprocess.run([dvc_bin, "remote", "add", "-d", "-f", "gdrive_remote", f"gdrive://{folder_id}"],
                   cwd=str(project_dir), capture_output=True)

    # 2. Configure Service Account locally
    subprocess.run([dvc_bin, "remote", "modify", "--local", "gdrive_remote", "gdrive_use_service_account", "true"],
                   cwd=str(project_dir), capture_output=True)
    subprocess.run([dvc_bin, "remote", "modify", "--local", "gdrive_remote", "gdrive_service_account_json_file_path", str(creds_file)],
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
