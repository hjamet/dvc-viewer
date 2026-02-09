"""
CLI entry point for dvc-viewer.

Detects dvc.yaml, sets up the project directory, and starts the web server.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


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

    # 1. Run the auto-updater to ensure hashes and dvc.yaml are consistent
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
