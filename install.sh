#!/usr/bin/env bash
# ─────────────────────────────────────────────
#  DVC Viewer — One-line installer
#  Usage: curl -fsSL <url>/install.sh | bash
# ─────────────────────────────────────────────
set -euo pipefail

INSTALL_DIR="$HOME/.dvc-viewer"
BIN_DIR="$HOME/.local/bin"
REPO_URL="https://github.com/hjamet/dvc-viewer.git"

echo ""
echo "  🔍 DVC Viewer — Installer"
echo "  ─────────────────────────"
echo ""

# ─── 1. Clone or update ───
if [ -d "$INSTALL_DIR" ]; then
    echo "  📦 Updating existing installation…"
    
    # Try to determine the default branch dynamically
    cd "$INSTALL_DIR"
    git fetch --quiet origin
    DEFAULT_BRANCH=$(git remote show origin | sed -n '/HEAD branch/s/.*: //p' || echo "main")
    
    # Attempt a forceful reset to remote state.
    if ! git reset --hard "origin/$DEFAULT_BRANCH" --quiet; then
        echo "  ⚠️  Update failed, re-installing from scratch…"
        rm -rf "$INSTALL_DIR"
    else
        # Remove untracked files to ensure we match remote exactly
        git clean -fd --quiet
    fi
fi

if [ ! -d "$INSTALL_DIR" ]; then
    echo "  📦 Cloning repository…"
    git clone --quiet "$REPO_URL" "$INSTALL_DIR"
fi
cd "$INSTALL_DIR"

# ─── 2. Create or validate venv ───
RECREATE_VENV=false
if [ -d ".venv" ]; then
    # Check if venv is healthy (can run python)
    if ! .venv/bin/python --version >/dev/null 2>&1; then
        echo "  ⚠️  Virtual environment is broken, recreating…"
        RECREATE_VENV=true
    fi
fi

if [ ! -d ".venv" ] || [ "$RECREATE_VENV" = true ]; then
    rm -rf ".venv"
    echo "  🐍 Creating Python virtual environment…"
    python3 -m venv .venv
fi

# ─── 3. Install package ───
echo "  📥 Installing dependencies…"
.venv/bin/pip install --quiet --upgrade pip
.venv/bin/pip install --quiet --upgrade --upgrade-strategy eager -e .

# ─── 4. Symlink binary ───
mkdir -p "$BIN_DIR"
ln -sf "$INSTALL_DIR/.venv/bin/dvc-viewer" "$BIN_DIR/dvc-viewer"

echo ""
echo "  ✅ Installed successfully!"
echo ""
echo "  Usage:  cd /path/to/dvc-project && dvc-viewer"
echo ""

# ─── 5. Check PATH ───
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo "  ⚠️  $BIN_DIR is not in your PATH."
    echo "     Add this to your shell profile (~/.bashrc or ~/.zshrc):"
    echo ""
    echo "       export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo ""
fi
