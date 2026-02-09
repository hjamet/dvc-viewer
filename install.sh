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
    cd "$INSTALL_DIR"
    git pull --quiet
else
    echo "  📦 Cloning repository…"
    git clone --quiet "$REPO_URL" "$INSTALL_DIR"
    cd "$INSTALL_DIR"
fi

# ─── 2. Create venv ───
if [ ! -d "$INSTALL_DIR/.venv" ]; then
    echo "  🐍 Creating Python virtual environment…"
    python3 -m venv "$INSTALL_DIR/.venv"
fi

# ─── 3. Install package ───
echo "  📥 Installing dependencies…"
"$INSTALL_DIR/.venv/bin/pip" install --quiet --upgrade pip
"$INSTALL_DIR/.venv/bin/pip" install --quiet "$INSTALL_DIR"

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
