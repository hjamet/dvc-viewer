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

# ─── Helper: Check for conflicts ───
check_conflicts() {
    echo "  🔍 Validating update…"
    # Check for git merge markers in relevant files
    if grep -rE "<<<<<<<|=======|>>>>>>>" "$INSTALL_DIR/dvc_viewer" --include="*.py" --include="*.html" --quiet; then
        echo ""
        echo "  ❌ ERROR: Merge conflicts detected in $INSTALL_DIR"
        echo "     The installation is in a broken state with syntax errors."
        echo "     Please resolve conflicts manually in $INSTALL_DIR and run the installer again."
        echo ""
        exit 1
    fi
}

# ─── 1. Clone or update ───
if [ -d "$INSTALL_DIR" ]; then
    echo "  📦 Updating existing installation…"
    cd "$INSTALL_DIR"
    
    # Check if dirty
    if [ -n "$(git status --porcelain)" ]; then
        echo "  ⚠️  Local changes detected. Attempting to update with autostash…"
    fi

    # Try to pull. If it fails due to conflicts, our trap/set -e might catch it,
    # but we also explicitly check for markers afterward.
    if ! git pull --quiet --autostash; then
        echo ""
        echo "  ❌ ERROR: Git pull failed."
        echo "     This usually happens due to complex merge conflicts."
        echo "     Please go to $INSTALL_DIR, resolve conflicts, and try again."
        echo ""
        exit 1
    fi
    check_conflicts
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
