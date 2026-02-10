#!/usr/bin/env bash
# Quick-launch dvc-viewer in dev/test mode on a different port.
# Usage: ./dvc-viewer-test.sh [PROJECT_DIR]
#   PROJECT_DIR defaults to ~/code/trail-rag

set -euo pipefail

PROJECT_DIR="${1:-$HOME/code/trail-rag}"
TEST_PORT=8687

if [[ ! -f "$PROJECT_DIR/dvc.yaml" ]]; then
    echo "❌ No dvc.yaml in $PROJECT_DIR" >&2
    exit 1
fi

# Re-install dvc-viewer from source (picks up latest code changes)
echo "📦 Installing dvc-viewer in editable mode..."
pip uninstall -y dvc-viewer 2>/dev/null || true
pip install -e .


echo "🧪 DVC Viewer TEST — port $TEST_PORT, project: $PROJECT_DIR"
echo "🌐 http://localhost:$TEST_PORT"
echo "   Press Ctrl+C to stop."
echo ""


echo "🔍 Python version: $(python --version)"
echo "🔍 dvc-viewer location: $(pip show dvc-viewer | grep Location)"


# Capture the python executable that has the package installed
PYTHON_EXEC=$(python -c "import sys; print(sys.executable)")
echo "🔍 Using Python: $PYTHON_EXEC"
echo "🔍 dvc-viewer location: $(pip show dvc-viewer | grep Location)"

cd "$PROJECT_DIR"
echo "📂 Changed directory to: $(pwd)"

# Run using the specific python executable to bypass local .python-version
"$PYTHON_EXEC" -m dvc_viewer.cli --port "$TEST_PORT"
