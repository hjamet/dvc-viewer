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
pip install -e /home/lopilo/code/dvc-viewer --quiet 2>/dev/null

echo "🧪 DVC Viewer TEST — port $TEST_PORT, project: $PROJECT_DIR"
echo "🌐 http://localhost:$TEST_PORT"
echo "   Press Ctrl+C to stop."
echo ""

cd "$PROJECT_DIR"
dvc-viewer --port "$TEST_PORT"
