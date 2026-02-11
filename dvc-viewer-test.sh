#!/usr/bin/env bash
# Launch dvc-viewer with a temporary local DVC project for testing.

set -euo pipefail

# Create or use a fixed test directory
TEST_DIR="$HOME/dvc-viewer-test-project"
FIXTURES_DIR="$HOME/code/dvc-viewer/tests/fixtures"

# Handle arguments
FIXTURE_NAME=${1:-default}
FIXTURE_PATH="$FIXTURES_DIR/$FIXTURE_NAME"

if [ ! -d "$FIXTURE_PATH" ]; then
    echo "❌ Fixture '$FIXTURE_NAME' not found in $FIXTURES_DIR"
    echo "   Available fixtures:"
    ls "$FIXTURES_DIR"
    exit 1
fi

echo "🧪 Using fixture '$FIXTURE_NAME' for test project: $TEST_DIR"

# Always re-initialize for clean test
rm -rf "$TEST_DIR"
mkdir -p "$TEST_DIR"
cd "$TEST_DIR"

# Initialize git and dvc
git init -q
dvc init -q
git commit -m "Initialize DVC" --allow-empty

# Copy fixture files
cp -rv "$FIXTURE_PATH/"* .

# Ensure all mentioned script paths exist even if empty
# (Extract .py files from dvc.yaml)
if [ -f "dvc.yaml" ]; then
    grep -oE "[a-zA-Z0-9_/]+\.py" dvc.yaml | sort -u | xargs touch 2>/dev/null || true
fi

# Add and commit
git add .
git commit -m "Add pipeline from fixture $FIXTURE_NAME"
echo "✅ Project initialized with fixture '$FIXTURE_NAME'."

# Re-install dvc-viewer from source
echo "📦 Installing dvc-viewer in editable mode..."
cd "$HOME/code/dvc-viewer"
pip install -e . >/dev/null 2>&1

echo "🚀 Starting dvc-viewer on port 8687..."
echo "🌐 http://localhost:8687"
echo "   Press Ctrl+C to stop."

# Run server
cd "$TEST_DIR"
python -m dvc_viewer.cli --port 8687
