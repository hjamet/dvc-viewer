#!/usr/bin/env bash
# Launch dvc-viewer with a temporary local DVC project for testing.

set -euo pipefail

# Create a temporary directory for the test project
TEST_DIR=$(mktemp -d -t dvc-viewer-test-XXXXXX)
echo "🧪 Creating test project in: $TEST_DIR"

# Clean up on exit
cleanup() {
    echo ""
    echo "🧹 Cleaning up..."
    # rm -rf "$TEST_DIR" # Optional: keep it for inspection if needed
    echo "Done."
}
trap cleanup EXIT

# Initialize git and dvc
cd "$TEST_DIR"
git init -q
dvc init -q
git commit -m "Initialize DVC" --allow-empty

# Create a dummy dvc.yaml with a multi-stage dependency graph
# Graph:
#   prepare -> train -> evaluate
#           -> process
#   standalone
cat > dvc.yaml <<EOF
stages:
  prepare:
    cmd: echo "Preparing data..." > data.txt
    outs:
      - data.txt
  
  process:
    cmd: cat data.txt > processed.txt
    deps:
      - data.txt
    outs:
      - processed.txt

  train:
    cmd: echo "Training model..." > model.pkl
    deps:
      - data.txt
    outs:
      - model.pkl
    metrics:
      - metrics.json:
          cache: false

  evaluate:
    cmd: echo "Evaluating..." > scores.json
    deps:
      - model.pkl
      - processed.txt
    metrics:
      - scores.json:
          cache: false

  standalone:
    cmd: echo "Standalone stage"
EOF

# Create dummy script files so DVC doesn't complain (though cmd is just echo)
touch data.txt processed.txt model.pkl metrics.json scores.json

# Commit dvc.yaml
git add .
git commit -m "Add pipeline"

echo "✅ Test project created."
echo "graph:"
echo "  prepare -> (process, train)"
echo "  process -> (evaluate)"
echo "  train -> (evaluate)"
echo "  standalone"
echo ""

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
