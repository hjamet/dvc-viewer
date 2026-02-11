#!/usr/bin/env bash
# Launch dvc-viewer with a temporary local DVC project for testing.

set -euo pipefail

# Create or use a fixed test directory
TEST_DIR="$HOME/dvc-viewer-test-project"
echo "🧪 Using test project in: $TEST_DIR"

# Check if dvc.yaml exists, if not initialize
if [ ! -f "$TEST_DIR/dvc.yaml" ]; then
    echo "⚡ Initializing new test project..."
    mkdir -p "$TEST_DIR"
    cd "$TEST_DIR"
    
    # Initialize git and dvc
    git init -q
    dvc init -q
    git commit -m "Initialize DVC" --allow-empty

    # Create a dummy dvc.yaml
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

    # Create dummy script files
    touch data.txt processed.txt model.pkl metrics.json scores.json

    # Commit
    git add .
    git commit -m "Add pipeline"
    echo "✅ Project initialized."
else
    echo "✅ Found existing project."
fi

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
