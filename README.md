# 🔍 DVC Viewer

> A modern, interactive web interface to visualize your DVC pipeline DAGs.

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- **Interactive DAG** — Explore your pipeline as a navigable directed graph
- **Stage states** — See at a glance which stages are ✅ valid, 🔄 need rerun, or ⬜ never run
- **Global Stage List** — View all stages and their status in a sorted sidebar list when nothing is selected
- **One-click Navigation** — Click the running notification to immediately zoom and center on the active stage
- **Click-to-inspect** — Click any node to view its command, dependencies, and outputs
- **Dark theme** — Sleek glassmorphism UI with smooth animations
- **Zero config** — Just run `dvc-viewer` inside any DVC project

## 🚀 Quick Install

```bash
curl -fsSL "https://raw.githubusercontent.com/hjamet/dvc-viewer/main/install.sh?$(date +%s)" | bash
```

This will:
1. Clone the repo to `~/.dvc-viewer`
2. Create an isolated Python virtual environment
3. Install the `dvc-viewer` command to `~/.local/bin`

## 📦 Manual Install

```bash
git clone https://github.com/hjamet/dvc-viewer.git ~/.dvc-viewer
cd ~/.dvc-viewer
python3 -m venv .venv
source .venv/bin/activate
pip install .
```

## 🎯 Usage

```bash
cd /path/to/your/dvc-project
dvc-viewer
```

The web interface opens automatically at [http://localhost:8686](http://localhost:8686).

### Options

| Flag | Description | Default |
|------|-------------|---------|
| `--port` | Server port | `8686` |
| `--no-open` | Don't auto-open browser | `false` |

## 🛠 Requirements

- Python ≥ 3.9
- DVC installed and accessible in `$PATH`
- A project with a `dvc.yaml` file

## 🪝 Hooks

DVC-Viewer supports **project-level hooks** — scripts that run automatically after specific operations.

### `post_hash` hook

Create `.dvc-viewer/hooks/post_hash.py` in your project. It runs after every `dvc-viewer hash`.

```bash
mkdir -p .dvc-viewer/hooks
cat > .dvc-viewer/hooks/post_hash.py << 'EOF'
"""Example post_hash hook — runs after code hashing."""
print("✅ Post-hash hook executed!")
EOF
```

**Python resolution** — the hook runs with the project's Python, resolved in order:

1. Active virtualenv (`$VIRTUAL_ENV`)
2. Project venv (`.venv/`, `venv/`, `.env/`, `env/`)
3. System `python3`

**Error handling** — if the hook fails, a warning is printed but `dvc-viewer hash` always succeeds. Hooks never block the pipeline.

## 📄 License

MIT
