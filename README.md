# 🔍 DVC Viewer

> A modern, interactive web interface to visualize your DVC pipeline DAGs.

![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)

## ✨ Features

- **Interactive DAG** — Explore your pipeline as a navigable directed graph
- **Stage states** — See at a glance which stages are ✅ valid, 🔄 need rerun, or ⬜ never run
- **Precise Dependency Coloring** — 🎯 When a stage needs rerun, only the specific dependency that changed is highlighted yellow, not all dependencies. Missing files are red, unchanged files stay green.
- **Invalidation Diagnostic** — ⚠️ When a code change invalidates a stage, the console shows exactly which file changed and the **transitive import chain** responsible. Now with **Git Awareness**: alerts you if a file is not tracked by Git (the #1 cause of cross-PC invalidation).
- **Search & Filter** — Instantly find and filter stages by name or status (valid, changed, frozen, etc.).
- **Global Stage List** — View all stages in the sidebar, sorted by topological order with stable status-based prioritization (dirty stages first).
- **Progress Bar & ETA** — ⏱️ During `dvc repro`, a progress bar shows completed stages and estimated completion time based on average execution speed.
- **DVC `--keep-going` Mode** — 🚀 The "Run All" button now supports `--keep-going`. If a stage fails, the pipeline continues with independent branches, and all failed nodes are highlighted in red.
- **Frozen Status** — Visual indicators (❄️) for frozen stages.
- **One-click Navigation** — Click notifications or list items to zoom and center on any stage.
- **Smart Code Hashing** — 🧠 Only executable code changes trigger reruns. Comments, docstrings, and whitespace are ignored. Stable across Python versions (3.9+) via pure AST unparsing.
- **Symbol-Level Invalidation** — 🎯 If a script imports `foo` from `utils.py`, changing `bar` in `utils.py` will NOT invalidate the stage.
- **Click-to-inspect** — Click any node to view its command, dependencies, and outputs
- **Dark theme** — Sleek glassmorphism UI with smooth animations
- **Concurrent Robustness** — 🛡️ Zero-contention design prevents `rwlock` corruption by monitoring DVC state via the dedicated `dvc_client` module, which performs passive monitoring without triggering internal DVC write-locks.
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

## ☁️ Auto-Sync with Google Drive

DVC-Viewer can automatically pull, push, and clean up your remote DVC data on Google Drive without any manual configuration. This is ideal for seamlessly transferring data from ephemeral VMs to your personal Drive.

DVC-Viewer will automatically create a folder named `DVC` at the root of your Google Drive and push all artifacts there.

### 1. Configure an OAuth 2.0 Client
1. Go to the [Google Cloud Console](https://console.cloud.google.com/) and create a project.
2. Enable the **Google Drive API**.
3. Go to **APIs & Services > OAuth consent screen**.
    - Configure the screen and add yourself as a Test User.
    - **CRITICAL:** Once configured, set the Publishing status to **In production**. If left in "Testing", your refresh token will expire in 7 days, breaking the headless auto-push.
    - Add the scope `https://www.googleapis.com/auth/drive.file`.
4. Go to **APIs & Services > Credentials** > **Create Credentials** > **OAuth client ID**.
    - Choose **Desktop app**.
    - Download the resulting `client_secret_*.json` file.

### 2. Generate Your Tokens
Run the provided setup script on your local machine (where you have a browser) to generate your Access and Refresh tokens:

```bash
pip install google-auth-oauthlib
python scripts/setup_gdrive_auth.py
```

The script will ask you to paste your `client_secret` JSON, open a browser window for you to grant access, and then output two environment variables (`DVC_GDRIVE_CREDENTIALS` and `DVC_GDRIVE_TOKEN`).

### 3. Run DVC-Viewer
On your target machine (e.g., your VM or cluster), simply export the generated tokens before running `dvc-viewer`:

```bash
export DVC_GDRIVE_CREDENTIALS='{...}'
export DVC_GDRIVE_TOKEN='{...}'
export DVC_VIEWER_GIT_AUTO_COMMIT="false"  # (Optional) Set to false to disable git commits

dvc-viewer
```

**What it does automatically:**
- **Auto-Discovery/Creation:** Searches for a `DVC` folder at the root of your Drive and creates it if it doesn't exist.
- **Auto-Pull:** Performs `dvc pull` silently before starting any pipeline execution.
- **Auto Git Commit:** By default (unless disabled), `dvc.yaml` and `dvc.lock` changes are automatically committed to Git.
- **Auto-Push:** Performs `dvc push` in the background after a successful pipeline run.
- **Auto-Cleanup (GC):** Runs `dvc gc --cloud --workspace` in the background to delete unused remote data.

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

## 📚 Documentation Index

| Titre | Description |
|-------|-------------|
| [Index Tasks](docs/index_tasks.md) | Tâches planifiées et en cours |
| [Index Hashing](docs/index_hashing.md) | Mécanismes de hash, d'invalidation et portabilité |

## 📂 Plan du repo

```text
dvc-viewer/
├── dvc_viewer/
│   ├── dvc_client.py   # Interactions DVC (subprocess, rwlock, pgrep)
│   ├── git_client.py   # Interactions Git (log, show)
│   ├── parser.py       # Pure YAML/JSON parser & DAG builder
│   ├── server.py       # FastAPI server & API endpoints
│   ├── updater.py      # dvc.yaml enhancement (hasher injection)
│   ├── hasher.py       # AST-based code analysis
│   └── cli.py          # Entry point
├── tests/              # Test suite (46 tests)
└── docs/               # Documentation
```

## 📜 Scripts d'entrée principaux

| Commande | Description |
|----------|-------------|
| `dvc-viewer` | Lance le serveur web et l'auto-updater |
| `dvc-viewer hash` | Calcule les hashes et met à jour `dvc.yaml` |

## 🛣️ Roadmap

- [x] [Isolation des interactions DVC (dvc_client.py)](docs/tasks/fix-rwlock-corruption.md)
- [x] [Correction coloration graphe et ordre sidebar](docs/tasks/fix_dvc_graph.md)
- [x] [Mode DVC `--keep-going` (Échecs multiples)](docs/tasks/dvc-keep-going.md)
- [ ] [Refactoring Front-End : Split `index.html`](docs/tasks/refactor-frontend-split.md)
- [ ] [Refactoring Backend : Extraction `server.py`](docs/tasks/refactor-server-extraction.md)
- [ ] [Compléter la couverture de tests](docs/tasks/improve-test-coverage.md)
- [ ] Support pour les pipelines multi-projets

## 📄 License

MIT
