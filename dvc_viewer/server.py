"""
FastAPI web server for DVC Viewer.

Serves the static frontend and exposes the pipeline API,
plus file inspection endpoints for CSV, images, PDFs, JSON, etc.
"""

from __future__ import annotations

import asyncio
import csv
import io
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
import yaml

from .parser import (
    build_pipeline,
    build_pipeline_at_commit,
    pipeline_to_dict,
    mark_stage_complete,
    mark_stage_failed,
    mark_stage_started,
)
from .git_client import get_commit_list
from .dvc_client import (
    resolve_dvc_bin,
    run_dvc_repro,
    start_dvc_repro,
    stop_dvc_process,
)

app = FastAPI(title="DVC Viewer", version="0.1.0")

# The project dir is set at startup via environment variable
_project_dir: str = os.environ.get("DVC_VIEWER_PROJECT_DIR", os.getcwd())

# Resolve the DVC binary path (checks system PATH, project .venv, our venv)
_dvc_bin = resolve_dvc_bin(_project_dir)

# Initialize status cache at startup to avoid "blind" fallback during active runs
try:
    build_pipeline(_project_dir)
except Exception:
    pass
 

# Track the running dvc repro process so we can stop it
_running_proc: subprocess.Popen | None = None

# TTL cache for pipeline data to avoid running dvc status on every 1s poll
_pipeline_cache: dict | None = None
_pipeline_cache_time: float = 0.0
_PIPELINE_CACHE_TTL: float = 5.0  # seconds

# Serve static files
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# Supported file categories
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".ico"}
_CSV_EXTENSIONS = {".csv", ".tsv"}
_PDF_EXTENSIONS = {".pdf"}
_TEXT_EXTENSIONS = {".json", ".yaml", ".yml", ".txt", ".md", ".log", ".jsonl"}


def _safe_resolve(rel_path: str, require_exists: bool = True) -> Path | None:
    """Resolve a relative path safely within the project directory."""
    project = Path(_project_dir).resolve()
    target = (project / rel_path).resolve()
    # Prevent path traversal
    if not str(target).startswith(str(project)):
        return None
    if require_exists and (not target.exists() or not target.is_file()):
        return None
    return target


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main HTML page."""
    index_path = _static_dir / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/api/pipeline", response_class=JSONResponse)
async def get_pipeline(force_refresh: bool = Query(False, description="Bypass cache")):
    """Return the parsed DVC pipeline as JSON."""
    global _pipeline_cache, _pipeline_cache_time
    now = time.monotonic()
    if not force_refresh and _pipeline_cache and (now - _pipeline_cache_time) < _PIPELINE_CACHE_TTL:
        return JSONResponse(content=_pipeline_cache)
    try:
        pipeline = await asyncio.to_thread(build_pipeline, _project_dir)
        data = pipeline_to_dict(pipeline)
        _pipeline_cache = data
        _pipeline_cache_time = now
        return JSONResponse(content=data)
    except FileNotFoundError as e:
        return JSONResponse(content={"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/commits", response_class=JSONResponse)
async def list_commits(limit: int = Query(100, description="Max commits to return")):
    """Return the git commit history for the project repository."""
    try:
        commits = get_commit_list(_project_dir, limit=limit)
        return JSONResponse(content={"commits": commits})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/pipeline/at-commit", response_class=JSONResponse)
async def get_pipeline_at_commit(
    commit: str = Query(..., description="Git commit hash"),
):
    """Return the parsed DVC pipeline as it was at a specific git commit."""
    # Validate commit hash (alphanumeric only)
    if not re.match(r'^[a-f0-9]+$', commit):
        return JSONResponse(content={"error": "Invalid commit hash"}, status_code=400)
    try:
        pipeline = build_pipeline_at_commit(_project_dir, commit)
        data = pipeline_to_dict(pipeline)
        # Add commit info to the response
        data["commit"] = commit
        data["is_history"] = True
        return JSONResponse(content=data)
    except FileNotFoundError as e:
        return JSONResponse(content={"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/file/info")
async def file_info(path: str = Query(..., description="Relative file path")):
    """Return metadata about a file: type, size, exists."""
    # Allow resolution even if missing for metadata purposes in history view
    target = _safe_resolve(path, require_exists=False)
    if target is None:
        return JSONResponse(content={"error": "File not found", "exists": False}, status_code=404)

    ext = target.suffix.lower()
    if ext in _CSV_EXTENSIONS:
        file_type = "csv"
    elif ext in _IMAGE_EXTENSIONS:
        file_type = "image"
    elif ext in _PDF_EXTENSIONS:
        file_type = "pdf"
    elif ext in _TEXT_EXTENSIONS:
        file_type = "text"
    else:
        file_type = "binary"

    exists = target.exists() and target.is_file()
    size = target.stat().st_size if exists else 0

    return JSONResponse(content={
        "exists": exists,
        "type": file_type,
        "name": target.name,
        "size": size,
        "extension": ext,
    })


@app.get("/api/file/csv")
async def file_csv(path: str = Query(..., description="Relative file path")):
    """Parse a CSV/TSV file and return as JSON (columns + rows)."""
    target = _safe_resolve(path)
    if target is None:
        return JSONResponse(content={"error": "File not found"}, status_code=404)

    ext = target.suffix.lower()
    delimiter = "\t" if ext == ".tsv" else ","

    try:
        content = target.read_text(encoding="utf-8")
        reader = csv.DictReader(io.StringIO(content), delimiter=delimiter)
        columns = reader.fieldnames or []
        rows = []
        for i, row in enumerate(reader):
            if i >= 10000:  # Safety cap
                break
            rows.append(row)
        return JSONResponse(content={"columns": columns, "rows": rows, "total": len(rows)})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/file/text")
async def file_text(path: str = Query(..., description="Relative file path")):
    """Return text file contents."""
    target = _safe_resolve(path)
    if target is None:
        return JSONResponse(content={"error": "File not found"}, status_code=404)

    try:
        content = target.read_text(encoding="utf-8")
        # Cap at 500KB
        if len(content) > 500_000:
            content = content[:500_000] + "\n\n... (truncated)"
        return JSONResponse(content={"content": content, "name": target.name})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/file/raw")
async def file_raw(path: str = Query(..., description="Relative file path")):
    """Serve a raw file (images, PDFs, etc.)."""
    target = _safe_resolve(path)
    if target is None:
        return JSONResponse(content={"error": "File not found"}, status_code=404)

    mime, _ = mimetypes.guess_type(str(target))
    if mime is None:
        mime = "application/octet-stream"

    return FileResponse(path=str(target), media_type=mime)


def _flatten_yaml(data: dict | list | None, prefix: str = "", source: str = "") -> list[dict]:
    """Flatten a nested YAML dict into a list of {key, value, type, source} entries."""
    result = []
    if data is None:
        return result
    if isinstance(data, dict):
        for k, v in data.items():
            full_key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            if isinstance(v, dict):
                result.extend(_flatten_yaml(v, full_key, source))
            elif isinstance(v, list):
                result.append({"key": full_key, "value": v, "type": "list", "source": source})
            elif isinstance(v, bool):
                result.append({"key": full_key, "value": v, "type": "bool", "source": source})
            elif isinstance(v, int):
                result.append({"key": full_key, "value": v, "type": "int", "source": source})
            elif isinstance(v, float):
                result.append({"key": full_key, "value": v, "type": "float", "source": source})
            elif v is None:
                result.append({"key": full_key, "value": None, "type": "null", "source": source})
            else:
                result.append({"key": full_key, "value": str(v), "type": "str", "source": source})
    return result


def _parse_default_entry(entry) -> tuple[str, str] | None:
    """Parse a single Hydra defaults entry into (group, name).

    Examples:
      '/config'              → ('', 'config')
      'override /algo: colbert' → ('algo', 'colbert')
      'dataset: musique'     → ('dataset', 'musique')
      {'dataset': 'musique'} → ('dataset', 'musique')
      '_self_'               → None (handled separately)
    """
    if isinstance(entry, dict):
        for group, name in entry.items():
            if group == "_self_":
                return None
            # Handle 'override /algo' style dict keys
            g = str(group)
            if g.startswith("override "):
                g = g[len("override "):]
            g = g.strip().lstrip("/")
            return (g, str(name))
        return None

    if not isinstance(entry, str):
        return None

    entry = entry.strip()
    if entry == "_self_":
        return None

    # Remove 'override ' prefix if present
    if entry.startswith("override "):
        entry = entry[len("override "):]

    # Format: /config or /algo: colbert or dataset: musique
    if ":" in entry:
        group_part, name = entry.split(":", 1)
        group_part = group_part.strip().lstrip("/")
        name = name.strip()
        return (group_part, name)
    else:
        # Simple reference like '/config'
        ref = entry.lstrip("/")
        return ("", ref)


def _resolve_hydra_defaults(
    config_path: Path,
    configs_dir: Path,
    visited: set[str] | None = None,
) -> list[tuple[str, dict]]:
    """Recursively resolve Hydra defaults into an ordered list of (source_rel_path, data).

    The result is ordered from base → leaf so that later entries override earlier ones
    (matching Hydra's merge semantics).
    """
    if visited is None:
        visited = set()

    abs_path = config_path.resolve()
    path_key = str(abs_path)
    if path_key in visited:
        return []
    visited.add(path_key)

    if not config_path.exists():
        return []

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return []

    if not isinstance(raw, dict):
        return []

    defaults = raw.get("defaults", [])
    if not isinstance(defaults, list):
        defaults = []

    # Determine _self_ position — controls where the file's own keys merge
    self_idx = None
    for i, entry in enumerate(defaults):
        if entry == "_self_" or (isinstance(entry, dict) and "_self_" in entry):
            self_idx = i
            break

    # Resolve defaults entries (excluding _self_)
    resolved_defaults: list[tuple[str, dict]] = []
    for entry in defaults:
        parsed = _parse_default_entry(entry)
        if parsed is None:
            continue  # _self_ or unparseable

        group, name = parsed
        if group:
            ref_path = configs_dir / group / f"{name}.yaml"
        else:
            ref_path = configs_dir / f"{name}.yaml"

        # Recursively resolve
        resolved_defaults.extend(
            _resolve_hydra_defaults(ref_path, configs_dir, visited)
        )

    # Build self data (strip 'defaults' key — it's metadata, not params)
    self_data = {k: v for k, v in raw.items() if k != "defaults"}

    # Compute relative path from project dir for display
    try:
        rel = str(config_path.relative_to(configs_dir.parent))
    except ValueError:
        rel = str(config_path)

    # Merge order: if _self_ is early, file's own keys go before defaults
    # Default Hydra behavior: _self_ last → file keys override defaults
    if self_idx is not None and self_idx == 0:
        return [(rel, self_data)] + resolved_defaults
    else:
        return resolved_defaults + [(rel, self_data)]


@app.get("/api/hydra-config")
async def get_hydra_config(path: str = Query(..., description="Relative path to Hydra config YAML")):
    """Read a Hydra config YAML with resolved defaults and return structured parameters."""
    target = _safe_resolve(path)
    if target is None:
        return JSONResponse(content={"error": "Config file not found"}, status_code=404)
    try:
        project = Path(_project_dir).resolve()
        configs_dir = project / "configs"

        # Resolve defaults chain
        sources_data = _resolve_hydra_defaults(target, configs_dir)

        if not sources_data:
            # Fallback: just read the file directly
            content = target.read_text(encoding="utf-8")
            parsed = yaml.safe_load(content) or {}
            if isinstance(parsed, dict):
                parsed.pop("defaults", None)
            params = _flatten_yaml(parsed, source=path) if isinstance(parsed, dict) else []
            return JSONResponse(content={"params": params, "path": path, "sources": [path]})

        # Flatten params from each source, keeping track of origin
        all_params: list[dict] = []
        seen_keys: dict[str, int] = {}  # key → index in all_params
        sources: list[str] = []

        for source_path, data in sources_data:
            if source_path not in sources:
                sources.append(source_path)
            flat = _flatten_yaml(data, source=source_path)
            for param in flat:
                key = param["key"]
                if key in seen_keys:
                    # Override: replace with later value
                    all_params[seen_keys[key]] = param
                else:
                    seen_keys[key] = len(all_params)
                    all_params.append(param)

        return JSONResponse(content={
            "params": all_params,
            "path": path,
            "sources": sources,
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


def _apply_params(data: dict, params: dict) -> dict:
    """Apply dot-notation param updates to a nested dict (in place)."""
    for dotted_key, new_value in params.items():
        keys = dotted_key.split(".")
        target = data
        for k in keys[:-1]:
            if isinstance(target, dict) and k in target:
                target = target[k]
            else:
                break  # key path doesn't exist — skip silently
        else:
            last = keys[-1]
            if isinstance(target, dict) and last in target:
                old = target[last]
                # Coerce new_value to match the original type
                if isinstance(old, bool):
                    if isinstance(new_value, str):
                        new_value = new_value.lower() in ("true", "1", "yes")
                    else:
                        new_value = bool(new_value)
                elif isinstance(old, int) and not isinstance(old, bool):
                    try:
                        new_value = int(new_value)
                    except (ValueError, TypeError):
                        pass
                elif isinstance(old, float):
                    try:
                        new_value = float(new_value)
                    except (ValueError, TypeError):
                        pass
                elif old is None:
                    if isinstance(new_value, str) and new_value.strip().lower() in ("null", "none", ""):
                        new_value = None
                target[last] = new_value
    return data


@app.put("/api/hydra-config")
async def put_hydra_config(request: Request):
    """Apply parameter edits to Hydra config YAML file(s).

    Expects JSON body:
      {"path": "...", "params": {"dot.key": {"value": newValue, "source": "configs/..."}}}
    OR (legacy, no source info):
      {"path": "...", "params": {"dot.key": newValue, ...}}
    """
    body = await request.json()
    path = body.get("path")
    params = body.get("params")
    if not path or not isinstance(params, dict):
        return JSONResponse(content={"error": "Missing 'path' or 'params'"}, status_code=400)

    project = Path(_project_dir).resolve()

    try:
        # Group edits by source file
        edits_by_source: dict[str, dict[str, Any]] = {}

        for key, val in params.items():
            if isinstance(val, dict) and "source" in val:
                # New format: {key: {value: ..., source: ...}}
                source = val["source"]
                edits_by_source.setdefault(source, {})[key] = val["value"]
            else:
                # Legacy format: {key: value} — edit the main file
                edits_by_source.setdefault(path, {})[key] = val

        # Apply edits to each source file
        for source_path, source_params in edits_by_source.items():
            target = (project / source_path).resolve()
            # Security: ensure within project
            if not str(target).startswith(str(project)):
                continue
            if not target.exists() or not target.is_file():
                continue

            content = target.read_text(encoding="utf-8")
            data = yaml.safe_load(content) or {}
            if not isinstance(data, dict):
                continue
            _apply_params(data, source_params)
            output = yaml.dump(data, sort_keys=False, default_flow_style=False, allow_unicode=True)
            target.write_text(output, encoding="utf-8")

        return JSONResponse(content={"success": True, "path": path})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/stage/freeze")
async def freeze_stage(request: Request):
    """Freeze or unfreeze a stage and all its descendants."""
    body = await request.json()
    stage_name = body.get("stage")
    frozen = body.get("frozen")

    if not stage_name or frozen is None:
        return JSONResponse(content={"error": "Missing 'stage' or 'frozen'"}, status_code=400)

    # 1. Build pipeline to get dependencies
    try:
        pipeline = await asyncio.to_thread(build_pipeline, _project_dir)
    except Exception as e:
        return JSONResponse(content={"error": f"Failed to parse pipeline: {e}"}, status_code=500)

    if stage_name not in pipeline.stages:
        return JSONResponse(content={"error": f"Stage '{stage_name}' not found"}, status_code=404)

    # 2. Find descendants
    # Build adjacency list: source -> [targets]
    adjacency = {}
    for edge in pipeline.edges:
        adjacency.setdefault(edge.source, []).append(edge.target)

    descendants = set()
    queue = [stage_name]
    while queue:
        current = queue.pop(0)
        # Add to descendants set (including the target stage itself)
        descendants.add(current)
        
        for child in adjacency.get(current, []):
            if child not in descendants and child not in queue:
                queue.append(child)
    
    # 3. Update dvc.yaml
    dvc_yaml_path = Path(_project_dir) / "dvc.yaml"
    if not dvc_yaml_path.exists():
        return JSONResponse(content={"error": "dvc.yaml not found"}, status_code=404)

    try:
        # Use a round-trip loader if available? No, sticking to PyYAML as per existing code.
        with open(dvc_yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
            
        stages_data = data.get("stages", {})
        count = 0
        
        for name in descendants:
            # Handle foreach stages: name like "train@small" -> base "train"
            base_name = name.split("@")[0]
            if base_name in stages_data:
                # For foreach stages, DVC expects 'frozen' inside the 'do' block
                if "foreach" in stages_data[base_name] and "do" in stages_data[base_name]:
                    if frozen:
                        stages_data[base_name]["do"]["frozen"] = True
                    else:
                        stages_data[base_name]["do"].pop("frozen", None)
                else:
                    if frozen:
                        stages_data[base_name]["frozen"] = True
                    else:
                        stages_data[base_name].pop("frozen", None)
                count += 1
        
        data["stages"] = stages_data
        
        with open(dvc_yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(data, f, sort_keys=False, default_flow_style=False, allow_unicode=True)

        return JSONResponse(content={
            "success": True, 
            "stages_affected": list(descendants),
            "frozen": frozen
        })

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/run")
async def run_pipeline(request: Request):
    """Run a single stage or the entire pipeline via `dvc repro`."""
    body = await request.json()
    stage = body.get("stage")  # None = run all
    force = body.get("force", False)
    keep_going = stage is None
    success, returncode, logs = run_dvc_repro(_project_dir, _dvc_bin, stage, force, keep_going)
    
    if returncode == 127: # FileNotFoundError
        return JSONResponse(content={
            "success": False,
            "logs": "Error: DVC is not installed or not found in PATH.",
            "failed_stage": None,
            "stage": stage,
        }, status_code=500)
    
    if returncode == -1: # Timeout
        return JSONResponse(content={
            "success": False,
            "logs": "Error: Pipeline execution timed out (10 minute limit).",
            "failed_stage": stage,
            "stage": stage,
        }, status_code=500)

    # Try to extract the failed stage from DVC error output
    failed_stage = None
    if not success:
        # DVC typically outputs "ERROR: failed to reproduce '<stage>'"
        import re
        match = re.search(r"failed to reproduce '([\w@.-]+)'", logs)
        if match:
            failed_stage = match.group(1)
        else:
            # Also look for "stage '<name>' failed"
            match = re.search(r"Stage '([\w@.-]+)' failed", logs, re.IGNORECASE)
            if match:
                failed_stage = match.group(1)

    # Trigger Auto-Commit and Auto-Push/GC if successful
    if success:
        def background_sync():
            try:
                # 1. Auto Git Commit
                if os.environ.get("DVC_VIEWER_GIT_AUTO_COMMIT") in ("1", "true", "True", "yes"):
                    has_git = Path(_project_dir, ".git").exists()
                    if has_git:
                        print("📝 Auto-committing DVC changes to Git...")

                        log_lines = logs.splitlines()[-100:]
                        last_logs = "\n".join(log_lines)

                        stage_name = stage or "All Stages"

                        msg = f"{stage_name}\n\nExecution logs (last 100 lines):\n{last_logs}"
                        subprocess.run(["git", "add", "dvc.lock", "dvc.yaml"], cwd=str(_project_dir), capture_output=True)
                        # Only commit if there are changes
                        subprocess.run(["git", "commit", "-m", msg], cwd=str(_project_dir), capture_output=True)

                        # Optional auto git push
                        rebase_res = subprocess.run(["git", "pull", "--rebase"], cwd=str(_project_dir), capture_output=True)
                        if rebase_res.returncode != 0:
                            subprocess.run(["git", "rebase", "--abort"], cwd=str(_project_dir), capture_output=True)
                        else:
                            subprocess.run(["git", "push"], cwd=str(_project_dir), capture_output=True)

                # 2. Auto Push & GC
                if os.environ.get("DVC_GDRIVE_CREDENTIALS_DATA") and os.environ.get("DVC_GDRIVE_FOLDER_ID"):
                    print("☁️ Auto-Pushing to Google Drive...")
                    subprocess.run([_dvc_bin, "push"], cwd=str(_project_dir), capture_output=True)
                    print("☁️ Auto-Cleaning (GC) on Google Drive...")
                    subprocess.run([_dvc_bin, "gc", "--cloud", "--date", "10 days ago", "-f"], cwd=str(_project_dir), capture_output=True)
            except Exception as e:
                print(f"⚠️ Background sync task failed: {e}")
        import threading
        threading.Thread(target=background_sync, daemon=True).start()

    return JSONResponse(content={
        "success": success,
        "returncode": returncode,
        "logs": logs,
        "failed_stage": failed_stage,
        "stage": stage,
    })


@app.get("/api/run-stream")
async def run_pipeline_stream(
    request: Request,
    stage: str = Query(None, description="Stage to run (null = all)"),
    force: bool = Query(False, description="Force rerun"),
):
    """Stream pipeline execution via Server-Sent Events.

    Events:
      stage_start  {"stage": "..."}                 — a stage begins executing
      stage_skip   {"stage": "..."}                 — stage skipped (unchanged)
      log          {"stage": "...", "line": "..."}  — output line
      stage_done   {"stage": "...", "success": bool} — stage finished
      done         {"success": bool, "failed_stage": ...} — pipeline complete
    """
    keep_going = stage is None
    def sse_event(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    # We will trigger push and gc ONLY ONCE at the very end of the pipeline run.
    # Triggering per stage causes DVC lock contention and saturates the network/Drive API.

    def generate():
        current_stage = None
        success = True
        failed_stages = []
        stage_logs = []

        # We process git commits strictly sequentially in the event loop instead of threads to avoid index.lock issues
        def _commit_stage_logs(stg_name, stage_lines):
            try:
                has_git = Path(_project_dir, ".git").exists()
                if has_git:
                    print(f"📝 Auto-committing DVC changes to Git for {stg_name}...")
                    last_logs = "\n".join(stage_lines[-100:])
                    msg = f"{stg_name}\n\nExecution logs (last 100 lines):\n{last_logs}"
                    subprocess.run(["git", "add", "dvc.lock", "dvc.yaml"], cwd=str(_project_dir), capture_output=True)
                    res = subprocess.run(["git", "commit", "-m", msg], cwd=str(_project_dir), capture_output=True)

                    if res.returncode == 0:
                        # Auto-push DVC first so we never push a git commit referencing missing remote data
                        if os.environ.get("DVC_GDRIVE_CREDENTIALS_DATA") and os.environ.get("DVC_GDRIVE_FOLDER_ID"):
                            print(f"☁️ Auto-Pushing {stg_name} to Google Drive...")
                            subprocess.run([_dvc_bin, "push", stg_name], cwd=str(_project_dir), capture_output=True)

                        # Now push Git
                        rebase_res = subprocess.run(["git", "pull", "--rebase"], cwd=str(_project_dir), capture_output=True)
                        if rebase_res.returncode != 0:
                            subprocess.run(["git", "rebase", "--abort"], cwd=str(_project_dir), capture_output=True)
                        else:
                            subprocess.run(["git", "push"], cwd=str(_project_dir), capture_output=True)
            except Exception as e:
                print(f"⚠️ Git commit failed: {e}")

        global _running_proc
        try:
            from .dvc_client import _auto_pull
            _auto_pull(_project_dir, _dvc_bin)
            proc = start_dvc_repro(_project_dir, _dvc_bin, stage, force, keep_going)
            _running_proc = proc
        except FileNotFoundError:
            yield sse_event("done", {
                "success": False,
                "failed_stage": None,
                "error": "DVC is not installed or not found in PATH.",
            })
            return

        re_running = re.compile(r"Running stage '([\w@.-]+)'")
        re_skipped = re.compile(r"Stage '([\w@.-]+)' didn't change")
        re_failed  = re.compile(r"failed to reproduce '([\w@.-]+)'")
        re_failed2 = re.compile(r"ERROR:.*stage '([\w@.-]+)'")

        # Open the log file for reading
        try:
            # Wait briefly to ensure file is created
            for _ in range(10):
                if log_file_path.exists():
                    break
                await asyncio.sleep(0.1)

            with open(log_file_path, "r", encoding="utf-8") as f:
                buffer = ""
                while True:
                    # Check if client disconnected
                    if await request.is_disconnected():
                        print("Client disconnected, stopping log tailing.")
                        break

                    chunk = f.read(1024) # Read in chunks
                    if not chunk:
                        if _running_proc and _running_proc.poll() is not None:
                            # Process finished, send any remaining buffer
                            if buffer:
                                # Append newline so the while loop below will process it
                                buffer += "\n"
                            else:
                                break
                        else:
                            # Wait for more output
                            await asyncio.sleep(0.1)
                            continue
                    else:
                        buffer += chunk

                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)

            if m_run:
                new_stage = m_run.group(1)
                # Previous stage finished successfully
                if current_stage and current_stage != new_stage:
                    mark_stage_complete(current_stage)
                    if os.environ.get("DVC_VIEWER_GIT_AUTO_COMMIT") in ("1", "true", "True", "yes"):
                        _commit_stage_logs(current_stage, list(stage_logs))

                    yield sse_event("stage_done", {
                        "stage": current_stage, "success": True,
                    })
                current_stage = new_stage
                stage_logs.clear()
                mark_stage_started(new_stage)
                yield sse_event("stage_start", {"stage": new_stage})

            elif m_skip:
                skipped = m_skip.group(1)
                # Close previous stage if needed
                if current_stage and current_stage != skipped:
                    mark_stage_complete(current_stage)
                    if os.environ.get("DVC_VIEWER_GIT_AUTO_COMMIT") in ("1", "true", "True", "yes"):
                        _commit_stage_logs(current_stage, list(stage_logs))

                    yield sse_event("stage_done", {
                        "stage": current_stage, "success": True,
                    })
                mark_stage_complete(skipped)
                current_stage = None
                stage_logs.clear()
                yield sse_event("stage_skip", {"stage": skipped})

                        # Detect stage transitions
                        m_run = re_running.search(line)
                        m_skip = re_skipped.search(line)

                        # Send log line
                        print(f"[{current_stage or 'dvc'}] {line}")
                        stage_logs.append(line)
                        yield sse_event("log", {
                            "stage": current_stage,
                            "line": line,
                        })

                        elif m_skip:
                            skipped = m_skip.group(1)
                            # Close previous stage if needed
                            if current_stage and current_stage != skipped:
                                mark_stage_complete(current_stage)
                                yield sse_event("stage_done", {
                                    "stage": current_stage, "success": True,
                                })
                            mark_stage_complete(skipped)
                            current_stage = None
                            yield sse_event("stage_skip", {"stage": skipped})

                        # Check for failures
                        m_fail = re_failed.search(line) or re_failed2.search(line)
                        if m_fail:
                            f_stage = m_fail.group(1)
                            mark_stage_failed(f_stage)
                            if f_stage not in failed_stages:
                                failed_stages.append(f_stage)
                            success = False

                        # Send log line
                        yield sse_event("log", {
                            "stage": current_stage,
                            "line": line,
                        })

        except Exception as e:
            print(f"Error reading log file: {e}")

        # If client disconnected, don't send completion events, process is still running
        if await request.is_disconnected():
            return

        if _running_proc:
            _running_proc.wait()
            returncode = _running_proc.returncode
            if hasattr(_running_proc, '_out_file') and _running_proc._out_file:
                try:
                    _running_proc._out_file.close()
                except Exception:
                    pass
            # Only clear if we are still the tracking process (avoid race condition on fast restarts)
            if _running_proc is not None and _running_proc.poll() is not None:
                _running_proc = None
        else:
            returncode = 0

        # Close final stage
        if current_stage:
            stage_ok = returncode == 0 and (current_stage not in failed_stages)
            if stage_ok:
                mark_stage_complete(current_stage)
                if os.environ.get("DVC_VIEWER_GIT_AUTO_COMMIT") in ("1", "true", "True", "yes"):
                    _commit_stage_logs(current_stage, list(stage_logs))

            yield sse_event("stage_done", {
                "stage": current_stage,
                "success": stage_ok,
            })

        cancelled = returncode in (-15, -9, 137)  # SIGTERM / SIGKILL
        if returncode != 0:
            success = False

        yield sse_event("done", {
            "success": success,
            "failed_stage": failed_stages[0] if failed_stages else None,
            "failed_stages": failed_stages,
            "cancelled": cancelled,
        })

        # Trigger GC AFTER the full run if it was somewhat successful
        if success:
            def background_push():
                try:
                    # Auto Push & GC
                    if os.environ.get("DVC_GDRIVE_CREDENTIALS_DATA") and os.environ.get("DVC_GDRIVE_FOLDER_ID"):
                        print("☁️ Final Auto-Pushing to Google Drive...")
                        subprocess.run([_dvc_bin, "push"], cwd=str(_project_dir), capture_output=True)
                        print("☁️ Auto-Cleaning (GC) on Google Drive...")
                        subprocess.run([_dvc_bin, "gc", "--cloud", "--workspace", "--all-commits", "--date", "10 days ago", "-f"], cwd=str(_project_dir), capture_output=True)
                except Exception as e:
                    print(f"⚠️ Background sync task failed: {e}")
            import threading
            threading.Thread(target=background_push, daemon=True).start()

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/stop")
async def stop_pipeline():
    """Stop the running dvc repro process.

    Works for both UI-launched runs (via _running_proc) and external runs
    (by reading the PID from .dvc/tmp/rwlock).
    """
    global _running_proc
    return stop_dvc_process(_project_dir, _running_proc)


@app.get("/api/file/history")
async def file_history(path: str = Query(..., description="Relative file path")):
    """Get git commit history for a file."""
    project = Path(_project_dir).resolve()
    target = (project / path).resolve()
    if not str(target).startswith(str(project)):
        return JSONResponse(content={"error": "Invalid path"}, status_code=400)

    try:
        result = subprocess.run(
            ["git", "log", "--pretty=format:%H|%s|%an|%ai", "--follow", "--", path],
            capture_output=True, text=True, cwd=str(project), timeout=15,
        )
        if result.returncode != 0:
            return JSONResponse(content={"commits": [], "error": result.stderr.strip()})

        commits = []
        for line in result.stdout.strip().split("\n"):
            if not line.strip():
                continue
            parts = line.split("|", 3)
            if len(parts) >= 4:
                commits.append({
                    "hash": parts[0],
                    "message": parts[1],
                    "author": parts[2],
                    "date": parts[3],
                })
        return JSONResponse(content={"commits": commits, "path": path})
    except Exception as e:
        return JSONResponse(content={"commits": [], "error": str(e)})


@app.get("/api/file/at-commit")
async def file_at_commit(
    path: str = Query(..., description="Relative file path"),
    commit: str = Query(..., description="Git commit hash"),
):
    """Get file content at a specific git commit."""
    project = Path(_project_dir).resolve()
    target = (project / path).resolve()
    if not str(target).startswith(str(project)):
        return JSONResponse(content={"error": "Invalid path"}, status_code=400)

    # Validate commit hash (alphanumeric only)
    if not re.match(r'^[a-f0-9]+$', commit):
        return JSONResponse(content={"error": "Invalid commit hash"}, status_code=400)

    ext = Path(path).suffix.lower()
    is_binary = ext in _IMAGE_EXTENSIONS or ext in _PDF_EXTENSIONS

    try:
        result = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            capture_output=True,
            cwd=str(project),
            timeout=15,
            text=not is_binary,
        )
        if result.returncode != 0:
            error_msg = result.stderr if isinstance(result.stderr, str) else result.stderr.decode()
            return JSONResponse(content={"error": error_msg.strip()}, status_code=404)

        if is_binary:
            mime, _ = mimetypes.guess_type(path)
            if mime is None:
                mime = "application/octet-stream"
            return Response(content=result.stdout, media_type=mime)

        content = result.stdout

        # For CSV files, parse into structured data
        if ext in _CSV_EXTENSIONS:
            delimiter = "\t" if ext == ".tsv" else ","
            reader = csv.DictReader(io.StringIO(content), delimiter=delimiter)
            columns = reader.fieldnames or []
            rows = []
            for i, row in enumerate(reader):
                if i >= 10000:
                    break
                rows.append(row)
            return JSONResponse(content={
                "type": "csv",
                "columns": columns,
                "rows": rows,
                "total": len(rows),
                "commit": commit,
            })

        # For text files
        return JSONResponse(content={
            "type": "text",
            "content": content[:500000],
            "commit": commit,
        })
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
