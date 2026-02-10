"""
FastAPI web server for DVC Viewer.

Serves the static frontend and exposes the pipeline API,
plus file inspection endpoints for CSV, images, PDFs, JSON, etc.
"""

from __future__ import annotations

import csv
import io
import json
import mimetypes
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
import yaml

from .parser import build_pipeline, pipeline_to_dict, resolve_dvc_bin

app = FastAPI(title="DVC Viewer", version="0.1.0")

# The project dir is set at startup via environment variable
_project_dir: str = os.environ.get("DVC_VIEWER_PROJECT_DIR", os.getcwd())

# Resolve the DVC binary path (checks system PATH, project .venv, our venv)
_dvc_bin = resolve_dvc_bin(_project_dir)

# Serve static files
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

# Supported file categories
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg", ".bmp", ".ico"}
_CSV_EXTENSIONS = {".csv", ".tsv"}
_PDF_EXTENSIONS = {".pdf"}
_TEXT_EXTENSIONS = {".json", ".yaml", ".yml", ".txt", ".md", ".log", ".jsonl"}


def _safe_resolve(rel_path: str) -> Path | None:
    """Resolve a relative path safely within the project directory."""
    project = Path(_project_dir).resolve()
    target = (project / rel_path).resolve()
    # Prevent path traversal
    if not str(target).startswith(str(project)):
        return None
    if not target.exists() or not target.is_file():
        return None
    return target


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main HTML page."""
    index_path = _static_dir / "index.html"
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/api/pipeline", response_class=JSONResponse)
async def get_pipeline():
    """Return the parsed DVC pipeline as JSON."""
    try:
        pipeline = build_pipeline(_project_dir)
        data = pipeline_to_dict(pipeline)
        return JSONResponse(content=data)
    except FileNotFoundError as e:
        return JSONResponse(content={"error": str(e)}, status_code=404)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.get("/api/file/info")
async def file_info(path: str = Query(..., description="Relative file path")):
    """Return metadata about a file: type, size, exists."""
    target = _safe_resolve(path)
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

    stat = target.stat()
    return JSONResponse(content={
        "exists": True,
        "type": file_type,
        "name": target.name,
        "size": stat.st_size,
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


@app.get("/api/hydra-config")
async def get_hydra_config(path: str = Query(..., description="Relative path to Hydra config YAML")):
    """Read a Hydra config YAML file and return its content."""
    target = _safe_resolve(path)
    if target is None:
        return JSONResponse(content={"error": "Config file not found"}, status_code=404)
    try:
        content = target.read_text(encoding="utf-8")
        return JSONResponse(content={"content": content, "path": path})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.put("/api/hydra-config")
async def put_hydra_config(request: Request):
    """Write updated content to a Hydra config YAML file."""
    body = await request.json()
    path = body.get("path")
    content = body.get("content")
    if not path or content is None:
        return JSONResponse(content={"error": "Missing 'path' or 'content'"}, status_code=400)

    target = _safe_resolve(path)
    if target is None:
        return JSONResponse(content={"error": "Config file not found"}, status_code=404)

    try:
        # Validate YAML syntax before writing
        yaml.safe_load(content)
    except yaml.YAMLError as e:
        return JSONResponse(content={"error": f"Invalid YAML: {e}"}, status_code=422)

    try:
        target.write_text(content, encoding="utf-8")
        return JSONResponse(content={"success": True, "path": path})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)


@app.post("/api/run")
async def run_pipeline(request: Request):
    """Run a single stage or the entire pipeline via `dvc repro`."""
    import subprocess

    body = await request.json()
    stage = body.get("stage")  # None = run all
    force = body.get("force", False)

    cmd = [_dvc_bin or "dvc", "repro"]
    if stage:
        cmd.append(stage)
    if force:
        cmd.append("--force")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=str(_project_dir),
            timeout=600,
        )

        logs = (result.stdout or "") + (result.stderr or "")
        success = result.returncode == 0

        # Try to extract the failed stage from DVC error output
        failed_stage = None
        if not success:
            # DVC typically outputs "ERROR: failed to reproduce '<stage>'"
            import re
            match = re.search(r"failed to reproduce '(\w+)'", logs)
            if match:
                failed_stage = match.group(1)
            else:
                # Also look for "stage '<name>' failed"
                match = re.search(r"Stage '(\w[\w-]*)' failed", logs, re.IGNORECASE)
                if match:
                    failed_stage = match.group(1)

        return JSONResponse(content={
            "success": success,
            "returncode": result.returncode,
            "logs": logs,
            "failed_stage": failed_stage,
            "stage": stage,
        })
    except FileNotFoundError:
        return JSONResponse(content={
            "success": False,
            "logs": "Error: DVC is not installed or not found in PATH.",
            "failed_stage": None,
            "stage": stage,
        }, status_code=500)
    except subprocess.TimeoutExpired:
        return JSONResponse(content={
            "success": False,
            "logs": "Error: Pipeline execution timed out (10 minute limit).",
            "failed_stage": stage,
            "stage": stage,
        }, status_code=500)


@app.get("/api/run-stream")
async def run_pipeline_stream(
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
    import select
    import threading

    cmd = [_dvc_bin or "dvc", "repro"]
    if stage:
        cmd.append(stage)
    if force:
        cmd.append("--force")

    def sse_event(event: str, data: dict) -> str:
        return f"event: {event}\ndata: {json.dumps(data)}\n\n"

    def generate():
        current_stage = None
        success = True
        failed_stage = None

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=str(_project_dir),
                bufsize=1,  # line-buffered
            )
        except FileNotFoundError:
            yield sse_event("done", {
                "success": False,
                "failed_stage": None,
                "error": "DVC is not installed or not found in PATH.",
            })
            return

        re_running = re.compile(r"Running stage '([\w-]+)'")
        re_skipped = re.compile(r"Stage '([\w-]+)' didn't change")
        re_failed  = re.compile(r"failed to reproduce '([\w-]+)'")
        re_failed2 = re.compile(r"ERROR:.*stage '([\w-]+)'")

        for raw_line in proc.stdout:
            line = raw_line.rstrip("\n")

            # Detect stage transitions
            m_run = re_running.search(line)
            m_skip = re_skipped.search(line)

            if m_run:
                new_stage = m_run.group(1)
                # Previous stage finished successfully
                if current_stage and current_stage != new_stage:
                    yield sse_event("stage_done", {
                        "stage": current_stage, "success": True,
                    })
                current_stage = new_stage
                yield sse_event("stage_start", {"stage": new_stage})

            elif m_skip:
                skipped = m_skip.group(1)
                # Close previous stage if needed
                if current_stage and current_stage != skipped:
                    yield sse_event("stage_done", {
                        "stage": current_stage, "success": True,
                    })
                current_stage = None
                yield sse_event("stage_skip", {"stage": skipped})

            # Check for failures
            m_fail = re_failed.search(line) or re_failed2.search(line)
            if m_fail:
                failed_stage = m_fail.group(1)
                success = False

            # Send log line
            yield sse_event("log", {
                "stage": current_stage,
                "line": line,
            })

        proc.wait()

        # Close final stage
        if current_stage:
            stage_ok = proc.returncode == 0 and (failed_stage != current_stage)
            yield sse_event("stage_done", {
                "stage": current_stage,
                "success": stage_ok,
            })

        if proc.returncode != 0:
            success = False

        yield sse_event("done", {
            "success": success,
            "failed_stage": failed_stage,
        })

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


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
