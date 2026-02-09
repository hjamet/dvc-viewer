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
from pathlib import Path

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .parser import build_pipeline, pipeline_to_dict

app = FastAPI(title="DVC Viewer", version="0.1.0")

# The project dir is set at startup via environment variable
_project_dir: str = os.environ.get("DVC_VIEWER_PROJECT_DIR", os.getcwd())

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
