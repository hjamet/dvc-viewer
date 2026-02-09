"""
FastAPI web server for DVC Viewer.

Serves the static frontend and exposes the pipeline API.
"""

from __future__ import annotations

import os
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .parser import build_pipeline, pipeline_to_dict

app = FastAPI(title="DVC Viewer", version="0.1.0")

# The project dir is set at startup via environment variable
_project_dir: str = os.environ.get("DVC_VIEWER_PROJECT_DIR", os.getcwd())

# Serve static files
_static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


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
