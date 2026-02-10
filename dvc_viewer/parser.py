"""
DVC pipeline parser.

Reads dvc.yaml / dvc.lock and runs `dvc status --json` to build
a DAG model with per-stage state information.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Stage:
    """A single DVC pipeline stage."""

    name: str
    cmd: str = ""
    deps: list[str] = field(default_factory=list)
    outs: list[str] = field(default_factory=list)
    params: list[str] = field(default_factory=list)
    metrics: list[str] = field(default_factory=list)
    plots: list[str] = field(default_factory=list)
    state: str = "never_run"  # valid | needs_rerun | never_run
    hydra_config: str | None = None  # resolved relative path to config YAML


@dataclass
class Edge:
    """An edge in the pipeline DAG: source → target."""

    source: str
    target: str
    label: str = ""  # the file connecting them


@dataclass
class Pipeline:
    """Complete parsed pipeline with nodes and edges."""

    stages: dict[str, Stage] = field(default_factory=dict)
    edges: list[Edge] = field(default_factory=list)


def _resolve_dep_or_out(item: Any) -> list[str]:
    """Extract file paths from a deps/outs entry (can be str or dict)."""
    if item is None:
        return []
    result = []
    for entry in item:
        if isinstance(entry, str):
            result.append(entry)
        elif isinstance(entry, dict):
            # e.g. {path: ..., cache: false}
            for key in entry:
                result.append(str(key))
                break
        else:
            result.append(str(entry))
    return result


def _resolve_params(item: Any) -> list[str]:
    """Extract parameter references from a params entry."""
    if item is None:
        return []
    result = []
    for entry in item:
        if isinstance(entry, str):
            result.append(entry)
        elif isinstance(entry, dict):
            for file_path, params in entry.items():
                if isinstance(params, list):
                    for p in params:
                        result.append(f"{file_path}:{p}")
                else:
                    result.append(str(file_path))
    return result


# Regex to extract --config-name or -cn value from a command string
_RE_CONFIG_NAME = re.compile(r"(?:--config-name|--config_name|-cn)\s+(\S+)")


def _extract_hydra_config(cmd: str, project_dir: Path) -> str | None:
    """Extract the Hydra config path from a stage command.

    Detects ``--config-name <name>`` / ``-cn <name>`` and resolves to
    ``configs/<name>.yaml``.  Returns the relative path if the file
    exists on disk, otherwise ``None``.
    """
    m = _RE_CONFIG_NAME.search(cmd)
    if not m:
        return None
    config_name = m.group(1)
    # Hydra resolves config names relative to a config dir; the convention
    # in the consuming project is ``configs/<name>.yaml``.
    rel_path = f"configs/{config_name}.yaml"
    if (project_dir / rel_path).exists():
        return rel_path
    return None


def parse_dvc_yaml(project_dir: str | Path) -> dict[str, Stage]:
    """Parse dvc.yaml to extract all stage definitions."""
    dvc_yaml_path = Path(project_dir) / "dvc.yaml"
    if not dvc_yaml_path.exists():
        raise FileNotFoundError(f"No dvc.yaml found in {project_dir}")

    with open(dvc_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    stages_data = data.get("stages", {})
    stages: dict[str, Stage] = {}

    for name, definition in stages_data.items():
        if not isinstance(definition, dict):
            continue
        cmd = definition.get("cmd", "")
        stage = Stage(
            name=name,
            cmd=cmd,
            deps=_resolve_dep_or_out(definition.get("deps")),
            outs=_resolve_dep_or_out(definition.get("outs")),
            params=_resolve_params(definition.get("params")),
            metrics=_resolve_dep_or_out(definition.get("metrics")),
            plots=_resolve_dep_or_out(definition.get("plots")),
            hydra_config=_extract_hydra_config(cmd, Path(project_dir)),
        )
        stages[name] = stage

    return stages


def parse_dvc_lock(project_dir: str | Path) -> set[str]:
    """Parse dvc.lock to find which stages have been run at least once."""
    lock_path = Path(project_dir) / "dvc.lock"
    if not lock_path.exists():
        return set()

    with open(lock_path, "r") as f:
        data = yaml.safe_load(f)

    if not data:
        return set()

    stages = data.get("stages", {})
    return set(stages.keys())


def resolve_dvc_bin(project_dir: str | Path) -> str:
    """Find the DVC binary, checking system PATH, project .venv, and our own venv."""
    # 1. System PATH
    found = shutil.which("dvc")
    if found:
        return found
    # 2. Project's own .venv
    project_venv = Path(project_dir) / ".venv" / "bin" / "dvc"
    if project_venv.exists():
        return str(project_venv)
    # 3. Same venv as dvc-viewer itself
    own_venv = Path(sys.executable).parent / "dvc"
    if own_venv.exists():
        return str(own_venv)
    return "dvc"  # fallback


def get_dvc_status(project_dir: str | Path) -> dict[str, Any] | None:
    """Run `dvc status --json` and return parsed output.

    Returns None if the command could not be executed (DVC not found, timeout,
    or DVC lock held by another process such as a running ``dvc repro``).
    Returns {} if DVC reports no changes.
    """
    dvc_bin = resolve_dvc_bin(project_dir)
    try:
        result = subprocess.run(
            [dvc_bin, "status", "--json"],
            capture_output=True,
            text=True,
            cwd=str(project_dir),
            timeout=60,
        )
        if result.returncode != 0:
            # dvc status returns non-zero when there are changes, that's fine
            # Only truly fail if there's no output at all
            if result.stdout.strip():
                return json.loads(result.stdout)
            # No stdout → DVC error (e.g. lock held by running dvc repro)
            return None
        output = result.stdout.strip()
        if not output or output == "{}":
            return {}
        return json.loads(output)
    except FileNotFoundError:
        return None
    except (subprocess.TimeoutExpired, json.JSONDecodeError):
        return None


def build_pipeline(project_dir: str | Path) -> Pipeline:
    """Build the full pipeline DAG from a DVC project directory."""
    project_dir = Path(project_dir)
    pipeline = Pipeline()

    # 1. Parse stages from dvc.yaml
    stages = parse_dvc_yaml(project_dir)
    pipeline.stages = stages

    # 2. Find which stages have been executed (have lock entries)
    locked_stages = parse_dvc_lock(project_dir)

    # 3. Get current status (which stages need re-running)
    status = get_dvc_status(project_dir)

    if status is None:
        # DVC status failed (e.g. lock held by running dvc repro).
        # Mark locked stages as needs_rerun — better amber than incorrect green.
        for name, stage in pipeline.stages.items():
            if name in locked_stages:
                stage.state = "needs_rerun"
            else:
                stage.state = "never_run"
    else:
        stages_needing_rerun = set(status.keys()) if status else set()

        # 4. Assign states
        for name, stage in pipeline.stages.items():
            if name not in locked_stages:
                stage.state = "never_run"
            elif name in stages_needing_rerun:
                stage.state = "needs_rerun"
            else:
                stage.state = "valid"

    # 5. Build edges: if stage B depends on a file that is stage A's output
    output_to_stage: dict[str, str] = {}
    for name, stage in stages.items():
        for out in stage.outs:
            output_to_stage[out] = name
        for metric in stage.metrics:
            output_to_stage[metric] = name

    for name, stage in stages.items():
        for dep in stage.deps:
            if dep in output_to_stage:
                source_stage = output_to_stage[dep]
                if source_stage != name:
                    pipeline.edges.append(
                        Edge(source=source_stage, target=name, label=dep)
                    )

    return pipeline


def pipeline_to_dict(pipeline: Pipeline) -> dict[str, Any]:
    """Convert a Pipeline to a JSON-serializable dict for the API."""
    # Determine the project dir from the first stage or fallback
    project_dir = Path(os.environ.get("DVC_VIEWER_PROJECT_DIR", os.getcwd()))

    def _file_status(path: str, stage_state: str) -> dict:
        """Return file info with existence check and status color."""
        full_path = project_dir / path
        exists = full_path.exists()
        if not exists:
            status = "missing"     # red
        elif stage_state == "needs_rerun":
            status = "outdated"    # orange
        elif stage_state == "valid":
            status = "current"     # green
        else:
            status = "unknown"     # grey (never_run but file exists)
        return {"path": path, "exists": exists, "status": status}

    nodes = []
    for name, stage in pipeline.stages.items():
        node_dict: dict[str, Any] = {
            "id": name,
            "cmd": stage.cmd,
            "deps": [_file_status(d, stage.state) for d in stage.deps],
            "outs": [_file_status(o, stage.state) for o in stage.outs],
            "params": stage.params,
            "metrics": [_file_status(m, stage.state) for m in stage.metrics],
            "plots": [_file_status(p, stage.state) for p in stage.plots],
            "state": stage.state,
        }
        if stage.hydra_config:
            node_dict["hydra_config"] = stage.hydra_config
            node_dict["hydra_config_exists"] = (
                (project_dir / stage.hydra_config).exists()
            )
        nodes.append(node_dict)

    edges = []
    for edge in pipeline.edges:
        edges.append(
            {
                "source": edge.source,
                "target": edge.target,
                "label": edge.label,
            }
        )

    return {"nodes": nodes, "edges": edges}
