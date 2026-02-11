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

# Cache the last successful dvc status result for use during pipeline runs
# (when dvc status can't be called because the lock is held).
_last_dvc_status: dict[str, Any] | None = None

# Track stages that failed during a run (since DVC status might not reflect this immediately)
_failed_stages: set[str] = set()


def mark_stage_started(name: str):
    """Mark a stage as started, clearing any previous failure state."""
    global _failed_stages
    if name in _failed_stages:
        _failed_stages.remove(name)


def mark_stage_complete(name: str):
    """Mark a stage as complete in the cached status."""
    global _last_dvc_status, _failed_stages
    if _last_dvc_status and name in _last_dvc_status:
        del _last_dvc_status[name]
    if name in _failed_stages:
        _failed_stages.remove(name)


def mark_stage_failed(name: str):
    """Mark a stage as failed."""
    global _failed_stages
    if name:
        _failed_stages.add(name)


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
    state: str = "never_run"  # valid | needs_rerun | never_run | running
    always_changed: bool = False
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
    is_running: bool = False
    running_stage: str | None = None
    running_pid: int | None = None


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
            always_changed=definition.get("always_changed", False),
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
    """Find the DVC binary, checking project .venv first, then system PATH."""
    # 1. Project's own .venv (most reliable — avoids broken pyenv shims)
    project_venv = Path(project_dir) / ".venv" / "bin" / "dvc"
    if project_venv.exists():
        return str(project_venv)
    # 2. System PATH
    found = shutil.which("dvc")
    if found:
        return found
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


def _extract_pids_from_locks(lock_section: dict) -> set[int]:
    """Extract all unique PIDs from a rwlock section (read or write)."""
    pids: set[int] = set()
    for _path, info in lock_section.items():
        if isinstance(info, list):
            for entry in info:
                if isinstance(entry, dict) and "pid" in entry:
                    pids.add(entry["pid"])
        elif isinstance(info, dict) and "pid" in info:
            pids.add(info["pid"])
    return pids


def _is_dvc_process_alive(pid: int) -> bool:
    """Check if a PID is alive AND is a 'dvc repro' process (not PID reuse)."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False

    if sys.platform.startswith("linux"):
        # Check it's actually a 'dvc repro' process (not just any process
        # with 'dvc' in its path, e.g. 'dvc-viewer hash')
        try:
            cmdline_path = f"/proc/{pid}/cmdline"
            if os.path.exists(cmdline_path):
                with open(cmdline_path, "rb") as f:
                    cmdline_raw = f.read()
                # cmdline is null-byte separated
                cmdline_parts = cmdline_raw.decode("utf-8", errors="replace").split("\x00")
                # Look for 'dvc' and 'repro' as separate arguments
                # e.g. ['/path/to/dvc', 'repro'] or ['python', '-m', 'dvc', 'repro']
                has_dvc = any("dvc" in part and "dvc-viewer" not in part for part in cmdline_parts)
                has_repro = "repro" in cmdline_parts
                if not (has_dvc and has_repro):
                    return False
        except OSError:
            pass
        # Check for zombie
        try:
            stat_path = f"/proc/{pid}/stat"
            if os.path.exists(stat_path):
                with open(stat_path, "r") as f:
                    stat_parts = f.read().split()
                if len(stat_parts) > 2 and stat_parts[2] == "Z":
                    return False
        except (OSError, IndexError):
            pass

    return True


def detect_running_stage(
    project_dir: str | Path,
    stages: dict[str, Stage],
    locked_stages: set[str] | None = None,
) -> tuple[bool, str | None, int | None]:
    """Detect if a DVC run is in progress.

    Returns (is_running, running_stage_name, pid).

    Strategy:
    1. Use `pgrep` to find a live `dvc repro` process (most reliable).
    2. If found, read rwlock to identify which specific stage is running.
    """
    # --- Step 1: Check if any `dvc repro` process is actually running ---
    live_pid: int | None = None
    try:
        result = subprocess.run(
            ["pgrep", "-f", "dvc repro"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().splitlines():
                line = line.strip()
                if line.isdigit():
                    candidate_pid = int(line)
                    # Exclude our own process (in case dvc-viewer itself
                    # matches the grep pattern)
                    if candidate_pid != os.getpid():
                        live_pid = candidate_pid
                        break
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass

    if live_pid is None:
        return False, None, None

    # --- Step 2: Identify which stage is running via rwlock ---
    rwlock_path = Path(project_dir) / ".dvc" / "tmp" / "rwlock"
    running_stage: str | None = None

    if rwlock_path.exists():
        try:
            raw = rwlock_path.read_text(encoding="utf-8").strip()
            if raw:
                lock_data = json.loads(raw)
                read_locks: dict = lock_data.get("read", {})
                write_locks: dict = lock_data.get("write", {})
                project_path = Path(project_dir).resolve()

                # Collect write-locked files held by the live PID
                live_write_files: set[str] = set()
                for locked_path, info_list in write_locks.items():
                    entries = info_list if isinstance(info_list, list) else [info_list]
                    for entry in entries:
                        if isinstance(entry, dict) and entry.get("pid") == live_pid:
                            try:
                                rel = str(Path(locked_path).resolve().relative_to(project_path))
                            except ValueError:
                                rel = locked_path
                            live_write_files.add(rel)

                # Collect read-locked files held by the live PID
                live_read_files: set[str] = set()
                for locked_path, info_list in read_locks.items():
                    entries = info_list if isinstance(info_list, list) else [info_list]
                    for entry in entries:
                        if isinstance(entry, dict) and entry.get("pid") == live_pid:
                            try:
                                rel = str(Path(locked_path).resolve().relative_to(project_path))
                            except ValueError:
                                rel = locked_path
                            live_read_files.add(rel)

                # Match write-locked files to stage outputs (most precise)
                if live_write_files:
                    for name, stage in stages.items():
                        stage_outputs = set(stage.outs) | set(stage.metrics) | set(stage.plots)
                        if stage_outputs & live_write_files:
                            running_stage = name
                            break

                # Fallback: match read-locked files to stage deps
                if running_stage is None and live_read_files:
                    best_match: str | None = None
                    best_count = 0
                    for name, stage in stages.items():
                        stage_deps = set(stage.deps)
                        overlap = stage_deps & live_read_files
                        if len(overlap) > best_count:
                            best_count = len(overlap)
                            best_match = name
                    running_stage = best_match
        except (json.JSONDecodeError, OSError):
            pass

    return True, running_stage, live_pid


def build_pipeline(project_dir: str | Path) -> Pipeline:
    """Build the full pipeline DAG from a DVC project directory."""
    global _last_dvc_status
    project_dir = Path(project_dir)
    pipeline = Pipeline()

    # 1. Parse stages from dvc.yaml
    stages = parse_dvc_yaml(project_dir)
    pipeline.stages = stages

    # 2. Find which stages have been executed (have lock entries)
    locked_stages = parse_dvc_lock(project_dir)

    # 3. Detect if a DVC run is in progress
    is_running, running_stage, running_pid = detect_running_stage(
        project_dir, stages, locked_stages
    )
    pipeline.is_running = is_running
    pipeline.running_stage = running_stage
    pipeline.running_pid = running_pid

    # 3 (already done above). locked_stages used for both detection and state.

    # 4. Get current status (which stages need re-running)
    #    Skip dvc status if a run is in progress (the lock would make it fail)
    #    but use the last known status to preserve needs_rerun information.
    if is_running:
        status = _last_dvc_status  # use cached status from before the run
    else:
        status = get_dvc_status(project_dir)
        if status is not None:
            _last_dvc_status = status  # cache for use during runs

    if status is None:
        # DVC status completely unavailable (DVC not found, timeout, etc.)
        # Use dvc.lock as best-effort fallback.
        if is_running:
            for name, stage in pipeline.stages.items():
                if name == running_stage:
                    stage.state = "running"
                elif name in locked_stages:
                    stage.state = "valid"
                else:
                    stage.state = "never_run"
        else:
            # Not running, but DVC couldn't provide status.
            # Default to needs_rerun so the user knows something is off.
            for name, stage in pipeline.stages.items():
                if name in _failed_stages:
                    stage.state = "failed"
                elif name in locked_stages:
                    stage.state = "needs_rerun"
                else:
                    stage.state = "never_run"
    else:
        stages_needing_rerun = set(status.keys()) if status else set()

        for name, stage in pipeline.stages.items():
            if name == running_stage:
                stage.state = "running"
            elif name in _failed_stages:
                stage.state = "failed"
            elif name not in locked_stages:
                stage.state = "never_run"
            elif name in stages_needing_rerun:
                stage.state = "needs_rerun"
            else:
                stage.state = "valid"

    # Post-validation: Override "valid" to "needs_rerun" if any required file
    # (dep or out) is missing from disk. DVC status only checks changes,
    # not existence.
    for name, stage in pipeline.stages.items():
        if stage.state != "valid":
            continue
        # Check deps, outs, metrics, plots, and hydra_config
        all_files = stage.deps + stage.outs + stage.metrics + stage.plots
        if stage.hydra_config:
            all_files.append(stage.hydra_config)
            
        for f in all_files:
            if ":" in f: continue  # Skip params like 'params.yaml:lr'
            full_path = project_dir / f
            exists = full_path.exists()
            if not exists:
                stage.state = "needs_rerun"
                break

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

    # 6. Propagate needs_rerun transitively through the DAG.
    #    If stage A needs rerun (or is running), all downstream stages
    #    should be marked needs_rerun (Yellow).
    downstream: dict[str, list[str]] = {}
    for edge in pipeline.edges:
        downstream.setdefault(edge.source, []).append(edge.target)

    # If a stage is running, its descendants are pending execution -> needs_rerun
    if is_running and running_stage:
        # Mark all descendants of the running stage as 'needs_rerun' (Yellow)
        # regardless of what dvc.lock says (which reflects the previous successful run)
        queue = [running_stage]
        visited_descendants = set()
        while queue:
            current = queue.pop(0)
            children = downstream.get(current, [])
            for child in children:
                if child not in visited_descendants:
                    visited_descendants.add(child)
                    queue.append(child)
                    # Force state to needs_rerun (Yellow)
                    child_stage = pipeline.stages.get(child)
                    if child_stage:
                        child_stage.state = "needs_rerun"

    # Also perform standard propagation for static needs_rerun states
    dirty = {
        name
        for name, stage in pipeline.stages.items()
        if stage.state in ("needs_rerun", "never_run") and not stage.always_changed
    }
    # Note: "running" state propagation is handled above more aggressively
    
    visited: set[str] = set()
    queue = list(dirty)
    
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        
        children = downstream.get(current, [])
        for child in children:
            child_stage = pipeline.stages.get(child)
            # If child was 'valid' (Green), it is invalidated
            if child_stage and child_stage.state == "valid":
                child_stage.state = "needs_rerun"
            if child not in visited:
                queue.append(child)

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

    return {
        "nodes": nodes,
        "edges": edges,
        "is_running": pipeline.is_running,
        "running_stage": pipeline.running_stage,
        "running_pid": pipeline.running_pid,
    }
