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
    frozen: bool = False
    definition_order: int = 0



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
    dvc_status: dict[str, Any] | None = None  # raw dvc status --json output


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

# Regex to extract PID from DVC lock error message
_RE_LOCK_PID = re.compile(r"\(PID (\d+)\)")


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


def _load_params(project_dir: Path, dvc_data: dict) -> dict:
    """Load parameters from params.yaml and any vars section in dvc.yaml."""
    params: dict = {}
    # 1. Load params.yaml (DVC default parameter file)
    params_path = project_dir / "params.yaml"
    if params_path.exists():
        with open(params_path, "r") as f:
            loaded = yaml.safe_load(f) or {}
            params.update(loaded)
    # 2. Load any vars files or inline dicts declared in dvc.yaml
    vars_section = dvc_data.get("vars", [])
    if isinstance(vars_section, list):
        for var_entry in vars_section:
            if isinstance(var_entry, str):
                var_path = project_dir / var_entry
                if var_path.exists():
                    with open(var_path, "r") as f:
                        loaded = yaml.safe_load(f) or {}
                        params.update(loaded)
            elif isinstance(var_entry, dict):
                params.update(var_entry)
    return params


def _resolve_interpolation(value: Any, params: dict) -> Any:
    """Resolve a ${var.path} reference using the params dict.

    If *value* is a string matching ``${some.key}``, traverse *params*
    by dot-separated keys and return the actual value (preserving type).
    Non-matching strings and other types are returned as-is.
    """
    if not isinstance(value, str):
        return value
    m = re.fullmatch(r"\$\{(.+)\}", value.strip())
    if not m:
        return value
    keys = m.group(1).split(".")
    result: Any = params
    for k in keys:
        if isinstance(result, dict) and k in result:
            result = result[k]
        else:
            return value  # unresolvable → keep raw string
    return result


def parse_dvc_yaml(project_dir: str | Path) -> dict[str, Stage]:
    """Parse dvc.yaml to extract all stage definitions."""
    dvc_yaml_path = Path(project_dir) / "dvc.yaml"
    if not dvc_yaml_path.exists():
        raise FileNotFoundError(f"No dvc.yaml found in {project_dir}")

    with open(dvc_yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # Load parameters for interpolation (params.yaml + vars)
    params = _load_params(Path(project_dir), data)

    stages_data = data.get("stages", {})
    stages: dict[str, Stage] = {}

    for i, (name, definition) in enumerate(stages_data.items()):
        if not isinstance(definition, dict):
            continue

        if "foreach" in definition and "do" in definition:
            items = _resolve_interpolation(definition["foreach"], params)
            do_block = definition["do"]
            
            is_frozen = definition.get("frozen", False)
            expanded_stages = _expand_foreach(name, items, do_block, project_dir, is_frozen, definition_order_start=i * 1000)
            stages.update(expanded_stages)
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
            frozen=definition.get("frozen", False),
            definition_order=i,
        )
        stages[name] = stage

    return stages


def _expand_foreach(base_name: str, items: Any, do_block: dict, project_dir: str | Path, parent_frozen: bool = False, definition_order_start: int = 0) -> dict[str, Stage]:
    """Expand a foreach block into multiple individual stages."""
    expanded: dict[str, Stage] = {}
    
    # DVC support list or dict for foreach
    iterable: list[tuple[Any, Any]] = []
    if isinstance(items, list):
        iterable = [(None, item) for item in items]
    elif isinstance(items, dict):
        iterable = list(items.items())
    else:
        # Fallback for unexpected data types
        return {}

    for i, (key, item) in enumerate(iterable):
        # Generate stage name (e.g. stage@item or stage@key)
        suffix = str(key if key is not None else item)
        name = f"{base_name}@{suffix}"
        
        # Helper for substitution
        def sub(val: Any) -> Any:
            if isinstance(val, str):
                res = val.replace("${item}", str(item))
                if key is not None:
                    res = res.replace("${key}", str(key))
                return res
            if isinstance(val, list):
                return [sub(v) for v in val]
            if isinstance(val, dict):
                return {sub(k): sub(v) for k, v in val.items()}
            return val

        cmd = sub(do_block.get("cmd", ""))
        stage = Stage(
            name=name,
            cmd=cmd,
            deps=_resolve_dep_or_out(sub(do_block.get("deps"))),
            outs=_resolve_dep_or_out(sub(do_block.get("outs"))),
            params=_resolve_params(sub(do_block.get("params"))),
            metrics=_resolve_dep_or_out(sub(do_block.get("metrics"))),
            plots=_resolve_dep_or_out(sub(do_block.get("plots"))),
            always_changed=do_block.get("always_changed", False),
            hydra_config=_extract_hydra_config(cmd, Path(project_dir)),
            frozen=do_block.get("frozen", parent_frozen),
            definition_order=definition_order_start + i,
        )
        expanded[name] = stage
        
    return expanded


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
    1. Run `dvc status` to check if the repository is locked.
    2. If locked, extract the PID from the error message.
    3. Verify candidate PIDs (from status or pgrep) against the project's rwlock.
    """
    project_dir = Path(project_dir)
    dvc_bin = resolve_dvc_bin(project_dir)
    lock_pid: int | None = None

    # --- Step 1: Check dvc status for lock errors ---
    try:
        # Short timeout: if it doesn't fail quickly with a lock error,
        # it might be actually calculating status, which is fine.
        result = subprocess.run(
            [dvc_bin, "status", "--json"],
            capture_output=True, text=True, cwd=str(project_dir), timeout=10
        )
        if result.returncode == 0:
            # No lock held -> likely not running a repro that blocks us
            return False, None, None
        
        m = _RE_LOCK_PID.search(result.stderr)
        if m:
            lock_pid = int(m.group(1))
    except (subprocess.TimeoutExpired, OSError):
        pass

    # --- Step 2: Collect candidate PIDs ---
    candidate_pids: set[int] = set()
    if lock_pid:
        candidate_pids.add(lock_pid)
    
    try:
        pgrep_res = subprocess.run(
            ["pgrep", "-f", "dvc repro"],
            capture_output=True, text=True, timeout=5,
        )
        if pgrep_res.returncode == 0:
            for line in pgrep_res.stdout.strip().splitlines():
                if line.strip().isdigit():
                    pid = int(line.strip())
                    if pid != os.getpid():
                        candidate_pids.add(pid)
    except (subprocess.TimeoutExpired, OSError):
        pass

    if not candidate_pids:
        return False, None, None

    # --- Step 3: Verify candidate PIDs via rwlock ---
    rwlock_path = project_dir / ".dvc" / "tmp" / "rwlock"
    live_pid: int | None = None
    running_stage: str | None = None

    if rwlock_path.exists():
        try:
            raw = rwlock_path.read_text(encoding="utf-8").strip()
            if raw:
                lock_data = json.loads(raw)
                read_locks: dict = lock_data.get("read", {})
                write_locks: dict = lock_data.get("write", {})
                project_path = project_dir.resolve()

                # Find which candidate PID is actually mentioned in our rwlock
                for pid in candidate_pids:
                    if not _is_dvc_process_alive(pid):
                        continue
                    
                    # Check write locks for this PID
                    live_write_files: set[str] = set()
                    for locked_path, info_list in write_locks.items():
                        entries = info_list if isinstance(info_list, list) else [info_list]
                        for entry in entries:
                            if isinstance(entry, dict) and entry.get("pid") == pid:
                                try:
                                    rel = str(Path(locked_path).resolve().relative_to(project_path))
                                except ValueError:
                                    rel = locked_path
                                live_write_files.add(rel)

                    # Check read locks for this PID
                    live_read_files: set[str] = set()
                    for locked_path, info_list in read_locks.items():
                        entries = info_list if isinstance(info_list, list) else [info_list]
                        for entry in entries:
                            if isinstance(entry, dict) and entry.get("pid") == pid:
                                try:
                                    rel = str(Path(locked_path).resolve().relative_to(project_path))
                                except ValueError:
                                    rel = locked_path
                                live_read_files.add(rel)

                    if live_write_files or live_read_files:
                        live_pid = pid
                        
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
                        
                        if live_pid:
                            break
        except (json.JSONDecodeError, OSError):
            pass

    # If we didn't find anything in rwlock but dvc status said we're locked,
    # it's a strong signal even if we can't identify the specific stage yet.
    if live_pid is None and lock_pid and _is_dvc_process_alive(lock_pid):
        live_pid = lock_pid

    if live_pid:
        return True, running_stage, live_pid

    return False, None, None


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

    # Store the raw status in the pipeline for use in pipeline_to_dict
    pipeline.dvc_status = status

    if status is None:
        # DVC status completely unavailable (DVC not found, timeout, etc.)
        # Use dvc.lock as best-effort fallback.
        if is_running:
            for name, stage in pipeline.stages.items():
                if name == running_stage:
                    stage.state = "running"
                elif name in locked_stages:
                    # Conservative fallback: if we don't know the status,
                    # assume needs_rerun until proven otherwise.
                    stage.state = "needs_rerun"
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
            # 1. Exact match
            if dep in output_to_stage:
                source_stage = output_to_stage[dep]
                if source_stage != name:
                    pipeline.edges.append(
                        Edge(source=source_stage, target=name, label=dep)
                    )
            # 2. Match directory outputs (if dep is file inside an out dir)
            else:
                for out, source_stage in output_to_stage.items():
                    if dep.startswith(out.rstrip("/") + "/"):
                        if source_stage != name:
                            pipeline.edges.append(
                                Edge(source=source_stage, target=name, label=dep)
                            )
                        break

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
        if (stage.state in ("needs_rerun", "never_run", "failed", "running"))
        and not stage.always_changed
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

    # Extract per-stage changed file paths from raw dvc status output.
    # dvc status --json returns:
    #   {"stage_name": [{"changed deps": {"path": "modified"}},
    #                   {"changed outs": {"path": "modified"}}]}
    changed_files_per_stage: dict[str, set[str]] = {}
    if pipeline.dvc_status:
        for stage_name, entries in pipeline.dvc_status.items():
            changed: set[str] = set()
            if isinstance(entries, list):
                for entry in entries:
                    if isinstance(entry, dict):
                        for category_key, files_dict in entry.items():
                            if isinstance(files_dict, dict):
                                changed.update(files_dict.keys())
            elif isinstance(entries, dict):
                # Alternative format: {"changed deps": {...}, ...}
                for category_key, files_dict in entries.items():
                    if isinstance(files_dict, dict):
                        changed.update(files_dict.keys())
            changed_files_per_stage[stage_name] = changed

    # Build a map: output_path -> producing_stage_name
    output_to_stage: dict[str, str] = {}
    for name, stage in pipeline.stages.items():
        for out in stage.outs:
            output_to_stage[out] = name
        for metric in stage.metrics:
            output_to_stage[metric] = name

    # Set of stages that are dirty (need rerun/running/failed/never_run)
    dirty_stages: set[str] = {
        name for name, stage in pipeline.stages.items()
        if stage.state in ("needs_rerun", "running", "failed", "never_run")
    }

    def _file_status(path: str, stage_name: str, stage_state: str, role: str) -> dict:
        """Return file info with existence check and precise status color.

        Args:
            path: relative file path
            stage_name: name of the stage this file belongs to
            stage_state: state of the stage (valid, needs_rerun, ...)
            role: 'dep' for dependencies, 'out' for outputs/metrics/plots
        """
        full_path = project_dir / path
        exists = full_path.exists()
        if not exists:
            status = "missing"     # red
        elif role == "dep":
            # A dependency is outdated ONLY if:
            # 1. It is explicitly listed as changed in dvc status for this stage, OR
            # 2. It is the output of another stage that is dirty
            stage_changed = changed_files_per_stage.get(stage_name, set())
            is_directly_changed = path in stage_changed
            producer = output_to_stage.get(path)
            # Also check directory containment (dep inside an output dir)
            if producer is None:
                for out_path, out_stage in output_to_stage.items():
                    if path.startswith(out_path.rstrip("/") + "/"):
                        producer = out_stage
                        break
            is_from_dirty_upstream = producer is not None and producer in dirty_stages
            if is_directly_changed or is_from_dirty_upstream:
                status = "outdated"    # yellow/orange
            elif stage_state == "valid":
                status = "current"     # green
            else:
                status = "current"     # green — dep exists and is not the cause
        elif role == "out":
            # An output is outdated if its stage needs rerun
            if stage_state in ("needs_rerun", "running", "failed"):
                status = "outdated"    # yellow/orange
            elif stage_state == "valid":
                status = "current"     # green
            else:
                status = "unknown"     # grey (never_run but file exists)
        else:
            status = "unknown"
        return {"path": path, "exists": exists, "status": status}

    # Infrastructure stages to hide from the graph (connect to everything,
    # cluttering the visualization without adding useful information).
    _HIDDEN_STAGES = {"dvc-code-analysis"}

    nodes = []
    for name, stage in pipeline.stages.items():
        if name in _HIDDEN_STAGES:
            continue

        # Filter out .dvc-viewer/hashes/*.hash deps (infrastructure noise)
        visible_deps = [
            d for d in stage.deps
            if not d.startswith(".dvc-viewer/hashes/")
        ]

        node_dict: dict[str, Any] = {
            "id": name,
            "cmd": stage.cmd,
            "deps": [_file_status(d, name, stage.state, "dep") for d in visible_deps],
            "outs": [_file_status(o, name, stage.state, "out") for o in stage.outs],
            "params": stage.params,
            "metrics": [_file_status(m, name, stage.state, "out") for m in stage.metrics],
            "plots": [_file_status(p, name, stage.state, "out") for p in stage.plots],
            "state": stage.state,
            "frozen": stage.frozen,
        }

        if stage.hydra_config:
            node_dict["hydra_config"] = stage.hydra_config
            node_dict["hydra_config_exists"] = (
                (project_dir / stage.hydra_config).exists()
            )
        nodes.append(node_dict)

    edges = []
    for edge in pipeline.edges:
        if edge.source in _HIDDEN_STAGES or edge.target in _HIDDEN_STAGES:
            continue
        edges.append(
            {
                "source": edge.source,
                "target": edge.target,
                "label": edge.label,
            }
        )

    # Compute topological execution order (Kahn's algorithm)
    # 1. Build adjacency list and in-degrees
    in_degree = {n["id"]: 0 for n in nodes}
    adj = {n["id"]: [] for n in nodes}
    for e in edges:
        # nodes and edges are already filtered for _HIDDEN_STAGES
        if e["source"] in adj and e["target"] in adj:
            adj[e["source"]].append(e["target"])
            in_degree[e["target"]] += 1
            
    # 2. Initial queue: all nodes with in-degree 0 (sorted by definition order for stability)
    def get_order(node_id):
        return pipeline.stages[node_id].definition_order if node_id in pipeline.stages else 999999

    queue = sorted([node_id for node_id, deg in in_degree.items() if deg == 0], key=get_order)
    execution_order = []
    
    # 3. Process
    while queue:
        u = queue.pop(0)
        execution_order.append(u)
        # Sort children by definition order for stable ties
        children = sorted(adj[u], key=get_order)
        for v in children:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                # Re-sort queue to maintain definition order priority for independent branches
                queue.sort(key=get_order)

    return {
        "nodes": nodes,
        "edges": edges,
        "execution_order": execution_order,
        "is_running": pipeline.is_running,
        "running_stage": pipeline.running_stage,
        "running_pid": pipeline.running_pid,
    }
