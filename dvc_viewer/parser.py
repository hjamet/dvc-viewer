"""
DVC pipeline parser.

Reads dvc.yaml / dvc.lock and runs `dvc status --json` to build
a DAG model with per-stage state information.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .dvc_client import (
    get_dvc_status,
    detect_running_stage,
    StageFiles,
)
from .git_client import git_show_file

# Cache the last successful dvc status result for use during pipeline runs
# (when dvc status can't be called because the lock is held).
_last_dvc_status: dict[str, Any] | None = None

# Track stages that failed during a run (since DVC status might not reflect this immediately)
_failed_stages: set[str] = set()

# Snapshot of dvc.lock at the beginning of a run to detect completed stages by diffing hashes
_dvc_lock_snapshot: dict[str, dict[str, Any]] | None = None

# Track previous running state to detect transitions
_was_running: bool = False


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


def parse_dvc_lock(project_dir: str | Path) -> dict[str, dict[str, Any]]:
    """Parse dvc.lock and return detailed per-stage hash information.

    Returns a dict mapping stage_name -> {
        "deps": {path: md5, ...},
        "outs": {path: md5, ...},
        "params": {file: {key: value, ...}, ...},
    }
    """
    lock_path = Path(project_dir) / "dvc.lock"
    if not lock_path.exists():
        return {}

    with open(lock_path, "r") as f:
        try:
            data = yaml.safe_load(f)
        except yaml.YAMLError:
            return {}

    if not data:
        return {}

    stages_data = data.get("stages", {})
    parsed_stages: dict[str, dict[str, Any]] = {}

    for name, stage_data in stages_data.items():
        if not isinstance(stage_data, dict):
            continue
            
        stage_info = {
            "deps": {},
            "outs": {},
            "params": {},
        }

        # Extract deps hashes
        for dep in stage_data.get("deps", []):
            path = dep.get("path")
            md5 = dep.get("md5") or dep.get("etag") or dep.get("checksum")
            if path and md5:
                stage_info["deps"][path] = md5

        # Extract outs hashes
        for out in stage_data.get("outs", []):
            path = out.get("path")
            md5 = out.get("md5") or out.get("etag") or out.get("checksum")
            if path and md5:
                stage_info["outs"][path] = md5

        # Extract params
        params = stage_data.get("params", {})
        if params:
            stage_info["params"] = params

        parsed_stages[name] = stage_info

    return parsed_stages


    return True


def _check_stage_hashes_on_disk(
    project_dir: Path,
    stage_lock_info: dict[str, Any],
) -> bool:
    """Check if the MD5 hashes in dvc.lock match the actual files on disk.

    Returns True if all deps/outs match (stage is valid).
    """
    import hashlib

    def compute_md5(path: Path) -> str | None:
        if not path.exists() or path.is_dir():
            return None
        # Optimization: for very large files, this could be slow
        # but for correctness we follow the "no cheating" rule.
        hash_md5 = hashlib.md5()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except OSError:
            return None

    # Check dependencies
    for path_str, expected_md5 in stage_lock_info.get("deps", {}).items():
        if ":" in path_str:
            continue # Skip params in deps if any
        full_path = project_dir / path_str
        if not full_path.exists():
            return False
        if full_path.is_dir():
            # For directories, DVC uses a .dir hash. 
            # Recalculating it exactly like DVC is complex.
            # Minimal check: if it exists, assume OK for now or skip.
            continue
        
        actual_md5 = compute_md5(full_path)
        if actual_md5 != expected_md5:
            return False

    # Check outputs
    for path_str, expected_md5 in stage_lock_info.get("outs", {}).items():
        full_path = project_dir / path_str
        if not full_path.exists():
            return False
        if full_path.is_dir():
            continue
            
        actual_md5 = compute_md5(full_path)
        if actual_md5 != expected_md5:
            return False

    return True


    return True


def build_pipeline(project_dir: str | Path) -> Pipeline:
    """Build the full pipeline DAG from a DVC project directory."""
    global _last_dvc_status, _dvc_lock_snapshot, _was_running
    project_dir = Path(project_dir)
    pipeline = Pipeline()

    # 1. Parse stages from dvc.yaml
    stages = parse_dvc_yaml(project_dir)
    pipeline.stages = stages

    # 2. Get detailed lock info (hashes)
    lock_data = parse_dvc_lock(project_dir)

    # 3. Detect if a DVC run is in progress
    stages_files = {
        name: StageFiles(s.deps, s.outs, s.metrics, s.plots)
        for name, s in stages.items()
    }
    is_running, running_stage, running_pid = detect_running_stage(
        project_dir, stages_files
    )
    pipeline.is_running = is_running
    pipeline.running_stage = running_stage
    pipeline.running_pid = running_pid

    # 4. Handle snapshot transitions
    if is_running and not _was_running:
        # Start of a run: take snapshot
        _dvc_lock_snapshot = lock_data
    elif not is_running and _was_running:
        # End of a run: clear snapshot
        _dvc_lock_snapshot = None
    
    _was_running = is_running

    # 5. Get current status (from DVC CLI if not running)
    if is_running:
        status = _last_dvc_status  # use cached status from before the run (for dirty info)
    else:
        status = get_dvc_status(project_dir)
        if status is not None:
            _last_dvc_status = status  # cache for use during runs

    pipeline.dvc_status = status
    stages_needing_rerun_cli = set(status.keys()) if status else set()

    # 6. Assign states
    for name, stage in pipeline.stages.items():
        if name == running_stage:
            stage.state = "running"
        elif name in _failed_stages:
            stage.state = "failed"
        elif name not in lock_data:
            stage.state = "never_run"
        elif is_running:
            # During a run, use Snapshot/Diff logic
            if _dvc_lock_snapshot and name in _dvc_lock_snapshot:
                # Compare current lock hash with snapshot
                curr_hashes = lock_data.get(name, {})
                snap_hashes = _dvc_lock_snapshot.get(name, {})
                
                # If any output hash changed, it means the stage finished successfully in this run
                if curr_hashes.get("outs") != snap_hashes.get("outs") and curr_hashes.get("outs"):
                    stage.state = "valid"
                elif name in stages_needing_rerun_cli:
                    stage.state = "needs_rerun"
                else:
                    stage.state = "valid"
            else:
                # No snapshot available (late join): Check MD5 on disk
                if _check_stage_hashes_on_disk(project_dir, lock_data[name]):
                    stage.state = "valid"
                else:
                    stage.state = "needs_rerun"
        else:
            # Normal mode: use DVC status CLI if available
            if status is not None:
                if name in stages_needing_rerun_cli:
                    stage.state = "needs_rerun"
                else:
                    stage.state = "valid"
            else:
                # Conservative fallback if status call failed but not running
                stage.state = "needs_rerun"

    # Post-validation: Override "valid" to "needs_rerun" if any required file
    # (dep or out) is missing from disk.
    for name, stage in pipeline.stages.items():
        if stage.state != "valid":
            continue
        all_files = stage.deps + stage.outs + stage.metrics + stage.plots
        if stage.hydra_config:
            all_files.append(stage.hydra_config)
            
        for f in all_files:
            if ":" in f:
                continue
            full_path = project_dir / f
            if not full_path.exists():
                stage.state = "needs_rerun"
                break

    # 7. Build edges
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
            else:
                for out, source_stage in output_to_stage.items():
                    if dep.startswith(out.rstrip("/") + "/"):
                        if source_stage != name:
                            pipeline.edges.append(
                                Edge(source=source_stage, target=name, label=dep)
                            )
                        break

    # 8. Propagate needs_rerun transitively through the DAG.
    downstream: dict[str, list[str]] = {}
    for edge in pipeline.edges:
        downstream.setdefault(edge.source, []).append(edge.target)

    if is_running and running_stage:
        queue = [running_stage]
        visited_descendants = set()
        while queue:
            current = queue.pop(0)
            for child in downstream.get(current, []):
                if child not in visited_descendants:
                    visited_descendants.add(child)
                    queue.append(child)
                    child_stage = pipeline.stages.get(child)
                    if child_stage:
                        child_stage.state = "needs_rerun"

    dirty = {
        name
        for name, stage in pipeline.stages.items()
        if (stage.state in ("needs_rerun", "never_run", "failed", "running"))
        and not stage.always_changed
    }
    
    visited: set[str] = set()
    queue = list(dirty)
    while queue:
        current = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)
        for child in downstream.get(current, []):
            child_stage = pipeline.stages.get(child)
            if child_stage and child_stage.state == "valid":
                child_stage.state = "needs_rerun"
            if child not in visited:
                queue.append(child)

    return pipeline


def _parse_dvc_yaml_from_data(
    data: dict,
    params: dict,
    project_dir: Path,
) -> dict[str, Stage]:
    """Parse stages from already-loaded dvc.yaml data (no filesystem access)."""
    stages_data = data.get("stages", {})
    stages: dict[str, Stage] = {}

    for i, (name, definition) in enumerate(stages_data.items()):
        if not isinstance(definition, dict):
            continue

        if "foreach" in definition and "do" in definition:
            items = _resolve_interpolation(definition["foreach"], params)
            do_block = definition["do"]
            is_frozen = definition.get("frozen", False)
            # _expand_foreach uses _extract_hydra_config which checks filesystem,
            # but we pass project_dir anyway — hydra_config will be None for
            # missing files, which is acceptable in history mode.
            expanded_stages = _expand_foreach(
                name, items, do_block, project_dir,
                is_frozen, definition_order_start=i * 1000,
            )
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
            # hydra_config requires filesystem check — skip in history mode
            hydra_config=_extract_hydra_config(cmd, project_dir),
            frozen=definition.get("frozen", False),
            definition_order=i,
        )
        stages[name] = stage

    return stages


def build_pipeline_at_commit(project_dir: str | Path, commit: str) -> Pipeline:
    """Build the pipeline DAG as it was at a specific git commit.

    Reads dvc.yaml, dvc.lock, and params.yaml via `git show` (no checkout).
    Stages with lock entries are marked 'valid', others 'never_run'.
    No dvc status or running detection is performed.
    """
    project_dir = Path(project_dir)
    pipeline = Pipeline()

    # 1. Read dvc.yaml at commit
    dvc_yaml_content = git_show_file(project_dir, commit, "dvc.yaml")
    if dvc_yaml_content is None:
        raise FileNotFoundError(f"No dvc.yaml found at commit {commit}")

    dvc_data = yaml.safe_load(dvc_yaml_content) or {}

    # 2. Load params for interpolation (params.yaml + inline vars at commit)
    params: dict = {}
    params_content = git_show_file(project_dir, commit, "params.yaml")
    if params_content:
        loaded = yaml.safe_load(params_content) or {}
        params.update(loaded)

    # Handle vars section (only inline dicts — file vars require filesystem)
    vars_section = dvc_data.get("vars", [])
    if isinstance(vars_section, list):
        for var_entry in vars_section:
            if isinstance(var_entry, dict):
                params.update(var_entry)
            elif isinstance(var_entry, str):
                var_content = git_show_file(project_dir, commit, var_entry)
                if var_content:
                    loaded = yaml.safe_load(var_content) or {}
                    params.update(loaded)

    # 3. Parse stages
    stages = _parse_dvc_yaml_from_data(dvc_data, params, project_dir)
    pipeline.stages = stages

    # 4. Read dvc.lock at commit to determine which stages were executed
    lock_content = git_show_file(project_dir, commit, "dvc.lock")
    locked_stages: set[str] = set()
    if lock_content:
        lock_data = yaml.safe_load(lock_content) or {}
        locked_stages = set(lock_data.get("stages", {}).keys())

    # 5. Assign states: locked → valid, unlocked → never_run
    for name, stage in pipeline.stages.items():
        if name in locked_stages:
            stage.state = "valid"
        else:
            stage.state = "never_run"

    # 6. Build edges (same logic as build_pipeline)
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
            else:
                for out, source_stage in output_to_stage.items():
                    if dep.startswith(out.rstrip("/") + "/"):
                        if source_stage != name:
                            pipeline.edges.append(
                                Edge(source=source_stage, target=name, label=dep)
                            )
                        break

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
            
    # 2. Initial queue: all nodes with in-degree 0 (sorted by execution priority and definition order)
    def get_order(node_id):
        stage = pipeline.stages.get(node_id)
        if not stage:
            return (999999, 5)
        
        # Primary sort: definition_order preserves dvc.yaml structure
        # Secondary sort: state_priority ensures running appears before needs_rerun
        # when two stages have the same definition_order (tiebreaker only)
        state_priority = {
            "running": 0,
            "failed": 1,
            "needs_rerun": 2,
            "never_run": 3,
            "valid": 4,
        }
        return (stage.definition_order, state_priority.get(stage.state, 5))

    queue = sorted([node_id for node_id, deg in in_degree.items() if deg == 0], key=get_order)
    execution_order = []
    
    # 3. Process
    while queue:
        u = queue.pop(0)
        execution_order.append(u)
        # Sort children by execution priority and definition order for stable ties
        children = sorted(adj[u], key=get_order)
        for v in children:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)
                # Re-sort queue to maintain priority for independent branches
                queue.sort(key=get_order)

    # Compute progress statistics
    total_stages = len(nodes)
    completed_stages = sum(1 for n in nodes if n["state"] == "valid")
    running_index = None
    if pipeline.running_stage:
        for i, name in enumerate(execution_order):
            if name == pipeline.running_stage:
                running_index = i
                break

    return {
        "nodes": nodes,
        "edges": edges,
        "execution_order": execution_order,
        "is_running": pipeline.is_running,
        "running_stage": pipeline.running_stage,
        "running_pid": pipeline.running_pid,
        "progress": {
            "total": total_stages,
            "completed": completed_stages,
            "running_index": running_index,
        },
    }
