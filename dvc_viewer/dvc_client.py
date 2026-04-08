"""
DVC client module.

Encapsulates all raw interactions with the DVC executable and its internal files
(rwlock, subprocess execution, process tracking).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple


class StageFiles(NamedTuple):
    """Simplified view of stage outputs and dependencies for process detection."""
    deps: list[str]
    outs: list[str]
    metrics: list[str]
    plots: list[str]


def resolve_dvc_bin(project_dir: str | Path) -> str:
    """Find the DVC binary, checking project .venv first, then system PATH."""
    project_venv = Path(project_dir) / ".venv" / "bin" / "dvc"
    if project_venv.exists():
        return str(project_venv)
    found = shutil.which("dvc")
    if found:
        return found
    own_venv = Path(sys.executable).parent / "dvc"
    if own_venv.exists():
        return str(own_venv)
    return "dvc"


def safe_read_rwlock(rwlock_path: Path) -> dict | None:
    """Read and parse the rwlock JSON file directly, with retries and cleanup."""
    if not rwlock_path.exists():
        return None

    for attempt in range(2):
        try:
            raw = rwlock_path.read_text(encoding="utf-8").strip()
            if not raw:
                if attempt == 0:
                    time.sleep(0.2)
                    continue
                break
            return json.loads(raw)
        except (json.JSONDecodeError, OSError):
            if attempt == 0:
                time.sleep(0.2)
                continue
            break

    # If corrupted, check if a repro is running before deleting
    pgrep_res = subprocess.run(["pgrep", "-f", "dvc repro"], capture_output=True)
    if pgrep_res.returncode != 0:
        try:
            rwlock_path.unlink()
        except OSError:
            pass
    return None


def extract_pids_from_locks(lock_section: dict) -> set[int]:
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


def is_dvc_process_alive(pid: int) -> bool:
    """Check if a PID is alive AND is a 'dvc repro' process (not PID reuse)."""
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False

    if sys.platform.startswith("linux"):
        try:
            cmdline_path = f"/proc/{pid}/cmdline"
            if os.path.exists(cmdline_path):
                with open(cmdline_path, "rb") as f:
                    cmdline_raw = f.read()
                cmdline_parts = cmdline_raw.decode("utf-8", errors="replace").split("\x00")
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


def get_dvc_status(project_dir: str | Path) -> dict[str, Any] | None:
    """Run `dvc status --json` and return parsed output."""
    project_dir = Path(project_dir)
    safe_read_rwlock(project_dir / ".dvc" / "tmp" / "rwlock")

    dvc_bin = resolve_dvc_bin(project_dir)
    try:
        result = subprocess.run(
            [dvc_bin, "status", "--json"],
            capture_output=True,
            text=True,
            cwd=str(project_dir),
            timeout=15,
        )
        if result.returncode != 0:
            if result.stdout.strip():
                try:
                    return json.loads(result.stdout)
                except json.JSONDecodeError:
                    return None
            return None
        output = result.stdout.strip()
        if not output or output == "{}":
            return {}
        try:
            return json.loads(output)
        except json.JSONDecodeError:
            return None
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def detect_running_stage(
    project_dir: str | Path,
    stages_files: dict[str, StageFiles],
) -> tuple[bool, str | None, int | None]:
    """Detect if a DVC run is in progress using rwlock and pgrep."""
    project_dir = Path(project_dir)
    rwlock_path = project_dir / ".dvc" / "tmp" / "rwlock"
    live_pid: int | None = None
    running_stage: str | None = None

    lock_data = safe_read_rwlock(rwlock_path)
    if lock_data:
        read_locks = lock_data.get("read", {})
        write_locks = lock_data.get("write", {})
        project_path = project_dir.resolve()

        all_pids = extract_pids_from_locks(read_locks) | extract_pids_from_locks(write_locks)
        for pid in all_pids:
            if is_dvc_process_alive(pid):
                live_pid = pid
                # Identify stage via write locks (outputs)
                live_write_files: set[str] = set()
                for locked_path, info_list in write_locks.items():
                    entries = info_list if isinstance(info_list, list) else [info_list]
                    for entry in entries:
                        if isinstance(entry, dict) and entry.get("pid") == pid:
                            try:
                                rel = str(Path(locked_path).resolve().relative_to(project_path))
                                live_write_files.add(rel)
                            except ValueError:
                                live_write_files.add(locked_path)

                if live_write_files:
                    for name, files in stages_files.items():
                        stage_outputs = set(files.outs) | set(files.metrics) | set(files.plots)
                        if stage_outputs & live_write_files:
                            running_stage = name
                            break

                if running_stage is None: # Fallback: read locks (deps)
                    live_read_files: set[str] = set()
                    for locked_path, info_list in read_locks.items():
                        entries = info_list if isinstance(info_list, list) else [info_list]
                        for entry in entries:
                            if isinstance(entry, dict) and entry.get("pid") == pid:
                                try:
                                    rel = str(Path(locked_path).resolve().relative_to(project_path))
                                    live_read_files.add(rel)
                                except ValueError:
                                    live_read_files.add(locked_path)

                    if live_read_files:
                        best_match, best_count = None, 0
                        for name, files in stages_files.items():
                            overlap = set(files.deps) & live_read_files
                            if len(overlap) > best_count:
                                best_count = len(overlap)
                                best_match = name
                        running_stage = best_match

                return True, running_stage, live_pid

    # Fallback to pgrep
    try:
        pgrep_res = subprocess.run(["pgrep", "-f", "dvc repro"], capture_output=True, text=True, timeout=2)
        if pgrep_res.returncode == 0:
            for line in pgrep_res.stdout.strip().splitlines():
                if line.strip().isdigit():
                    pid = int(line.strip())
                    if pid != os.getpid() and is_dvc_process_alive(pid):
                        return True, None, pid
    except (subprocess.TimeoutExpired, OSError):
        pass

    return False, None, None


def run_dvc_repro(project_dir: str | Path, dvc_bin: str, stage: str | None = None, force: bool = False, keep_going: bool = False) -> tuple[bool, int, str]:
    """Run `dvc repro` synchronously and return (success, returncode, logs)."""
    cmd = [dvc_bin, "repro"]
    if stage:
        cmd.append(stage)
    if force:
        cmd.append("--force")
    if keep_going:
        cmd.append("--keep-going")

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, cwd=str(project_dir), timeout=600
        )
        logs = (result.stdout or "") + (result.stderr or "")
        return result.returncode == 0, result.returncode, logs
    except FileNotFoundError:
        return False, 127, "Error: DVC not found."
    except subprocess.TimeoutExpired:
        return False, -1, "Error: Execution timed out."


def start_dvc_repro(project_dir: str | Path, dvc_bin: str, stage: str | None = None, force: bool = False, keep_going: bool = False, log_file: str | Path | None = None) -> subprocess.Popen:
    """Start `dvc repro` as a background process."""
    cmd = [dvc_bin, "repro"]
    if stage:
        cmd.append(stage)
    if force:
        cmd.append("--force")
    if keep_going:
        cmd.append("--keep-going")

    if log_file:
        out_file = open(log_file, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd,
            stdout=out_file,
            stderr=subprocess.STDOUT,
            cwd=str(project_dir),
            bufsize=1, # Line buffering
            text=True, # Ensure text mode for line buffering
            start_new_session=True,
            env={**os.environ, "PYTHONUNBUFFERED": "1"} # Force unbuffered output for Python subprocesses
        )
        proc._out_file = out_file
        return proc
    else:
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(project_dir),
            bufsize=1,
            start_new_session=True,
        )


def stop_dvc_process(project_dir: str | Path, running_proc: subprocess.Popen | None = None) -> dict:
    """Stop the running dvc repro process (UI-launched or external)."""
    import signal
    if running_proc and running_proc.poll() is None:
        try:
            os.killpg(os.getpgid(running_proc.pid), signal.SIGTERM)
        except (ProcessLookupError, OSError):
            try:
                running_proc.terminate()
            except (ProcessLookupError, OSError):
                pass
        return {"stopped": True, "source": "ui"}

    rwlock_path = Path(project_dir) / ".dvc" / "tmp" / "rwlock"
    lock_data = safe_read_rwlock(rwlock_path)
    if lock_data:
        for _path, info in lock_data.get("write", {}).items():
            if isinstance(info, dict) and "pid" in info:
                pid = info["pid"]
                try:
                    os.kill(pid, 0)
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                    return {"stopped": True, "source": "external", "pid": pid}
                except (ProcessLookupError, PermissionError, OSError):
                    pass
                break
    return {"stopped": False, "reason": "No pipeline running"}
