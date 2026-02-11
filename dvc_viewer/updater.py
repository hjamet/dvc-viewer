"""
dvc.yaml updater logic.

Scans the DVC pipeline, computes code hashes, and updates dvc.yaml
to include the hasher stage and hash dependencies.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import yaml
from pathlib import Path
from typing import Any

from .hasher import find_transitive_dependencies, compute_aggregate_hash


def _find_project_python(project_dir: Path) -> str:
    """Resolve the best Python interpreter for the project.

    Priority:
    1. Active virtualenv ($VIRTUAL_ENV/bin/python)
    2. Conventional venv directories (.venv, venv, .env, env)
    3. System python3 fallback
    """
    # 1. Active virtualenv
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        candidate = Path(venv) / "bin" / "python"
        if candidate.exists():
            return str(candidate)
    # 2. Conventional venv directories in project
    for dirname in (".venv", "venv", ".env", "env"):
        candidate = project_dir / dirname / "bin" / "python"
        if candidate.exists():
            return str(candidate)
    # 3. System fallback
    return shutil.which("python3") or shutil.which("python") or "python3"


def _find_script_in_cmd(cmd: str, project_dir: Path) -> Path | None:
    """Extract script path from cmd (simplistic approach: first .py file)."""
    if not cmd:
        return None
    for part in cmd.split():
        if part.endswith(".py"):
            part = part.strip("'\"")
            p = project_dir / part
            if p.exists():
                return p
    return None


def _resolve_foreach_items(items: Any, project_dir: Path) -> Any:
    """Resolve ${var} from params.yaml if items is a string variable reference."""
    if isinstance(items, str) and items.strip().startswith("${") and items.strip().endswith("}"):
        var_name = items.strip()[2:-1]  # remove ${ and }
        params_path = project_dir / "params.yaml"
        if params_path.exists():
            try:
                with open(params_path, "r") as f:
                    params = yaml.safe_load(f) or {}
                # Handle nested keys e.g. ${foo.bar}
                val = params
                for part in var_name.split("."):
                    if isinstance(val, dict) and part in val:
                        val = val[part]
                    else:
                        return items # Failed to resolve
                return val
            except Exception:
                return items
    return items


def update_dvc_yaml(project_dir: Path) -> None:
    """
    Main logic:
    1. Read dvc.yaml
    2. Identify python stages
    3. Compute hashes -> write to .dvc-viewer/hashes/
    4. Inject 'dvc-code-analysis' stage
    5. Inject hash file dependencies into python stages
    6. Write back dvc.yaml if changed
    """
    dvc_yaml_path = project_dir / "dvc.yaml"
    if not dvc_yaml_path.exists():
        return

    print("🔍 Scanning dependencies and computing hashes...")
    
    with open(dvc_yaml_path, "r") as f:
        config = yaml.safe_load(f) or {}

    stages = config.get("stages", {})
    if not stages:
        return

    # Prepare hash directory
    hash_dir = project_dir / ".dvc-viewer" / "hashes"
    hash_dir.mkdir(parents=True, exist_ok=True)
    
    modified = False
    
    # 1. Ensure the hasher stage exists
    hasher_stage_name = "dvc-code-analysis"
    expected_hasher = {
        "cmd": "dvc-viewer hash",
        "always_changed": True,
        "outs": [{".dvc-viewer/hashes": {"cache": False}}],
    }
    
    # Check if we need to add/update the hasher stage
    # If it's missing or different, update it
    # We place it first logic-wise, but dict order depends on insertion
    if hasher_stage_name not in stages:
        print(f"   ➕ Adding '{hasher_stage_name}' stage")
        # Put it at the start if possible (create new dict)
        new_stages = {hasher_stage_name: expected_hasher}
        new_stages.update(stages)
        stages = new_stages
        config["stages"] = stages
        modified = True
    else:
        # Update definition if needed
        if stages[hasher_stage_name] != expected_hasher:
            stages[hasher_stage_name] = expected_hasher
            modified = True

    # 2. Process other stages
    for name, stage in list(stages.items()):
        if name == hasher_stage_name:
            continue
            
        # Handle foreach
        if "foreach" in stage and "do" in stage:
            items = stage["foreach"]
            items = _resolve_foreach_items(items, project_dir)
            do_block = stage["do"]
            
            # We want to inject the dependency into the do_block
            # so that ALL expanded stages depend on their respective hash
            # DVC documentation says we can use ${item} in do_block.deps
            
            # But first, we need to know what to hash.
            # Usually it's the script in the 'cmd'.
            cmd = do_block.get("cmd", "")
            script_path = _find_script_in_cmd(cmd, project_dir)
            
            if script_path:
                # Compute hash (transitive)
                deps = find_transitive_dependencies(script_path, project_dir)
                code_hash = compute_aggregate_hash(deps, project_dir)
                
                # Now we need to decide how to name the hash files.
                # If we use stage@${item}.hash, DVC will expand it.
                # But we need to write these files during the 'dvc-viewer hash' run.
                # 'dvc-viewer hash' (which calls this function) runs BEFORE the stages.
                
                # If we are here, we are scanning the dvc.yaml.
                # We should expand the items to write the hash files.
                
                iterable = []
                if isinstance(items, list):
                    iterable = [(None, item) for item in items]
                elif isinstance(items, dict):
                    iterable = list(items.items())
                
                for key, item in iterable:
                    suffix = str(key if key is not None else item)
                    expanded_name = f"{name}@{suffix}"
                    
                    hash_file_name = f"{expanded_name}.hash"
                    hash_file_path = hash_dir / hash_file_name
                    
                    # Update hash file
                    hash_file_path.write_text(code_hash)
                
                # Use ${key} for dicts and ${item} for lists to match DVC's stage naming suffix
                var_name = "key" if isinstance(items, dict) else "item"
                hash_dep = f".dvc-viewer/hashes/{name}@${{{var_name}}}.hash"
                
                current_deps = do_block.get("deps", [])
                if current_deps is None: current_deps = []
                
                if not any((isinstance(d, str) and ".dvc-viewer/hashes/" in d) or (isinstance(d, dict) and any(".dvc-viewer/hashes/" in k for k in d)) for d in current_deps):
                    print(f"   🔗 Adding dependency to '{name}' (foreach): {hash_dep}")
                    current_deps.append(hash_dep)
                    do_block["deps"] = current_deps
                    modified = True
            continue

        cmd = stage.get("cmd", "")
        script_path = _find_script_in_cmd(cmd, project_dir)
        
        if not script_path:
            continue
            
        # Compute hash
        deps = find_transitive_dependencies(script_path, project_dir)
        code_hash = compute_aggregate_hash(deps, project_dir)
        
        # Write hash file
        hash_file_name = f"{name}.hash"
        hash_file_path = hash_dir / hash_file_name
        
        # Check if hash changed
        old_hash = hash_file_path.read_text().strip() if hash_file_path.exists() else None
        if old_hash != code_hash:
            hash_file_path.write_text(code_hash)
        
        # Ensure dependency exists in dvc.yaml
        hash_dep = f".dvc-viewer/hashes/{hash_file_name}"
        current_deps = stage.get("deps", [])
        
        # Handle if deps is None
        if current_deps is None:
            current_deps = []
            
        # Check if hash_dep is already present
        has_dep = False
        for dep in current_deps:
            if isinstance(dep, str) and dep == hash_dep:
                has_dep = True
                break
            elif isinstance(dep, dict) and hash_dep in dep:
                has_dep = True
                break
        
        if not has_dep:
            print(f"   🔗 Adding dependency to '{name}': {hash_dep}")
            current_deps.append(hash_dep)
            stage["deps"] = current_deps
            modified = True

    # --- Hook execution ---
    hooks_dir = project_dir / ".dvc-viewer" / "hooks"
    post_hash_hook = hooks_dir / "post_hash.py"
    if post_hash_hook.exists():
        python = _find_project_python(project_dir)
        print(f"   🪝 Running post_hash hook: {post_hash_hook}")
        try:
            result = subprocess.run(
                [python, str(post_hash_hook)],
                cwd=str(project_dir),
                check=True,
                capture_output=True,
                text=True,
            )
            if result.stdout.strip():
                for line in result.stdout.strip().splitlines():
                    print(f"      {line}")
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️ post_hash hook failed (non-blocking): {e}")

    if modified:
        print("   💾 Updating dvc.yaml...")
        with open(dvc_yaml_path, "w") as f:
            yaml.dump(config, f, sort_keys=False, default_flow_style=None)
    else:
        print("   ✅ dvc.yaml is up to date")



