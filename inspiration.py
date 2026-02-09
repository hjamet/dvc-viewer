#!/usr/bin/env python3
"""
🚀 CLIMB Transitive Code Hasher (Automated DVC Edition)
Calculates aggregated SHA256 hashes for entry-point scripts by recursively 
resolving and hashing all dependencies.

This version automatically parses dvc.yaml to find entry points.
"""

import ast
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Set, List, Dict, Optional, Tuple, Any
import yaml
try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("code_hasher")

# Base Configuration
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
HASH_DIR = PROJECT_ROOT / ".code_hash"

# Known sensitive extensions to track if found in strings
SENSITIVE_EXTENSIONS = {".py", ".yaml", ".yml", ".jsonl", ".json", ".sh"}

# --- GLOBAL CACHES (Persistent across stages in a single run) ---
_DEPENDENCY_CACHE: Dict[Path, Set[Path]] = {}
_AST_CACHE: Dict[Path, ast.AST] = {}

def get_ast(path: Path) -> ast.AST:
    """Read and parse AST with caching."""
    if path not in _AST_CACHE:
        content = path.read_text(encoding="utf-8")
        _AST_CACHE[path] = ast.parse(content)
    return _AST_CACHE[path]

def get_module_info(path: Path) -> Tuple[Set[str], List[Tuple[str, int]], Set[str]]:
    """Extract imports, relative imports, and candidate file paths from AST."""
    abs_imports = set()
    rel_imports = []
    strings = set()

    try:
        tree = get_ast(path)
    except Exception:
        return abs_imports, rel_imports, strings

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                abs_imports.add(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                rel_imports.append((node.module or "", node.level))
            elif node.module:
                abs_imports.add(node.module)
        elif isinstance(node, (ast.Str, ast.Constant)):
            val = getattr(node, "s", None) if isinstance(node, ast.Str) else getattr(node, "value", None)
            if isinstance(val, str) and 2 < len(val) < 255 and "\n" not in val:
                # Track if it looks like a file path or config
                if any(val.endswith(ext) for ext in SENSITIVE_EXTENSIONS) or "/" in val:
                    strings.add(val)
    return abs_imports, rel_imports, strings

def resolve_absolute_import(module_name: str, roots: List[Path]) -> Optional[Path]:
    """Resolve an absolute module name to a file path using provided roots."""
    parts = module_name.split(".")
    for root in roots:
        p = root.joinpath(*parts).with_suffix(".py")
        if p.exists(): return p
        p = root.joinpath(*parts).joinpath("__init__.py")
        if p.exists(): return p
    return None

def resolve_relative_import(current_file: Path, module_name: str, level: int) -> Optional[Path]:
    """Resolve a relative import (e.g. .utils or ..models)."""
    parent = current_file.parent
    for _ in range(level - 1):
        if parent.parent == parent: break
        parent = parent.parent
    parts = module_name.split(".") if module_name else []
    p = parent.joinpath(*parts).with_suffix(".py")
    if p.exists(): return p
    p = parent.joinpath(*parts).joinpath("__init__.py")
    if p.exists(): return p
    return None

def extract_sys_path_additions(path: Path) -> List[Path]:
    """Detect sys.path adjustments."""
    new_roots = []
    try:
        tree = get_ast(path)
    except Exception:
        return new_roots

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Attribute) and 
                node.func.attr in ("insert", "append") and 
                isinstance(node.func.value, ast.Attribute) and 
                node.func.value.attr == "path" and 
                isinstance(node.func.value.value, ast.Name) and 
                node.func.value.value.id == "sys"):
                new_roots.append(PROJECT_ROOT)
                new_roots.append(PROJECT_ROOT / "src")
    return list(set(new_roots))

def flatten_config(cfg: Any, prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested config into a dot-notation dict of leaf values."""
    items = {}
    if isinstance(cfg, (dict, DictConfig) if 'DictConfig' in globals() else dict):
        for k, v in cfg.items():
            new_key = f"{prefix}.{k}" if prefix else k
            items.update(flatten_config(v, new_key))
            if not isinstance(v, (dict, list)):
                items[k] = v
    elif isinstance(cfg, list):
        for i, v in enumerate(cfg):
            items.update(flatten_config(v, f"{prefix}[{i}]"))
    else:
        items[prefix] = cfg
    return items

def is_branch_active(node: ast.AST, flat_cfg: Dict[str, Any]) -> bool:
    """Semi-statically evaluate if a branch is active based on config values."""
    try:
        if isinstance(node, ast.Compare):
            left = node.left
            if len(node.ops) == 1 and isinstance(node.ops[0], (ast.Eq, ast.Is)):
                right = node.comparators[0]
                def get_val(n):
                    if isinstance(n, (ast.Str, ast.Constant)):
                        return getattr(n, "s", None) if isinstance(n, ast.Str) else getattr(n, "value", None)
                    return None
                const_val = get_val(left) or get_val(right)
                if const_val is not None:
                    return any(v == const_val for v in flat_cfg.values())
        if isinstance(node, ast.Name):
            if node.id in flat_cfg and isinstance(flat_cfg[node.id], bool):
                return flat_cfg[node.id]
    except Exception: pass
    return True

def find_transitive_dependencies(entry_path: Path, flat_cfg: Dict[str, Any]) -> Set[Path]:
    """Recursive dependency discovery with agnostic branch pruning."""
    all_deps = {entry_path.resolve()}
    to_visit = [entry_path.resolve()]
    visited_in_this_run = {entry_path.resolve()}
    
    base_roots = [PROJECT_ROOT, PROJECT_ROOT / "src"]
    
    # Path to dispatcher to apply special pruning logic
    DISPATCHER_PATH = (PROJECT_ROOT / "src/pathfinder_rag/training/trainers/dispatcher.py").resolve()
    
    # Cache for dispatcher mapping to avoid re-parsing
    dispatcher_mapping = {}
    if DISPATCHER_PATH.exists():
        try:
            tree = get_ast(DISPATCHER_PATH)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id == "trainers":
                            if isinstance(node.value, ast.Dict):
                                for k, v in zip(node.value.keys, node.value.values):
                                    key = getattr(k, "s", None) or getattr(k, "value", None)
                                    val = None
                                    if isinstance(v, ast.Name): val = v.id
                                    elif isinstance(v, ast.Attribute): val = v.attr
                                    if key and val: dispatcher_mapping[key] = val
        except Exception as e:
            logger.debug(f"Could not parse dispatcher mapping: {e}")

    while to_visit:
        current_file = to_visit.pop()
        
        if current_file in _DEPENDENCY_CACHE:
            immediate_deps = _DEPENDENCY_CACHE[current_file]
        else:
            immediate_deps = set()
            local_roots = list(set(base_roots + extract_sys_path_additions(current_file)))
            
            abs_imports = set()
            rel_imports = []
            string_paths = set()
            
            try:
                tree = get_ast(current_file)
                def walk_with_pruning(node_list):
                    for node in node_list:
                        # 1. Imports
                        if isinstance(node, ast.Import):
                            for n in node.names: abs_imports.add(n.name)
                        elif isinstance(node, ast.ImportFrom):
                            if node.level > 0: rel_imports.append((node.module or "", node.level))
                            elif node.module: abs_imports.add(node.module)
                        # 2. Strings
                        elif isinstance(node, (ast.Str, ast.Constant)):
                            val = getattr(node, "s", None) if isinstance(node, ast.Str) else getattr(node, "value", None)
                            if isinstance(val, str) and 2 < len(val) < 255 and "\n" not in val:
                                if any(val.endswith(ext) for ext in SENSITIVE_EXTENSIONS) or "/" in val:
                                    string_paths.add(val)
                        # 3. Branch Pruning
                        elif isinstance(node, ast.If):
                            if is_branch_active(node.test, flat_cfg): walk_with_pruning(node.body)
                            else: walk_with_pruning(node.orelse)
                        elif hasattr(node, "body") and isinstance(node.body, list):
                            walk_with_pruning(node.body)
                walk_with_pruning(tree.body)
            except Exception as e: logger.debug(f"Error walking {current_file}: {e}")

            for mod in abs_imports:
                if mod.startswith("pathfinder_rag") or mod.startswith("scripts"):
                    path = resolve_absolute_import(mod, local_roots)
                    if path: immediate_deps.add(path.resolve())
            for mod, level in rel_imports:
                path = resolve_relative_import(current_file, mod, level)
                if path: immediate_deps.add(path.resolve())
            for s in string_paths:
                for base in [current_file.parent, PROJECT_ROOT]:
                    p = (base / s).resolve()
                    if p.exists() and p.is_file(): immediate_deps.add(p)
            
            immediate_deps = {
                p for p in immediate_deps 
                if str(p).startswith(str(PROJECT_ROOT)) and 
                not any(x in str(p) for x in [".code_hash", "__pycache__"])
            }
            _DEPENDENCY_CACHE[current_file] = immediate_deps

        for p in immediate_deps:
            if p not in visited_in_this_run:
                all_deps.add(p)
                visited_in_this_run.add(p)
                if p.suffix == ".py": to_visit.append(p)
                    
    return all_deps

def compute_aggregate_hash(file_paths: Set[Path]) -> str:
    """Compute an aggregate SHA256 hash."""
    hasher = hashlib.sha256()
    for path in sorted(list(file_paths)):
        try:
            rel_path = path.relative_to(PROJECT_ROOT)
        except ValueError:
            rel_path = path
        hasher.update(str(rel_path).encode("utf-8"))
        hasher.update(path.read_bytes())
    return hasher.hexdigest()

def extract_python_script_and_config(cmd: str) -> Tuple[Optional[Path], Optional[str]]:
    """Extract .py script and Hydra config-name from cmd string."""
    parts = cmd.split()
    script = None
    config = None
    for i, part in enumerate(parts):
        if part in ("python", "python3") and i + 1 < len(parts):
            script_path = parts[i + 1]
            if script_path.endswith(".py"):
                p = PROJECT_ROOT / script_path
                script = p if p.exists() else None
        if part == "--config-name" and i + 1 < len(parts):
            config = parts[i + 1]
    return script, config

def detect_algo_from_config(config_name: str) -> Optional[str]:
    """Detect algorithm name from Hydra config."""
    if not OmegaConf: return None
    config_path = PROJECT_ROOT / "configs" / f"{config_name}.yaml"
    if not config_path.exists(): return None
    try:
        conf = OmegaConf.load(config_path)
        # Try different common patterns
        if "algo" in conf and "name" in conf.algo: return conf.algo.name
        if "algorithms" in conf: return list(conf.algorithms.keys())[0]
    except Exception:
        pass
    return None

def main():
    import argparse
    import yaml
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Print tracked files")
    args = parser.parse_args()
    HASH_DIR.mkdir(parents=True, exist_ok=True)
    DVC_PATH = PROJECT_ROOT / "dvc.yaml"
    if not DVC_PATH.exists():
        logger.error(f"❌ dvc.yaml not found at {DVC_PATH}"); sys.exit(1)
    try:
        with open(DVC_PATH, "r") as f: dvc_cfg = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"❌ Failed to parse dvc.yaml: {e}"); sys.exit(1)
    stages = dvc_cfg.get("stages", {})
    
    processed_count = 0
    for stage_name, stage_info in stages.items():
        cmd = stage_info.get("cmd", "")
        entry_path, config_name = extract_python_script_and_config(cmd)
        if not entry_path: continue
        
        # Load config to guide hashing (Branch-Aware Pruning)
        flat_cfg = {}
        if config_name and OmegaConf:
            config_path = PROJECT_ROOT / "configs" / f"{config_name}.yaml"
            if config_path.exists():
                try:
                    raw_cfg = OmegaConf.load(config_path)
                    flat_cfg = flatten_config(raw_cfg)
                except Exception: pass

        deps = find_transitive_dependencies(entry_path, flat_cfg)
        new_hash = compute_aggregate_hash(deps)
        
        hash_file = HASH_DIR / f"{stage_name}.hash"
        old_hash = hash_file.read_text().strip() if hash_file.exists() else None
        
        emoji = "✅" if old_hash == new_hash else ("🆕" if old_hash is None else "🔄")
        with open(hash_file, "w") as f: f.write(new_hash)
            
        logger.info(f"{emoji} {stage_name}: {len(deps)} files -> {stage_name}.hash")
        processed_count += 1
        if args.debug:
            print(f"\n--- {stage_name} ---")
            for d in sorted(list(deps)): print(f"  - {d.relative_to(PROJECT_ROOT)}")
            
    logger.info(f"🚀 Done. Processed {processed_count} stages.")

if __name__ == "__main__":
    main()
