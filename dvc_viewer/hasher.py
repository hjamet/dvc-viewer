"""
Code hashing module for change detection.

Analyzes Python scripts to find transitive dependencies (imports, relative imports)
and computes an aggregate hash. Adapted from the CLIMB Transitive Code Hasher.
"""

from __future__ import annotations

import ast
import hashlib
import logging
import os
import sys
from pathlib import Path
from typing import Set, Tuple

# Setup logging
logger = logging.getLogger("dvc_viewer.hasher")

# Extensions to track if found in strings (potential dynamic dependencies)
SENSITIVE_EXTENSIONS = {
    ".py", ".yaml", ".yml", ".json", ".sh", ".bash",
    ".r", ".R", ".pl", ".rb", ".js", ".ts", ".sql", ".lua",
    ".cfg", ".ini", ".toml", ".conf", ".csv", ".tsv",
}

# Functions that execute external commands
_SUBPROCESS_FUNCS = {
    # subprocess module
    "run", "call", "check_call", "check_output", "Popen",
    # os module
    "system", "popen",
}

# Attribute chains that indicate subprocess/os calls
_SUBPROCESS_MODULES = {"subprocess", "os"}


def _extract_subprocess_file_refs(call_node: ast.Call) -> list[str]:
    """
    Extract file-like references from subprocess/os.system calls.

    Detects patterns like:
        subprocess.run(["bash", "scripts/process.sh", ...])
        subprocess.run("./run.sh arg1 arg2")
        os.system("python train.py --epochs 10")
        subprocess.call(["sh", "-c", "echo hello"])
    """
    func = call_node.func
    is_subprocess = False

    # Check for module.function pattern (subprocess.run, os.system, etc.)
    if isinstance(func, ast.Attribute) and func.attr in _SUBPROCESS_FUNCS:
        if isinstance(func.value, ast.Name) and func.value.id in _SUBPROCESS_MODULES:
            is_subprocess = True

    if not is_subprocess:
        return []

    refs = []
    args = call_node.args

    if not args:
        return refs

    first_arg = args[0]

    # Case 1: String command — e.g. subprocess.run("python train.py --lr 0.01")
    if isinstance(first_arg, (ast.Str, ast.Constant)):
        val = getattr(first_arg, "s", None) if isinstance(first_arg, ast.Str) else getattr(first_arg, "value", None)
        if isinstance(val, str):
            # Split the command string and check each token for file-like patterns
            for token in val.split():
                token = token.strip("'\"")
                if _looks_like_file(token):
                    refs.append(token)

    # Case 2: List command — e.g. subprocess.run(["python", "scripts/train.py", ...])
    elif isinstance(first_arg, ast.List):
        for elt in first_arg.elts:
            if isinstance(elt, (ast.Str, ast.Constant)):
                val = getattr(elt, "s", None) if isinstance(elt, ast.Str) else getattr(elt, "value", None)
                if isinstance(val, str) and _looks_like_file(val):
                    refs.append(val)

    return refs


def _looks_like_file(token: str) -> bool:
    """Check if a token looks like a file path (has extension or path separator)."""
    if len(token) < 2 or len(token) > 255:
        return False
    # Skip common non-file tokens
    if token in ("python", "python3", "bash", "sh", "zsh", "perl", "ruby", "node",
                 "dvc", "git", "echo", "cat", "ls", "cd", "cp", "mv", "rm", "mkdir",
                 "-c", "-e", "-m", "--", "-"):
        return False
    # Check for file-like patterns
    if any(token.endswith(ext) for ext in SENSITIVE_EXTENSIONS):
        return True
    if "/" in token and not token.startswith("-"):
        return True
    if "." in token and not token.startswith("-") and " " not in token:
        # Has a dot, could be a file — but skip things like --flag=value
        suffix = "." + token.rsplit(".", 1)[-1] if "." in token else ""
        if suffix in SENSITIVE_EXTENSIONS:
            return True
    return False

# Caches to avoid re-parsing
_DEPENDENCY_CACHE: dict[Path, set[Path]] = {}
_AST_CACHE: dict[Path, ast.AST] = {}


def get_ast(path: Path) -> ast.AST:
    """Read and parse AST with caching."""
    if path not in _AST_CACHE:
        # Fail gracefully if file encoding is weird or file is missing
        try:
            content = path.read_text(encoding="utf-8")
            _AST_CACHE[path] = ast.parse(content)
        except Exception as e:
            logger.warning(f"Failed to parse {path}: {e}")
            _AST_CACHE[path] = ast.Module(body=[], type_ignores=[])
    return _AST_CACHE[path]


def resolve_absolute_import(module_name: str, roots: list[Path]) -> Path | None:
    """Resolve an absolute module name to a file path using provided roots."""
    parts = module_name.split(".")
    for root in roots:
        # Check for package/__init__.py
        p_init = root.joinpath(*parts).joinpath("__init__.py")
        if p_init.exists():
            return p_init
        # Check for module.py
        p_mod = root.joinpath(*parts).with_suffix(".py")
        if p_mod.exists():
            return p_mod
    return None


def resolve_relative_import(current_file: Path, module_name: str, level: int) -> Path | None:
    """Resolve a relative import (e.g. .utils or ..models)."""
    parent = current_file.parent
    for _ in range(level - 1):
        if parent.parent == parent:  # Hit root
            break
        parent = parent.parent
    
    parts = module_name.split(".") if module_name else []
    
    # Check for module.py
    p_mod = parent.joinpath(*parts).with_suffix(".py")
    if p_mod.exists():
        return p_mod
    
    # Check for package/__init__.py
    p_init = parent.joinpath(*parts).joinpath("__init__.py")
    if p_init.exists():
        return p_init
        
    return None


def extract_sys_path_additions(path: Path, project_root: Path) -> list[Path]:
    """Detect sys.path adjustments to find extra source roots."""
    new_roots = []
    try:
        tree = get_ast(path)
    except Exception:
        return new_roots

    for node in ast.walk(tree):
        # Look for sys.path.append(...) or sys.path.insert(...)
        if isinstance(node, ast.Call):
            if (isinstance(node.func, ast.Attribute) and 
                node.func.attr in ("insert", "append") and 
                isinstance(node.func.value, ast.Attribute) and 
                node.func.value.attr == "path" and 
                isinstance(node.func.value.value, ast.Name) and 
                node.func.value.value.id == "sys"):
                # Simplification: assume adding project root or source dir
                # Real static analysis of the argument is hard
                new_roots.append(project_root)
                new_roots.append(project_root / "src")
    return list(set(new_roots))


def find_transitive_dependencies(entry_path: Path, project_root: Path) -> tuple[set[Path], dict[Path, set[Path]]]:
    """
    Recursive dependency discovery for Python scripts.
    Returns:
        all_deps: set of all discovered file paths
        import_graph: dict mapping importer -> set of imported files
    """
    entry_path = entry_path.resolve()
    all_deps = {entry_path}
    import_graph: dict[Path, set[Path]] = {}
    to_visit = [entry_path]
    visited_in_this_run = {entry_path}
    
    base_roots = [project_root, project_root / "src"]
    
    while to_visit:
        current_file = to_visit.pop()
        
        if current_file in _DEPENDENCY_CACHE:
            immediate_deps = _DEPENDENCY_CACHE[current_file]
        else:
            immediate_deps = set()
            local_roots = list(set(base_roots + extract_sys_path_additions(current_file, project_root)))
            
            abs_imports = set()
            rel_imports = []
            string_paths = set()
            
            tree = get_ast(current_file)
            
            for node in ast.walk(tree):
                # 1. Imports
                if isinstance(node, ast.Import):
                    for n in node.names:
                        abs_imports.add(n.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.level > 0:
                        rel_imports.append((node.module or "", node.level))
                    elif node.module:
                        abs_imports.add(node.module)
                # 2. Strings (potential file paths)
                elif isinstance(node, (ast.Str, ast.Constant)):
                    val = getattr(node, "s", None) if isinstance(node, ast.Str) else getattr(node, "value", None)
                    if isinstance(val, str) and 2 < len(val) < 255 and "\n" not in val:
                        if any(val.endswith(ext) for ext in SENSITIVE_EXTENSIONS) or "/" in val:
                            string_paths.add(val)
                # 3. subprocess.run/call/Popen, os.system/popen — extract file refs
                elif isinstance(node, ast.Call):
                    refs = _extract_subprocess_file_refs(node)
                    for ref in refs:
                        string_paths.add(ref)

            # Resolve Absolute Imports
            for mod in abs_imports:
                path = resolve_absolute_import(mod, local_roots)
                if path:
                    immediate_deps.add(path.resolve())

            # Resolve Relative Imports
            for mod, level in rel_imports:
                path = resolve_relative_import(current_file, mod, level)
                if path:
                    immediate_deps.add(path.resolve())

            # Resolve String Paths (e.g. config filenames)
            for s in string_paths:
                # Try relative to current file and relative to project root
                for base in [current_file.parent, project_root]:
                    p = (base / s).resolve()
                    try:
                        if p.exists() and p.is_file():
                            immediate_deps.add(p)
                    except OSError:
                        pass
            
            # Filter deps: must be within project, ignore hidden/venv
            immediate_deps = {
                p for p in immediate_deps 
                if str(p).startswith(str(project_root)) and 
                not any(x in str(p) for x in [".git", ".dvc", ".venv", "__pycache__", ".code_hash", ".dvc-viewer"])
            }
            _DEPENDENCY_CACHE[current_file] = immediate_deps

        import_graph[current_file] = immediate_deps

        # Add new deps to visit
        for p in immediate_deps:
            if p not in visited_in_this_run:
                all_deps.add(p)
                visited_in_this_run.add(p)
                if p.suffix == ".py":
                    to_visit.append(p)
                    
    return all_deps, import_graph


def compute_per_file_hashes(file_paths: set[Path], project_root: Path) -> dict[str, str]:
    """Compute mapping of relative path to content hash."""
    hashes = {}
    for path in file_paths:
        try:
            rel_path = str(path.relative_to(project_root))
        except ValueError:
            rel_path = str(path)
        
        try:
            h = hashlib.sha256()
            h.update(path.read_bytes())
            hashes[rel_path] = h.hexdigest()
        except Exception:
            pass
    return hashes


def find_import_chain(import_graph: dict[str, list[str]], target_file: str, entry_point: str) -> list[str] | None:
    """Find the import chain from target_file to entry_point using BFS."""
    if target_file == entry_point:
        return [entry_point]
    
    # BFS to find path from entry_point to target_file
    queue = [[entry_point]]
    visited = {entry_point}
    
    while queue:
        path = queue.pop(0)
        node = path[-1]
        
        if node == target_file:
            # We want modified_file -> ... -> entry_point
            return list(reversed(path))
            
        for neighbor in import_graph.get(node, []):
            if neighbor not in visited:
                visited.add(neighbor)
                new_path = list(path)
                new_path.append(neighbor)
                queue.append(new_path)
    
    return None


def compute_aggregate_hash(file_paths: set[Path], project_root: Path) -> str:
    """Compute an aggregate SHA256 hash of all file contents and relative paths."""
    hasher = hashlib.sha256()
    sorted_paths = sorted(list(file_paths))
    
    for path in sorted_paths:
        try:
            rel_path = path.relative_to(project_root)
        except ValueError:
            rel_path = path
        
        hasher.update(str(rel_path).encode("utf-8"))
        try:
            hasher.update(path.read_bytes())
        except Exception as e:
            logger.warning(f"Could not read {path}: {e}")
            
    return hasher.hexdigest()
