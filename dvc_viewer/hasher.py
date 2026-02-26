"""
Code hashing module for change detection.

Analyzes Python scripts to find transitive dependencies (imports, relative imports)
and computes an aggregate hash. Adapted from the CLIMB Transitive Code Hasher.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import logging
import os
import symtable
import sys
from pathlib import Path
from typing import Set, Tuple, Dict, Any

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


def clear_caches():
    """Clear the internal AST and dependency caches."""
    _DEPENDENCY_CACHE.clear()
    _AST_CACHE.clear()


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


def _strip_docstrings(tree: ast.AST) -> None:
    """Recursively remove docstrings from a module, class, or function."""
    if hasattr(tree, "body") and isinstance(tree.body, list):
        # Docstring is an Expr node containing a Constant string as the first element
        if (tree.body and isinstance(tree.body[0], ast.Expr) and 
            isinstance(tree.body[0].value, (ast.Str, ast.Constant))):
            val = getattr(tree.body[0].value, "s", None) if isinstance(tree.body[0].value, ast.Str) else getattr(tree.body[0].value, "value", None)
            if isinstance(val, str):
                tree.body.pop(0)
        
        # Recurse into body elements
        for node in tree.body:
            _strip_docstrings(node)


def _compute_python_hash(path: Path) -> str:
    """Compute a hash of the executable code in a Python file (ignores docstrings/comments)."""
    try:
        # Use existing cached AST
        tree = get_ast(path)
        # Deepcopy to avoid mutating the cached AST when stripping docstrings
        tree_copy = copy.deepcopy(tree)
        _strip_docstrings(tree_copy)
        # ast.unparse() eliminates line numbers/whitespace/comments
        # while keeping the normalized structure of the code.
        normalized = ast.unparse(tree_copy)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    except Exception as e:
        logger.warning(f"AST hashing failed for {path}, falling back to raw bytes: {e}")
        try:
            return hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            return ""


def _compute_file_hash(path: Path, symbols: set[str] | None = None) -> str:
    """
    Computes hash using AST for .py files, and read_bytes for others.
    If symbols is provided, only hash those symbols and their closure.
    """
    if path.suffix == ".py":
        if symbols is not None:
            return _compute_symbol_level_hash(path, symbols)
        return _compute_python_hash(path)
    try:
        content = path.read_bytes()
        # Normalize CRLF to LF to avoid issues with git core.autocrlf on Windows
        content = content.replace(b'\r\n', b'\n')
        return hashlib.sha256(content).hexdigest()
    except Exception as e:
        logger.warning(f"Failed to read {path}: {e}")
        return ""


def _resolve_symbol_closure(path: Path, symbol_names: set[str]) -> set[str]:
    """Find all internal globals/functions referenced by the given symbols using symtable."""
    try:
        code = path.read_text(encoding="utf-8")
        table = symtable.symtable(code, str(path), "exec")
    except Exception:
        return symbol_names

    closure = set(symbol_names)
    to_check = list(symbol_names)
    checked = set()

    # Get all top-level symbols defined in this file
    top_level_defs = {s.get_name() for s in table.get_symbols()}

    while to_check:
        name = to_check.pop()
        if name in checked:
            continue
        checked.add(name)

        try:
            # Look for function/class scope
            symbol_scope = None
            for child in table.get_children():
                if child.get_name() == name:
                    symbol_scope = child
                    break
            
            if symbol_scope:
                # Find all names referenced in this scope that are globals in this module
                for ref_name in symbol_scope.get_globals():
                    if ref_name in top_level_defs and ref_name not in closure:
                        closure.add(ref_name)
                        to_check.append(ref_name)
        except Exception:
            pass
            
    return closure


def _compute_symbol_level_hash(path: Path, symbols: set[str]) -> str:
    """Hash only specific symbols and their transitive closure in a Python file."""
    try:
        tree = get_ast(path)
        closure = _resolve_symbol_closure(path, symbols)
        
        # Filter AST nodes
        needed_nodes = []
        for node in tree.body:
            name = None
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                name = node.name
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        name = target.id
                        break
            elif isinstance(node, ast.AnnAssign):
                if isinstance(node.target, ast.Name):
                    name = node.target.id
            
            if name in closure:
                # Copy and strip docstring
                node_copy = copy.deepcopy(node)
                _strip_docstrings(node_copy)
                needed_nodes.append(ast.unparse(node_copy))
        
        if not needed_nodes:
            # If no nodes found (e.g. symbols are imports), hash the whole file as fallback
            return _compute_python_hash(path)
            
        combined = "".join(sorted(needed_nodes))
        return hashlib.sha256(combined.encode("utf-8")).hexdigest()
    except Exception as e:
        logger.warning(f"Symbol hashing failed for {path}: {e}")
        return _compute_python_hash(path)


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


def find_transitive_dependencies(
    entry_path: Path, 
    project_root: Path
) -> tuple[set[Path], dict[Path, set[Path]], dict[tuple[Path, Path], set[str] | None]]:
    """
    Recursive dependency discovery for Python scripts.
    Returns:
        all_deps: set of all discovered file paths
        import_graph: dict mapping importer -> set of imported files
        import_names: dict mapping (importer, imported) -> set of imported symbols (None means full module)
    """
    entry_path = entry_path.resolve()
    all_deps = {entry_path}
    import_graph: dict[Path, set[Path]] = {}
    import_names: dict[tuple[Path, Path], set[str] | None] = {}
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
                        abs_imports.add((n.name, None))
                elif isinstance(node, ast.ImportFrom):
                    if node.level > 0:
                        names = frozenset(n.name for n in node.names) if "*" not in [n.name for n in node.names] else None
                        rel_imports.append((node.module or "", node.level, names))
                    elif node.module:
                        names = frozenset(n.name for n in node.names) if "*" not in [n.name for n in node.names] else None
                        abs_imports.add((node.module, names))
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
            for mod, names in abs_imports:
                path = resolve_absolute_import(mod, local_roots)
                if path:
                    p_res = path.resolve()
                    immediate_deps.add(p_res)
                    key = (current_file, p_res)
                    if names is None:
                        import_names[key] = None
                    else:
                        import_names.setdefault(key, set()).update(names)

            # Resolve Relative Imports
            for mod, level, names in rel_imports:
                path = resolve_relative_import(current_file, mod, level)
                if path:
                    p_res = path.resolve()
                    immediate_deps.add(p_res)
                    key = (current_file, p_res)
                    if names is None:
                        import_names[key] = None
                    else:
                        import_names.setdefault(key, set()).update(names)

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
                    
    return all_deps, import_graph, import_names


def compute_per_file_hashes(
    file_paths: set[Path], 
    project_root: Path,
    import_names: dict[tuple[Path, Path], set[str] | None] | None = None,
    entry_point: Path | None = None
) -> dict[str, str]:
    """Compute mapping of relative path to content hash."""
    # If import_names and entry_point are provided, calculate needed symbols for Phase 2
    needed_symbols: dict[Path, set[str] | None] = {}
    if import_names and entry_point:
        entry_point = entry_point.resolve()
        needed_symbols[entry_point] = None  # None means hash the whole executable file
        
        # BFS through the import graph to propagate needed symbols
        to_process = [entry_point]
        processed = {entry_point}
        while to_process:
            importer = to_process.pop(0)
            # Find all imported files from this importer in import_names
            for (curr_importer, imported), symbols in import_names.items():
                if curr_importer == importer:
                    # Get the closure of symbols actually needed in the importer
                    importer_needed = needed_symbols.get(importer)
                    
                    if importer_needed is None:
                        # Full module analyzed, take all its specific imports
                        if imported not in needed_symbols or needed_symbols[imported] is not None:
                            if symbols is None:
                                needed_symbols[imported] = None
                            else:
                                needed_symbols.setdefault(imported, set()).update(symbols)
                    else:
                        # Only specific symbols were needed. 
                        # We must check if 'symbols' (what we import from 'imported')
                        # are actually referenced by 'importer_needed'.
                        importer_closure = _resolve_symbol_closure(importer, importer_needed)
                        
                        if symbols is None:
                            # If we 'import imported', we keep it as None ONLY if ANY name 
                            # from 'imported' is in importer_closure. 
                            # Since we don't know the names in 'imported' without parsing it, 
                            # we stay conservative and keep None if importer_closure is not empty.
                            needed_symbols[imported] = None
                        else:
                            # Only propagate symbols that are actually used in importer's closure
                            actual_needed = symbols.intersection(importer_closure)
                            if actual_needed:
                                needed_symbols.setdefault(imported, set()).update(actual_needed)
                    
                    if imported not in processed:
                        processed.add(imported)
                        to_process.append(imported)

    hashes = {}
    for path in file_paths:
        try:
            rel_path = path.relative_to(project_root).as_posix()
        except ValueError:
            rel_path = path.as_posix()
        
        symbols = needed_symbols.get(path)
        h = _compute_file_hash(path, symbols=symbols)
        if h:
            hashes[rel_path] = h
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


def compute_aggregate_hash(
    file_paths: set[Path], 
    project_root: Path,
    import_names: dict[tuple[Path, Path], set[str] | None] | None = None,
    entry_point: Path | None = None,
    precomputed_hashes: dict[str, str] | None = None
) -> str:
    """Compute an aggregate SHA256 hash of all file contents and relative paths."""
    # Reuse precomputed hashes if provided to avoid double calculation
    file_hashes = precomputed_hashes if precomputed_hashes is not None else \
                  compute_per_file_hashes(file_paths, project_root, import_names, entry_point)
    
    hasher = hashlib.sha256()
    sorted_paths = sorted(list(file_paths))
    
    for path in sorted_paths:
        try:
            rel_path = path.relative_to(project_root).as_posix()
        except ValueError:
            rel_path = path.as_posix()
        
        hasher.update(rel_path.encode("utf-8"))
        h = file_hashes.get(rel_path)
        if h:
            hasher.update(h.encode("utf-8"))
            
    return hasher.hexdigest()
