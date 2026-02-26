
from pathlib import Path
import pytest
import ast
from dvc_viewer.hasher import (
    _compute_python_hash, _compute_file_hash, _strip_docstrings, 
    clear_caches, find_transitive_dependencies, compute_aggregate_hash
)

@pytest.fixture(autouse=True)
def cleanup_cache():
    clear_caches()
    yield
    clear_caches()

# --- Phase 1 Tests ---

def test_strip_docstrings():
    code = """
def foo():
    \"\"\"Docstring to remove.\"\"\"
    return 42
"""
    tree = ast.parse(code)
    _strip_docstrings(tree)
    
    # Verify docstring is gone
    func_node = tree.body[0]
    assert len(func_node.body) == 1
    assert isinstance(func_node.body[0], ast.Return)

def test_comment_change_no_invalidation(tmp_path):
    file1 = tmp_path / "script.py"
    file1.write_text("def foo():\n    return 42\n")
    clear_caches()
    hash1 = _compute_python_hash(file1)
    
    file1.write_text("def foo():\n    # This is a comment\n    return 42\n")
    clear_caches()
    hash2 = _compute_python_hash(file1)
    
    assert hash1 == hash2

def test_docstring_change_no_invalidation(tmp_path):
    file1 = tmp_path / "script.py"
    file1.write_text("def foo():\n    \"\"\"Old doc.\"\"\"\n    return 42\n")
    clear_caches()
    hash1 = _compute_python_hash(file1)
    
    file1.write_text("def foo():\n    \"\"\"New doc string.\"\"\"\n    return 42\n")
    clear_caches()
    hash2 = _compute_python_hash(file1)
    
    assert hash1 == hash2

def test_whitespace_change_no_invalidation(tmp_path):
    file1 = tmp_path / "script.py"
    file1.write_text("def foo():\n    return 42\n")
    clear_caches()
    hash1 = _compute_python_hash(file1)
    
    file1.write_text("def foo():\n\n    return    42\n")
    clear_caches()
    hash2 = _compute_python_hash(file1)
    
    assert hash1 == hash2

def test_code_change_invalidation(tmp_path):
    file1 = tmp_path / "script.py"
    file1.write_text("def foo():\n    return 42\n")
    clear_caches()
    hash1 = _compute_python_hash(file1)
    
    file1.write_text("def foo():\n    return 43\n")
    clear_caches()
    hash2 = _compute_python_hash(file1)
    
    assert hash1 != hash2

def test_non_python_file_invalidation(tmp_path):
    file1 = tmp_path / "config.yaml"
    file1.write_text("key: value\n")
    clear_caches()
    hash1 = _compute_file_hash(file1)
    
    file1.write_text("key: value # comment\n")
    clear_caches()
    hash2 = _compute_file_hash(file1)
    
    # YAML files don't use AST hashing, so comments IN THE FILE change the raw bytes
    assert hash1 != hash2

def test_type_annotation_change_invalidation(tmp_path):
    file1 = tmp_path / "script.py"
    file1.write_text("def foo(x: int):\n    return x\n")
    clear_caches()
    hash1 = _compute_python_hash(file1)
    
    file1.write_text("def foo(x: float):\n    return x\n")
    clear_caches()
    hash2 = _compute_python_hash(file1)
    
    assert hash1 != hash2

# --- Phase 2 Tests ---

def test_symbol_level_unimported_change(tmp_path):
    # lib.py has foo() and bar()
    # train.py imports ONLY foo()
    lib_path = tmp_path / "lib.py"
    lib_path.write_text("def foo(): return 1\ndef bar(): return 2\n")
    train_path = tmp_path / "train.py"
    train_path.write_text("from lib import foo\nprint(foo())\n")
    
    clear_caches()
    deps, graph, names = find_transitive_dependencies(train_path, tmp_path)
    hash1 = compute_aggregate_hash(deps, tmp_path, import_names=names, entry_point=train_path)
    
    # Modify bar() (not imported)
    lib_path.write_text("def foo(): return 1\ndef bar(): return 3\n")
    clear_caches()
    deps, graph, names = find_transitive_dependencies(train_path, tmp_path)
    hash2 = compute_aggregate_hash(deps, tmp_path, import_names=names, entry_point=train_path)
    
    assert hash1 == hash2

def test_symbol_level_imported_change(tmp_path):
    lib_path = tmp_path / "lib.py"
    lib_path.write_text("def foo(): return 1\ndef bar(): return 2\n")
    train_path = tmp_path / "train.py"
    train_path.write_text("from lib import foo\nprint(foo())\n")
    
    clear_caches()
    deps, graph, names = find_transitive_dependencies(train_path, tmp_path)
    hash1 = compute_aggregate_hash(deps, tmp_path, import_names=names, entry_point=train_path)
    
    # Modify foo() (imported)
    lib_path.write_text("def foo(): return 11\ndef bar(): return 2\n")
    clear_caches()
    deps, graph, names = find_transitive_dependencies(train_path, tmp_path)
    hash2 = compute_aggregate_hash(deps, tmp_path, import_names=names, entry_point=train_path)
    
    assert hash1 != hash2

def test_symbol_closure_transitive(tmp_path):
    # foo() uses _helper()
    lib_path = tmp_path / "lib.py"
    lib_path.write_text("def _helper(): return 10\ndef foo(): return _helper()\ndef bar(): return 2\n")
    train_path = tmp_path / "train.py"
    train_path.write_text("from lib import foo\nprint(foo())\n")
    
    clear_caches()
    deps, graph, names = find_transitive_dependencies(train_path, tmp_path)
    hash1 = compute_aggregate_hash(deps, tmp_path, import_names=names, entry_point=train_path)
    
    # Modify _helper() (not imported in train.py, but used by foo())
    lib_path.write_text("def _helper(): return 20\ndef foo(): return _helper()\ndef bar(): return 2\n")
    clear_caches()
    deps, graph, names = find_transitive_dependencies(train_path, tmp_path)
    hash2 = compute_aggregate_hash(deps, tmp_path, import_names=names, entry_point=train_path)
    
    assert hash1 != hash2

def test_import_star_fallback(tmp_path):
    lib_path = tmp_path / "lib.py"
    lib_path.write_text("def foo(): return 1\ndef bar(): return 2\n")
    train_path = tmp_path / "train.py"
    train_path.write_text("from lib import *\nprint(foo())\n")
    
    clear_caches()
    deps, graph, names = find_transitive_dependencies(train_path, tmp_path)
    hash1 = compute_aggregate_hash(deps, tmp_path, import_names=names, entry_point=train_path)
    
    # Modify bar() (not used, but wildcard import = fallback to full file hash)
    lib_path.write_text("def foo(): return 1\ndef bar(): return 3\n")
    clear_caches()
    deps, graph, names = find_transitive_dependencies(train_path, tmp_path)
    hash2 = compute_aggregate_hash(deps, tmp_path, import_names=names, entry_point=train_path)
    
    assert hash1 != hash2
def test_symbol_chain_3_levels(tmp_path):
    # C.py : helper()
    # B.py : from C import helper; def foo(): return helper(); def bar(): return 0
    # A.py : from B import foo; foo()
    
    c_path = tmp_path / "c.py"
    c_path.write_text("def helper(): return 1\ndef unused_c(): return 2\n")
    
    b_path = tmp_path / "b.py"
    b_path.write_text("from c import helper\ndef foo(): return helper()\ndef bar(): return 0\n")
    
    a_path = tmp_path / "a.py"
    a_path.write_text("from b import foo\nprint(foo())\n")
    
    clear_caches()
    deps, graph, names = find_transitive_dependencies(a_path, tmp_path)
    hash1 = compute_aggregate_hash(deps, tmp_path, import_names=names, entry_point=a_path)
    
    # Modify unused_c (not in closure of foo)
    c_path.write_text("def helper(): return 1\ndef unused_c(): return 999\n")
    clear_caches()
    deps, graph, names = find_transitive_dependencies(a_path, tmp_path)
    hash2 = compute_aggregate_hash(deps, tmp_path, import_names=names, entry_point=a_path)
    
    assert hash1 == hash2
    
    # Modify helper (in closure of foo)
    c_path.write_text("def helper(): return 2\ndef unused_c(): return 999\n")
    clear_caches()
    deps, graph, names = find_transitive_dependencies(a_path, tmp_path)
    hash3 = compute_aggregate_hash(deps, tmp_path, import_names=names, entry_point=a_path)
    
    assert hash1 != hash3


def test_hash_deterministic_golden(tmp_path):
    """
    Vérifie que le hash produit pour un code donné est strictement identique
    à une valeur de référence pré-calculée. Cela garantit la stabilité
    cross-version (via ast.unparse).
    """
    file_path = tmp_path / "golden.py"
    # Code très simple dont on a calculé le hash stable
    code = "def foo():\n    return 42\n"
    file_path.write_text(code)
    
    clear_caches()
    h = _compute_python_hash(file_path)
    
    # Valeur attendue calculée avec ast.unparse() + sha256
    expected = "ae7983a21a9e3a4ebc86f2cfad314696371ca76d16d07ccdaeed227f9c9bc8b6"
    assert h == expected
