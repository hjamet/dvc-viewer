import os
import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock dependencies before importing server
sys.modules["fastapi"] = MagicMock()
sys.modules["fastapi.responses"] = MagicMock()
sys.modules["fastapi.staticfiles"] = MagicMock()
sys.modules["yaml"] = MagicMock()

from dvc_viewer.server import _safe_resolve

def test_safe_resolve_happy_path(tmp_path):
    """Test that a valid relative path is correctly resolved."""
    import dvc_viewer.server
    original_project_dir = dvc_viewer.server._project_dir
    dvc_viewer.server._project_dir = str(tmp_path)

    try:
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello")

        resolved = _safe_resolve("test.txt")
        assert resolved == test_file.resolve()
    finally:
        dvc_viewer.server._project_dir = original_project_dir

def test_safe_resolve_traversal(tmp_path):
    """Test that path traversal attempts return None."""
    import dvc_viewer.server
    original_project_dir = dvc_viewer.server._project_dir
    dvc_viewer.server._project_dir = str(tmp_path)

    try:
        outside_dir = tmp_path.parent / "outside"
        outside_dir.mkdir(exist_ok=True)
        outside_file = outside_dir / "secret.txt"
        outside_file.write_text("secret")

        assert _safe_resolve("../outside/secret.txt") is None
    finally:
        dvc_viewer.server._project_dir = original_project_dir

def test_safe_resolve_absolute(tmp_path):
    """Test that absolute paths outside project return None."""
    import dvc_viewer.server
    original_project_dir = dvc_viewer.server._project_dir
    dvc_viewer.server._project_dir = str(tmp_path)

    try:
        assert _safe_resolve("/etc/passwd") is None
    finally:
        dvc_viewer.server._project_dir = original_project_dir

def test_safe_resolve_empty_none(tmp_path):
    """Test that empty or None paths return None."""
    import dvc_viewer.server
    original_project_dir = dvc_viewer.server._project_dir
    dvc_viewer.server._project_dir = str(tmp_path)

    try:
        assert _safe_resolve("") is None
        assert _safe_resolve(None) is None
    finally:
        dvc_viewer.server._project_dir = original_project_dir

def test_safe_resolve_non_existent(tmp_path):
    """Test that non-existent files return None when require_exists=True."""
    import dvc_viewer.server
    original_project_dir = dvc_viewer.server._project_dir
    dvc_viewer.server._project_dir = str(tmp_path)

    try:
        assert _safe_resolve("missing.txt", require_exists=True) is None
        resolved = _safe_resolve("missing.txt", require_exists=False)
        assert resolved == (tmp_path / "missing.txt").resolve()
    finally:
        dvc_viewer.server._project_dir = original_project_dir

def test_safe_resolve_null_byte(tmp_path):
    """Test that paths with null bytes are handled safely and return None."""
    import dvc_viewer.server
    original_project_dir = dvc_viewer.server._project_dir
    dvc_viewer.server._project_dir = str(tmp_path)

    try:
        assert _safe_resolve("file.txt\0something") is None
    finally:
        dvc_viewer.server._project_dir = original_project_dir

if __name__ == "__main__":
    import tempfile
    import shutil
    tmp = Path(tempfile.mkdtemp())
    try:
        test_safe_resolve_happy_path(tmp)
        test_safe_resolve_traversal(tmp)
        test_safe_resolve_absolute(tmp)
        test_safe_resolve_empty_none(tmp)
        test_safe_resolve_non_existent(tmp)
        test_safe_resolve_null_byte(tmp)
        print("All tests passed!")
    finally:
        shutil.rmtree(tmp)
