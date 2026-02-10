"""Tests for the dvc-viewer hook system."""

import os
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest

from dvc_viewer.updater import _find_project_python, update_dvc_yaml


# ---------------------------------------------------------------------------
# _find_project_python tests
# ---------------------------------------------------------------------------


def test_find_python_active_venv(tmp_path):
    """Priority 1: $VIRTUAL_ENV/bin/python."""
    fake_venv = tmp_path / "active_venv"
    python = fake_venv / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()

    with patch.dict(os.environ, {"VIRTUAL_ENV": str(fake_venv)}):
        result = _find_project_python(tmp_path)
    assert result == str(python)


def test_find_python_project_venv(tmp_path):
    """Priority 2: .venv/bin/python in project dir."""
    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.touch()

    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    with patch.dict(os.environ, env, clear=True):
        result = _find_project_python(tmp_path)
    assert result == str(python)


def test_find_python_system_fallback(tmp_path):
    """Priority 3: system python3."""
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    with patch.dict(os.environ, env, clear=True):
        result = _find_project_python(tmp_path)
    # Should return a valid python path
    assert "python" in result


# ---------------------------------------------------------------------------
# Hook execution tests
# ---------------------------------------------------------------------------


def _make_project(tmp_path, hook_code=None):
    """Create a minimal DVC project with an optional hook."""
    dvc_yaml = tmp_path / "dvc.yaml"
    dvc_yaml.write_text(textwrap.dedent("""\
        stages:
          train:
            cmd: python train.py
            deps:
              - train.py
    """))
    (tmp_path / "train.py").write_text("print('training')\n")

    if hook_code is not None:
        hooks_dir = tmp_path / ".dvc-viewer" / "hooks"
        hooks_dir.mkdir(parents=True)
        hook_file = hooks_dir / "post_hash.py"
        hook_file.write_text(hook_code)


def test_hook_runs_successfully(tmp_path, capsys):
    """A valid hook should execute and its output should be printed."""
    marker = tmp_path / "hook_ran.txt"
    hook_code = textwrap.dedent(f"""\
        from pathlib import Path
        Path("{marker}").write_text("yes")
        print("Hook executed!")
    """)
    _make_project(tmp_path, hook_code=hook_code)
    update_dvc_yaml(tmp_path)

    assert marker.exists()
    assert marker.read_text() == "yes"
    captured = capsys.readouterr()
    assert "Running post_hash hook" in captured.out
    assert "Hook executed!" in captured.out


def test_hook_failure_is_non_blocking(tmp_path, capsys):
    """A failing hook must not crash update_dvc_yaml."""
    hook_code = textwrap.dedent("""\
        import sys
        sys.exit(1)
    """)
    _make_project(tmp_path, hook_code=hook_code)

    # Should NOT raise
    update_dvc_yaml(tmp_path)

    captured = capsys.readouterr()
    assert "post_hash hook failed" in captured.out


def test_no_hooks_dir(tmp_path, capsys):
    """No hooks dir -> normal operation, no crash."""
    _make_project(tmp_path)
    update_dvc_yaml(tmp_path)

    captured = capsys.readouterr()
    assert "Running post_hash hook" not in captured.out
