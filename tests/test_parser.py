"""Tests for DVC pipeline parser."""

import textwrap
from pathlib import Path

from dvc_viewer.parser import parse_dvc_yaml, _load_params, _resolve_interpolation


# ---------------------------------------------------------------------------
# _resolve_interpolation tests
# ---------------------------------------------------------------------------


def test_resolve_simple_var():
    """${key} resolves to the value in params."""
    params = {"datasets": ["a", "b", "c"]}
    assert _resolve_interpolation("${datasets}", params) == ["a", "b", "c"]


def test_resolve_nested_var():
    """${nested.key} traverses dot-separated path."""
    params = {"config": {"items": [1, 2]}}
    assert _resolve_interpolation("${config.items}", params) == [1, 2]


def test_resolve_unknown_returns_raw():
    """Unresolvable ${var} is returned as-is."""
    params = {"other": 42}
    assert _resolve_interpolation("${missing}", params) == "${missing}"


def test_resolve_non_string_passthrough():
    """Non-string values pass through unchanged."""
    params = {}
    assert _resolve_interpolation(["a", "b"], params) == ["a", "b"]
    assert _resolve_interpolation(42, params) == 42


# ---------------------------------------------------------------------------
# _load_params tests
# ---------------------------------------------------------------------------


def test_load_params_from_params_yaml(tmp_path):
    """Loads parameters from params.yaml."""
    (tmp_path / "params.yaml").write_text("datasets:\n  - x\n  - y\n")
    params = _load_params(tmp_path, {})
    assert params == {"datasets": ["x", "y"]}


def test_load_params_with_vars_file(tmp_path):
    """Loads parameters from vars file declared in dvc.yaml."""
    (tmp_path / "config.yaml").write_text("lr: 0.01\n")
    dvc_data = {"vars": ["config.yaml"]}
    params = _load_params(tmp_path, dvc_data)
    assert params["lr"] == 0.01


def test_load_params_with_inline_vars(tmp_path):
    """Inline dict vars are merged into params."""
    dvc_data = {"vars": [{"batch_size": 32}]}
    params = _load_params(tmp_path, dvc_data)
    assert params["batch_size"] == 32


# ---------------------------------------------------------------------------
# parse_dvc_yaml with parameterized foreach
# ---------------------------------------------------------------------------


def test_foreach_with_param_interpolation(tmp_path):
    """foreach: ${datasets} resolves from params.yaml and expands stages."""
    (tmp_path / "params.yaml").write_text("datasets:\n  - alpha\n  - beta\n  - gamma\n")
    (tmp_path / "dvc.yaml").write_text(textwrap.dedent("""\
        stages:
          download:
            foreach: ${datasets}
            do:
              cmd: echo ${item}
              deps:
                - "data/${item}.txt"
              outs:
                - "out/${item}.csv"
    """))

    stages = parse_dvc_yaml(tmp_path)
    names = sorted(stages.keys())
    assert names == ["download@alpha", "download@beta", "download@gamma"]

    # Check substitution worked in deps/outs
    alpha = stages["download@alpha"]
    assert "data/alpha.txt" in alpha.deps
    assert "out/alpha.csv" in alpha.outs


def test_foreach_with_inline_list_still_works(tmp_path):
    """Inline foreach: [a, b] continues to work (regression check)."""
    (tmp_path / "dvc.yaml").write_text(textwrap.dedent("""\
        stages:
          process:
            foreach: [fast, slow]
            do:
              cmd: echo ${item}
    """))

    stages = parse_dvc_yaml(tmp_path)
    assert sorted(stages.keys()) == ["process@fast", "process@slow"]


def test_foreach_with_vars_section(tmp_path):
    """foreach: ${models} resolves from a vars file."""
    conf_dir = tmp_path / "config"
    conf_dir.mkdir()
    (conf_dir / "models.yaml").write_text("models:\n  - bert\n  - gpt\n")
    (tmp_path / "dvc.yaml").write_text(textwrap.dedent("""\
        vars:
          - config/models.yaml
        stages:
          train:
            foreach: ${models}
            do:
              cmd: python train.py --model ${item}
    """))

    stages = parse_dvc_yaml(tmp_path)
    assert sorted(stages.keys()) == ["train@bert", "train@gpt"]
