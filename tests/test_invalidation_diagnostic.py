
import os
import shutil
import textwrap
from pathlib import Path
from dvc_viewer.updater import update_dvc_yaml

def test_hash_invalidation_diagnostic(tmp_path):
    project_dir = tmp_path / "project"
    project_dir.mkdir()
    
    # 1. Create a chain of imports: train.py -> lib.py -> utils.py
    (project_dir / "utils.py").write_text("def tool(): return 42")
    (project_dir / "lib.py").write_text("from utils import tool\ndef helper(): return tool()")
    (project_dir / "train.py").write_text("from lib import helper\nprint(helper())")
    
    # Create dvc.yaml
    (project_dir / "dvc.yaml").write_text(textwrap.dedent("""\
        stages:
          train:
            cmd: python train.py
    """))
    
    print("\n--- First run (compute initial hashes) ---")
    update_dvc_yaml(project_dir)
    
    # 2. Modify utils.py (the deep dependency)
    print("\n--- Modifying utils.py ---")
    (project_dir / "utils.py").write_text("def tool(): return 43")
    
    print("\n--- Second run (should show diagnostic) ---")
    # Capture output
    import io
    from contextlib import redirect_stdout
    f = io.StringIO()
    with redirect_stdout(f):
        update_dvc_yaml(project_dir)
    
    output = f.getvalue()
    print(output)
    
    # 3. Assertions
    import re
    def strip_ansi(text):
        return re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])').sub('', text)
    
    clean_output = strip_ansi(output)
    print("Clean output:")
    print(clean_output)
    
    assert "Stage 'train' invalidated by code change" in clean_output
    assert "utils.py (modified)" in clean_output
    assert "→ lib.py" in clean_output
    assert "→ train.py" in clean_output

if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        test_hash_invalidation_diagnostic(Path(tmp))
