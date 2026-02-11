from dvc_viewer.parser import build_pipeline, parse_dvc_lock, get_dvc_status
import json
from pathlib import Path

project_dir = Path("/home/lopilo/dvc-viewer-foreach-test")

locked = parse_dvc_lock(project_dir)
status = get_dvc_status(project_dir)

print(f"Locked stages: {locked}")
print(f"DVC Status: {status}")

pipeline = build_pipeline(project_dir)
print(f"Is running: {pipeline.is_running}")
print(f"Running stage: {pipeline.running_stage}")

print(f"Stages found: {list(pipeline.stages.keys())}")
for name, stage in pipeline.stages.items():
    print(f"\nStage: {name}")
    print(f"  Cmd: {stage.cmd}")
    print(f"  Deps: {stage.deps}")
    print(f"  Outs: {stage.outs}")
    print(f"  State: {stage.state}")

