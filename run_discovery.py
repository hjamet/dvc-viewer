import sys
import os

env = os.environ.copy()
env['DVC_GDRIVE_CREDENTIALS_DATA'] = open('/home/jules/real_credentials.json').read()

import subprocess
proc = subprocess.Popen(
    ["/app/.venv/bin/python3", "-c", "from dvc_viewer.gdrive import discover_dvc_folder; import os; print(discover_dvc_folder(os.environ['DVC_GDRIVE_CREDENTIALS_DATA']))"],
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

for _ in range(20):
    line = proc.stdout.readline()
    if not line:
        break
    print("LOG:", line.strip())

proc.terminate()
