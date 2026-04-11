import subprocess
import os

env = os.environ.copy()
env['DVC_GDRIVE_CREDENTIALS_DATA'] = open('/home/jules/test_credentials.json').read()

proc = subprocess.Popen(
    ["/app/.venv/bin/dvc", "push", "-v"],
    cwd="/home/jules/dvc-viewer-realtest",
    env=env,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

for _ in range(20):
    line = proc.stdout.readline()
    if not line:
        break
    print(line.strip())

proc.terminate()
proc.wait()
