import subprocess
import sys
import os

SRC_DIR = os.path.dirname(__file__)

steps = [
    "preprocess_ids.py",
    "preprocess_static.py",
    "preprocess_timeseries.py",
    "preprocess_labels.py",
    "preprocess_merge.py"
]

for step in steps:
    path = os.path.join(SRC_DIR, step)
    print("Running:", path)
    r = subprocess.run([sys.executable, path], check=False)
    if r.returncode != 0:
        print(f"Step {step} exited with code {r.returncode}. Stopping.")
        break
print("Preprocessing run complete (or stopped on error). Check data_processed/intermediate and final for outputs and diagnostics.")
