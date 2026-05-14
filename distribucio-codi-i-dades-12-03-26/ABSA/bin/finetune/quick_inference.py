#!/usr/bin/env python3
"""Quick inference with 10 examples to generate FT.devel.json sample."""
import subprocess
import sys
import os
from pathlib import Path

ABSA_DIR = Path(__file__).resolve().parent.parent.parent
os.chdir(str(ABSA_DIR))

# Run inference with limit=10
cmd = [
    sys.executable,
    str(ABSA_DIR / "bin" / "finetune" / "finetune-inference.py"),
    "--weights", str(ABSA_DIR / "outputs" / "FT.train.fewshot.weights"),
    "--data", "devel",
    "--output", str(ABSA_DIR / "outputs" / "FT.devel.sample.json"),
    "--limit", "10"
]

print(f"Running: {' '.join(cmd)}")
result = subprocess.run(cmd, capture_output=False)
sys.exit(result.returncode)
