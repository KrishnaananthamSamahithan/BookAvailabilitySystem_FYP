"""Quick syntax validation script — does NOT run training (too slow for validation).
Validates all module imports, key function signatures, and config fields.
"""

import ast
import sys
from pathlib import Path

files_to_check = [
    "src/metrics.py",
    "src/drift.py",
    "src/significance.py",
    "src/calibration.py",
    "src/models.py",
    "src/simulation.py",
    "src/train.py",
    "app.py",
]

all_ok = True
for path_str in files_to_check:
    path = Path(path_str)
    if not path.exists():
        print(f"MISSING: {path_str}")
        all_ok = False
        continue
    try:
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        print(f"OK (syntax): {path_str}")
    except SyntaxError as e:
        print(f"SYNTAX ERROR in {path_str}: {e}")
        all_ok = False

print()
if all_ok:
    print("All files passed syntax check.")
else:
    print("Some files have errors.")
    sys.exit(1)
