"""
Modified from https://github.com/microsoft/Graphormer
"""

from pathlib import Path
import importlib

# automatically import any Python files in the criterions/ directory
for file in sorted(Path(__file__).parent.glob("*.py")):
    if not file.name.startswith("_"):
        importlib.import_module("tokengt.criterions." + file.name[:-3])
