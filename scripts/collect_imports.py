"""Collect top-level imports across the repository and print them.

Usage: python scripts/collect_imports.py
"""
import ast
import glob
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
mods = set()
for path in repo_root.glob('**/*.py'):
    # Skip virtualenv or hidden folders if present
    parts = path.parts
    if 'venv' in parts or '.venv' in parts:
        continue
    try:
        src = path.read_text(encoding='utf-8')
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    mods.add(n.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    mods.add(node.module.split('.')[0])
    except Exception:
        pass

# Heuristic stdlib-like modules to ignore
stdlib_like = {
    'sys','os','json','re','time','typing','glob','shutil','math','collections',
    'itertools','functools','pathlib','subprocess','logging','csv','statistics',
    'pprint','io','argparse','inspect','types'
}

external = sorted([m for m in mods if m not in stdlib_like])
print('\n'.join(external))
