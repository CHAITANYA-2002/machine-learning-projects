"""Collect imports from .py and .ipynb files and print unique top-level modules."""
import ast
import glob
import json
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
mods = set()

# scan .py files
for path in repo_root.glob('**/*.py'):
    if 'venv' in path.parts or '.venv' in path.parts:
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

# scan .ipynb files
for path in repo_root.glob('**/*.ipynb'):
    try:
        nb = json.loads(path.read_text(encoding='utf-8'))
        for cell in nb.get('cells', []):
            if cell.get('cell_type')!='code':
                continue
            source = ''.join(cell.get('source', []))
            try:
                tree = ast.parse(source)
            except Exception:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for n in node.names:
                        mods.add(n.name.split('.')[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        mods.add(node.module.split('.')[0])
    except Exception:
        pass

# Filter stdlib-ish modules
stdlib_like = {
    'sys','os','json','re','time','typing','glob','shutil','math','collections',
    'itertools','functools','pathlib','subprocess','logging','csv','statistics',
    'pprint','io','argparse','inspect','types'
}

external = sorted([m for m in mods if m not in stdlib_like])
print('Detected top-level modules across repo:')
for m in external:
    print(m)
