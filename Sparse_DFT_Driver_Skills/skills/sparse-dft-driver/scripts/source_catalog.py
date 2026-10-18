#!/usr/bin/env python3
"""Discover Sparse-DFT contracts from source without importing or executing it."""
from __future__ import annotations
import ast
import hashlib
import json
import re
from pathlib import Path


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1048576), b''): h.update(chunk)
    return h.hexdigest()


def literal(node):
    try: return ast.literal_eval(node)
    except (ValueError, TypeError): return ast.unparse(node)


def source_catalog(root):
    root = Path(root).resolve()
    tree = ast.parse((root / 'sparse_dft/cli.py').read_text())
    delegated = next(literal(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == '_DELEGATED' for t in n.targets))
    cli = {k: {'source': 'sparse_dft/cli.py', 'delegated_module': v[0]} for k,v in delegated.items()}
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_parser' and n.args and isinstance(n.args[0], ast.Constant):
            cli[literal(n.args[0])] = {'source':'sparse_dft/cli.py', 'line':n.lineno}
    api_tree = ast.parse((root/'sparse_dft/api.py').read_text())
    helpers = {n.name:n for n in api_tree.body if isinstance(n, ast.FunctionDef)}
    def parameters(nodes, visited=()):
        params, required = set(), set()
        for n in ast.walk(ast.Module(body=nodes, type_ignores=[])):
            if isinstance(n, ast.Call):
                if isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name) and n.func.value.id=='params' and n.func.attr=='get' and n.args:
                    params.add(literal(n.args[0]))
                elif isinstance(n.func, ast.Name) and n.func.id in helpers and n.func.id not in visited and any(isinstance(x, ast.Name) and x.id=='params' for x in n.args):
                    p,r=parameters(helpers[n.func.id].body, (*visited,n.func.id));params |= p;required |= r
            if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name) and n.value.id=='params':
                key=literal(n.slice);params.add(key);required.add(key)
        return params, required
    methods={}
    for n in ast.walk(helpers['dispatch']):
        if not isinstance(n, ast.If) or not isinstance(n.test, ast.Compare) or not isinstance(n.test.left,ast.Name) or n.test.left.id!='method':continue
        value=literal(n.test.comparators[0]);names=[value] if isinstance(value,str) else value
        params,required=parameters(n.body)
        for name in names:
            controls=params.copy()
            if name=='relax.ionic':controls-={'stress_tolerance','max_strain'}
            if name=='relax.cell':controls-={'force_tolerance','max_displacement'}
            methods[name]={'parameters':sorted(controls),'required':sorted(required),'status':'unsupported' if name.startswith('results.') else 'placeholder' if name=='job.status' else 'implemented','source':'sparse_dft/api.py','line':n.lineno}
    functions={}
    for file in sorted((root/'sparse_dft').rglob('*.py')):
        if any(p.startswith('_') for p in file.relative_to(root/'sparse_dft').parts):continue
        module='.'.join(file.relative_to(root).with_suffix('').parts)
        for n in ast.parse(file.read_text()).body:
            if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and not n.name.startswith('_'):
                functions[f'{module}:{n.name}']={'kind':'class' if isinstance(n,ast.ClassDef) else 'function','signature':ast.unparse(n.args) if isinstance(n,ast.FunctionDef) else 'inspect constructor','positional_parameters':[x.arg for x in n.args.posonlyargs+n.args.args] if isinstance(n,ast.FunctionDef) else [],'source':str(file.relative_to(root)),'line':n.lineno}
    tools={}
    for folder in ('scripts','benchmarks/scripts','packaging'):
        for file in sorted((root/folder).glob('*.py')):
            content=file.read_text();t=ast.parse(content)
            flags=sorted({literal(c.args[0]) for c in ast.walk(t) if isinstance(c,ast.Call) and isinstance(c.func,ast.Attribute) and c.func.attr=='add_argument' and c.args and isinstance(literal(c.args[0]),str)})
            tools[str(file.relative_to(root))]={'source':str(file.relative_to(root)),'sha256':sha256(file),'arguments':flags,'has_argparse':'argparse' in content,'review_required':True,'source_relative_writes_possible':True}
    native=sorted(set(re.findall(r'add_executable\(\s*(sdft_[\w]+)',(root/'CMakeLists.txt').read_text())))
    env={}
    for folder in ('sparse_dft','src','scripts'):
        for file in sorted((root/folder).rglob('*')):
            if file.is_file() and file.suffix in ('.py','.cpp','.hpp','.mm','.cu','.cuh'):
                for key in sorted(set(re.findall(r'\bSDFT_[A-Z0-9_]+\b',file.read_text()))):env.setdefault(key,[]).append(str(file.relative_to(root)))
    identity={p:sha256(root/p) for p in ('sparse_dft/cli.py','sparse_dft/api.py','sparse_dft/scf.py','sparse_dft/settings.py','sparse_dft/sparsity.py','sparse_dft/sparsity_rules.json','sparse_dft/spin_model.py','sparse_dft/parallel.py','sparse_dft/runner.py','CMakeLists.txt','pyproject.toml')}
    return {'schema':'sparse-dft-skill-catalog-2','identity':identity,'cli':dict(sorted(cli.items())),'api':dict(sorted(methods.items())),'python':functions,'tools':tools,'native':native,'environment_identifiers':env,'note':'Static reachability only; not runtime execution or numerical validation. Environment index includes compile macros. Tool keys are source-relative paths.'}


def verify_manifest(root):
    root=Path(root);manifest=root/'SOURCE_MANIFEST.json'
    if not manifest.is_file():return {'status':'UNAVAILABLE','reason':'This release has no SOURCE_MANIFEST.json; use the archive SHA-256 and catalogue identity'}
    data=json.loads(manifest.read_text());bad=[]
    for e in data['entries']:
        file=(root/e['path']).resolve()
        if not file.is_relative_to(root.resolve()) or not file.is_file() or file.stat().st_size!=e['bytes'] or sha256(file)!=e['sha256']:bad.append(e['path'])
    return {'status':'PASS' if not bad else 'FAIL','checked':len(data['entries']),'mismatches':bad}
