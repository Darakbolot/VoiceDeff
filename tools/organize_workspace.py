"""One-time migration of the supplied experiment bundle, with SHA-256 provenance."""
import ast
import collections
import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def clean(source):
    return '\n'.join(line for line in source.splitlines() if not line.startswith('%')) + '\n'


def main():
    if (ROOT / 'reports/original_inventory.csv').exists():
        raise SystemExit('Migration already completed; refusing to run twice.')
    reports = ROOT / 'reports'
    reports.mkdir(exist_ok=True)
    original = sorted(p for p in ROOT.rglob('*') if p.is_file() and p.parts[len(ROOT.parts)] not in {'tools', 'reports'})
    inventory = []
    for path in original:
        inventory.append({'original_path': path.relative_to(ROOT).as_posix(), 'bytes': path.stat().st_size,
                          'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    with (reports / 'original_inventory.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=inventory[0].keys())
        writer.writeheader(); writer.writerows(inventory)

    full_source = clean((ROOT / 'libritts-terbaru.py').read_text(encoding='utf-8'))
    notebook = json.loads((ROOT / 'static_ushape_factorial_ablation_kaggle_Libri.ipynb').read_text(encoding='utf-8'))
    full_tree = ast.parse(full_source)
    factorial_source = '\n'.join(clean(''.join(c['source'])) for c in notebook['cells'][1:6])
    factorial_tree = ast.parse(factorial_source)
    definitions = (ast.FunctionDef, ast.ClassDef)
    full_defs = {node.name: node for node in full_tree.body if isinstance(node, definitions)}
    factorial_defs = {node.name: node for node in factorial_tree.body if isinstance(node, definitions)}
    shared = {name for name, node in factorial_defs.items() if name in full_defs
              and ast.dump(node, include_attributes=False) == ast.dump(full_defs[name], include_attributes=False)}
    # Imports and setup are copied once. Definitions are compared structurally,
    # so only functions with identical behavior enter the common module.
    first_def_line = min(node.lineno for node in full_defs.values())
    prefix = '\n'.join(full_source.splitlines()[:first_def_line - 2]) + '\n'
    package = ROOT / 'voice_deff'
    package.mkdir(exist_ok=True)
    (package / '__init__.py').write_text('"""Voice protection experiments: full evaluation and factorial extension."""\n', encoding='utf-8')
    common = prefix + '\n\n' + '\n\n'.join(ast.get_source_segment(full_source, node) if not isinstance(node, ast.ClassDef) or not node.decorator_list
        else '\n'.join(full_source.splitlines()[node.decorator_list[0].lineno-1:node.end_lineno])
        for node in full_tree.body if isinstance(node, definitions) and node.name in shared)
    # AST source segments omit decorators; ExperimentConfig must remain a dataclass.
    (package / 'common.py').write_text(common + '\n', encoding='utf-8')
    imports = 'from __future__ import annotations\nfrom .common import *\n\n'
    methods_node = next(node for node in full_tree.body if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'METHODS' for t in node.targets))
    full = imports + ast.get_source_segment(full_source, methods_node) + '\n\n'
    full += '\n\n'.join(ast.get_source_segment(full_source, node) for node in full_tree.body if isinstance(node, definitions) and node.name not in shared)
    (package / 'full.py').write_text(full + '\n', encoding='utf-8')
    factorial = imports + '\n\n'.join(ast.get_source_segment(factorial_source, node) for node in factorial_tree.body if isinstance(node, definitions) and node.name not in shared)
    setup = clean(''.join(notebook['cells'][6]['source']))
    setup = setup.replace('TARGET_DATASET = "LibriTTS"  # ganti menjadi "VCTK" untuk run kedua', 'TARGET_DATASET = dataset')
    setup = setup.replace('OLD_RESULT_SOURCE = "./libri"', 'OLD_RESULT_SOURCE = str(old_results_path)')
    setup = setup.replace('RUN_ASR_WER = True', 'RUN_ASR_WER = run_asr_wer')
    setup = setup.replace('DATASET_GLOBS[TARGET_DATASET],', 'dataset_glob,')
    setup = setup.replace('f"/kaggle/working/factorial_ablation_{TARGET_DATASET.lower()}",', 'str(output_path),')
    # Reject the incompatible fixed-threshold protocol before loading models.
    marker = 'old_manifest = _read_manifest(old_result_dir)'
    setup = setup.replace(marker, marker + '\nif "primary_asv_evaluation_threshold" not in old_manifest["calibration"]:\n    raise ValueError("Mode factorial membutuhkan hasil calibrated_asv, bukan fixed_threshold.")')
    tail = clean(''.join(notebook['cells'][9]['source']))
    tail = tail.replace('archive_base = f"/kaggle/working/static_ushape_factorial_{TARGET_DATASET.lower()}"', 'archive_base = str(output_dir.parent / (output_dir.name + "_archive"))')
    body = '\n\n'.join([setup, clean(''.join(notebook['cells'][7]['source'])), clean(''.join(notebook['cells'][8]['source'])), tail])
    # Nested statistic helpers retain access to the selected dataset and thresholds.
    factorial += '\n\ndef run(dataset, dataset_glob, old_results_path, output_path, run_asr_wer=True):\n'
    factorial += '\n'.join('    ' + line if line else '' for line in body.splitlines()) + '\n'
    (package / 'factorial.py').write_text(factorial, encoding='utf-8')
    mapping = {'libri': 'results/libritts/calibrated_asv', 'vctk': 'results/vctk/calibrated_asv',
               'hasil_revisi_libritts': 'results/libritts/fixed_threshold', 'hasil_revisi_vctk': 'results/vctk/fixed_threshold'}
    for source, target in mapping.items():
        dest = ROOT / target
        dest.parent.mkdir(parents=True, exist_ok=True)
        (ROOT / source).rename(dest)
    archive = ROOT / 'archive/original_sources'
    archive.mkdir(parents=True, exist_ok=True)
    for path in original:
        if path.parent == ROOT:
            path.rename(archive / path.name)
    # Keep semantic filenames usable by notebooks and analysis tools, but retain
    # only one physical copy for each identical payload using NTFS hard links.
    groups = collections.defaultdict(list)
    for path in sorted((ROOT / 'results').rglob('*')):
        if path.is_file():
            groups[hashlib.sha256(path.read_bytes()).hexdigest()].append(path)
    duplicates = []
    for sha, paths in groups.items():
        canonical = min(paths, key=lambda p: ('.ipynb_checkpoints' in p.parts, 'checkpoint' in p.name, 'preview' in p.name, p.as_posix()))
        for path in paths:
            if path == canonical:
                continue
            # Create the link first; an unsupported filesystem leaves the original intact.
            temporary = path.with_name(path.name + '.dedup-link')
            os.link(canonical, temporary)
            os.replace(temporary, path)
            duplicates.append({'canonical': canonical.relative_to(ROOT).as_posix(), 'alias': path.relative_to(ROOT).as_posix(), 'sha256': sha})
    (reports / 'duplicate_files.json').write_text(json.dumps(duplicates, indent=2), encoding='utf-8')
    for item in inventory:
        parts = item['original_path'].split('/')
        item['current_path'] = (mapping[parts[0]] + '/' + '/'.join(parts[1:])) if len(parts) > 1 else 'archive/original_sources/' + parts[0]
        assert hashlib.sha256((ROOT / item['current_path']).read_bytes()).hexdigest() == item['sha256']
    (reports / 'migration_map.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
    (reports / 'code_consolidation.json').write_text(json.dumps({'shared_definitions': sorted(shared), 'protocol_specific': sorted(set(factorial_defs) - shared), 'original_files': len(inventory), 'unique_original_contents': len({r['sha256'] for r in inventory}), 'hardlink_aliases': len(duplicates)}, indent=2), encoding='utf-8')
    print(f'Preserved {len(inventory)} files; shared {len(shared)} definitions; deduplicated {len(duplicates)} physical copies.')


if __name__ == '__main__':
    main()
