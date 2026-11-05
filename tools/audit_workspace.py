"""Read-only checks of preserved artifacts, result pairing, and extracted code."""
import ast
import collections
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def records(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def definitions(source):
    tree = ast.parse(source)
    return {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))}


def main():
    migration = json.loads((ROOT / 'reports/migration_map.json').read_text())
    for entry in migration:
        path = ROOT / entry['current_path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], str(path)
    duplicates = json.loads((ROOT / 'reports/duplicate_files.json').read_text())
    for entry in duplicates:
        assert (ROOT / entry['canonical']).samefile(ROOT / entry['alias']), entry

    original = (ROOT / 'archive/original_sources/libritts-terbaru.py').read_text(encoding='utf-8')
    original = '\n'.join(line for line in original.splitlines() if not line.startswith('%'))
    old_defs = definitions(original)
    new_defs = {}
    for filename in ['common.py', 'full.py']:
        new_defs.update(definitions((ROOT / 'voice_deff' / filename).read_text(encoding='utf-8')))
    for name, node in old_defs.items():
        assert ast.dump(node, include_attributes=False) == ast.dump(new_defs[name], include_attributes=False), name
    notebook = json.loads((ROOT / 'archive/original_sources/static_ushape_factorial_ablation_kaggle_Libri.ipynb').read_text(encoding='utf-8'))
    notebook_defs = definitions('\n'.join(''.join(c['source']) for c in notebook['cells'][1:6]))
    factorial_defs = definitions((ROOT / 'voice_deff/factorial.py').read_text(encoding='utf-8'))
    common_defs = definitions((ROOT / 'voice_deff/common.py').read_text(encoding='utf-8'))
    for name, node in notebook_defs.items():
        assert ast.dump(node, include_attributes=False) == ast.dump((factorial_defs | common_defs)[name], include_attributes=False), name
    for path in (ROOT / 'voice_deff').glob('*.py'):
        compile(path.read_text(encoding='utf-8'), str(path), 'exec')

    details = []
    for dataset in ['libritts', 'vctk']:
        versions = {}
        for version in ['calibrated_asv', 'fixed_threshold']:
            directory = ROOT / 'results' / dataset / version
            rows = records(directory / 'per_sample_metrics.csv')
            selection = records(directory / 'sample_selection_manifest.csv')
            keys = {(r['sample_id'], r['run_seed'], r['method']) for r in rows}
            assert len(rows) == len(keys) == 5760
            counts = collections.Counter(row['method'] for row in rows)
            assert len(counts) == 6 and set(counts.values()) == {960}
            expected_pairs = None
            for method in counts:
                pairs = {(r['sample_id'], r['run_seed']) for r in rows if r['method'] == method}
                expected_pairs = expected_pairs or pairs
                assert pairs == expected_pairs
            assert len(selection) == 500 and len({r['speaker_id'] for r in selection}) == 60
            cal = {r['speaker_id'] for r in selection if r['split'] == 'calibration'}
            test = {r['speaker_id'] for r in selection if r['split'] == 'test'}
            assert not cal & test
            summary = records(directory / 'summary_statistics.csv')
            selected_metrics = {'s_asv', 'pesq', 'stoi', 'privacy_utility_success', 'dsr_success'}
            details.extend({'dataset': dataset, 'version': version, 'method': r['method'], 'metric': r['metric'], 'mean': r['mean']}
                           for r in summary if r['metric'] in selected_metrics)
            versions[version] = rows
        # Postprocessing can change threshold-derived columns, but compare the raw endpoints.
        lookup = lambda rows: {(r['sample_id'], r['run_seed'], r['method']): r for r in rows}
        calibrated, fixed = lookup(versions['calibrated_asv']), lookup(versions['fixed_threshold'])
        assert calibrated.keys() == fixed.keys()
        changed = {}
        for metric in ['s_asv', 'pesq', 'stoi', 'ssim', 'squim_mos', 'asv_obfuscated', 'dsr_success', 'privacy_utility_success']:
            count = 0
            for key in calibrated:
                a, b = float(calibrated[key][metric]), float(fixed[key][metric])
                count += abs(a - b) > 1e-10
            changed[metric] = count
        print(dataset, 'changed result cells:', json.dumps(changed))
    with (ROOT / 'reports/results_comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=details[0].keys())
        writer.writeheader(); writer.writerows(details)
    result = {'status': 'PASS', 'original_files_preserved': len(migration), 'hardlink_aliases_verified': len(duplicates),
              'full_definitions_equivalent': len(old_defs), 'factorial_definitions_equivalent': len(notebook_defs),
              'result_folders_validated': 4, 'metric_rows_validated': 23040,
              'checks': ['SHA-256 preservation', 'hardlink identity', 'AST equivalence', 'Python syntax',
                         'unique sample-seed-method keys', 'paired probes across methods', 'speaker-disjoint split'],
              'not_run': ['GPU experiments', 'model downloads', 'audio metric recomputation']}
    (ROOT / 'reports/validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
