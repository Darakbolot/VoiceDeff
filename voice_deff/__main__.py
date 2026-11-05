"""Dataset-independent entry point; --dry-run needs only Python's standard library."""
import argparse
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATASETS = {'libritts': 'LibriTTS', 'vctk': 'VCTK'}


def main(argv=None):
    parser = argparse.ArgumentParser(description='Evaluasi voice protection: enam metode utama atau tambahan Static PGD + U-Shape.')
    parser.add_argument('--dataset', required=True, choices=DATASETS)
    parser.add_argument('--mode', choices=['full', 'factorial'], default='full')
    parser.add_argument('--dataset-glob', required=True, help='Pola WAV rekursif; bungkus dengan tanda kutip.')
    parser.add_argument('--output', type=Path, help='Folder baru untuk hasil eksperimen.')
    parser.add_argument('--old-results', type=Path, help='Hasil calibrated_asv untuk mode factorial.')
    parser.add_argument('--skip-wer', action='store_true', help='Lewati ASR/WER pada hasil baru.')
    parser.add_argument('--dry-run', action='store_true', help='Periksa rencana tanpa memuat model atau menjalankan PGD.')
    args = parser.parse_args(argv)
    output = (args.output or ROOT / 'runs' / args.dataset / args.mode).resolve()
    if output.exists() and any(output.iterdir()):
        parser.error('Folder output harus baru atau kosong agar hasil sebelumnya tidak tertimpa.')
    old_results = (args.old_results or ROOT / 'results' / args.dataset / 'calibrated_asv').resolve()
    if args.mode == 'factorial':
        if not old_results.is_dir():
            parser.error('--old-results harus berupa folder hasil calibrated_asv.')
        try:
            manifest = json.loads((old_results / 'experiment_manifest.json').read_text(encoding='utf-8'))
        except (OSError, ValueError) as error:
            parser.error(f'Manifest hasil lama tidak dapat dibaca: {error}')
        if manifest['config']['dataset_name'] != DATASETS[args.dataset]:
            parser.error('Dataset pada hasil lama tidak cocok dengan --dataset.')
        if 'primary_asv_evaluation_threshold' not in manifest.get('calibration', {}):
            parser.error('Mode factorial membutuhkan hasil calibrated_asv, bukan fixed_threshold.')
    matched = sum(1 for _ in glob.iglob(args.dataset_glob, recursive=True))
    plan = {'dataset': DATASETS[args.dataset], 'mode': args.mode, 'dataset_glob': args.dataset_glob,
            'matching_audio_files': matched, 'output': str(output),
            'old_results': str(old_results) if args.mode == 'factorial' else None,
            'unique_utterances': 500, 'probe_utterances': 320, 'seeds': [42, 43, 44],
            'iterations_per_pgd_method': 1000, 'new_metric_rows': 5760 if args.mode == 'full' else 960,
            'run_asr_wer': not args.skip_wer}
    print(json.dumps(plan, indent=2))
    if args.dry_run:
        return 0
    if matched == 0:
        parser.error('Tidak ada WAV yang cocok dengan --dataset-glob.')
    if args.mode == 'factorial':
        from .factorial import run
        run(DATASETS[args.dataset], args.dataset_glob, old_results, output, not args.skip_wer)
    else:
        from .common import ExperimentConfig
        from .full import METHODS, run_experiment
        cfg = ExperimentConfig(dataset_name=DATASETS[args.dataset], dataset_glob=args.dataset_glob,
                               output_dir=str(output), checkpoint_every=1, run_asr_wer=not args.skip_wer)
        output.mkdir(parents=True, exist_ok=True)
        import csv
        with (output / 'evaluated_methods.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['method'])
            writer.writerows((method,) for method in METHODS)
        run_experiment(cfg)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
