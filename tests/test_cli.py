import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from voice_deff.__main__ import ROOT, main


class ExperimentCliTests(unittest.TestCase):
    def invoke(self, arguments):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(arguments)
        return code, json.loads(output.getvalue())

    def test_both_datasets_and_modes_are_configurable_without_ml_imports(self):
        for dataset in ['libritts', 'vctk']:
            for mode in ['full', 'factorial']:
                with self.subTest(dataset=dataset, mode=mode):
                    code, plan = self.invoke(['--dataset', dataset, '--mode', mode,
                                              '--dataset-glob', 'missing-dataset/**/*.wav', '--dry-run'])
                    self.assertEqual(code, 0)
                    self.assertEqual(plan['new_metric_rows'], 5760 if mode == 'full' else 960)
                    self.assertEqual(plan['matching_audio_files'], 0)
                    self.assertFalse(Path(plan['output']).exists())

    def test_rejects_mixing_fixed_threshold_results_with_factorial(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(['--dataset', 'libritts', '--mode', 'factorial', '--dataset-glob', '*.wav',
                  '--old-results', str(ROOT / 'results/libritts/fixed_threshold'), '--dry-run'])
        self.assertEqual(caught.exception.code, 2)

    def test_rejects_results_from_another_dataset(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(['--dataset', 'vctk', '--mode', 'factorial', '--dataset-glob', '*.wav',
                  '--old-results', str(ROOT / 'results/libritts/calibrated_asv'), '--dry-run'])
        self.assertEqual(caught.exception.code, 2)

    def test_rejects_nonempty_output_and_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            existing = Path(directory) / 'existing.csv'
            existing.write_text('preserve me', encoding='utf-8')
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(['--dataset', 'vctk', '--dataset-glob', '*.wav', '--output', directory, '--dry-run'])
            self.assertEqual(existing.read_text(encoding='utf-8'), 'preserve me')

    def test_missing_audio_is_rejected_before_model_loading(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
            main(['--dataset', 'vctk', '--dataset-glob', 'missing-dataset/**/*.wav'])
        self.assertEqual(caught.exception.code, 2)

    def test_custom_output_and_optional_wer(self):
        with tempfile.TemporaryDirectory() as directory:
            _, plan = self.invoke(['--dataset', 'libritts', '--dataset-glob', 'missing/**/*.wav',
                                   '--output', directory, '--skip-wer', '--dry-run'])
            self.assertEqual(Path(plan['output']), Path(directory).resolve())
            self.assertFalse(plan['run_asr_wer'])


if __name__ == '__main__':
    unittest.main()
