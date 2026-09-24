"""Regression checks for refreshing annotations without accepting stale scores."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

from datasets import Dataset, load_from_disk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'human_study'))
from sample_human_labels import capped_indices, histogram
from dataset_io import save_version
from combine_human_labels import hold_out_examples
from utils.human_freshness import evaluation_signature, prediction_problem


class HumanRefreshTests(unittest.TestCase):
    def test_unchanged_annotations_preserve_payload_and_changes_are_archived(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'human'
            rows = Dataset.from_dict({'human_query': ['old answer']})
            self.assertTrue(save_version(rows, path))
            before = (path / 'state.json').stat().st_mtime_ns
            self.assertFalse(save_version(rows, path))
            self.assertEqual(before, (path / 'state.json').stat().st_mtime_ns)
            self.assertTrue(save_version(Dataset.from_dict({'human_query': ['new answer']}), path))
            archive = next((path.parent / 'old').iterdir())
            self.assertEqual(load_from_disk(str(archive))['human_query'], ['old answer'])
            self.assertEqual(load_from_disk(str(path))['human_query'], ['new answer'])

    def test_sampling_caps_low_bins_and_preserves_all_six_plus_rows(self):
        rows = [{'human_pos_attributes': ','.join(['attribute'] * count),
                 'human_neg_attributes': ''}
                for count, frequency in [(0, 7), (2, 9), (5, 2), (6, 4), (8, 3), (12, 1)]
                for _ in range(frequency)]
        selected, cap = capped_indices(rows, seed=42)
        self.assertEqual(cap, 4)
        self.assertEqual((selected, cap), capped_indices(rows, seed=42))
        self.assertEqual(histogram([rows[i] for i in selected]),
                         {'0': 4, '2': 4, '5': 2, '6': 4, '8': 3, '12': 1})
        self.assertTrue(set(range(18, 26)).issubset(selected))
        self.assertEqual(len(selected), len(set(selected)))

    def test_fixed_example_product_is_excluded_after_answer_edit(self):
        rows = Dataset.from_dict({'positive_id': ['style-product', 'new-product'],
                                  'human_query': ['edited answer', 'new answer']})
        examples = [{'positive_id': 'style-product', 'human_query': 'training-time answer'}]
        kept = hold_out_examples(rows, examples, 'positive_id')
        self.assertEqual(kept['positive_id'], ['new-product'])

    def test_prediction_signature_tracks_data_model_and_corpus_but_not_cache(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            dataset = root / 'data_human-matched-in-context'
            corpus = root / 'corpus'
            model = root / 'model'
            run = root / 'run'
            for path in [dataset, corpus]:
                path.mkdir()
                for name in ['state.json', 'dataset_info.json', 'data-00000-of-00001.arrow']:
                    (path / name).write_text('initial payload')
            model.mkdir()
            (model / 'model.safetensors').write_text('initial weights')
            args = {'dataset': str(dataset), 'model_path': str(model),
                    'distractor_dataset': str(corpus), 'distractor_query_kind': 'rephrased',
                    'split_seed': 42, 'top_k': 100, 'modality': 'text'}
            signature = evaluation_signature(args)
            (dataset / 'cache-map.arrow').write_text('cache')
            self.assertEqual(signature, evaluation_signature(args))
            pred = run / 'preds_human'
            pred.mkdir(parents=True)
            for name in ['queries.jsonl', 'corpus.jsonl']:
                (pred / name).write_text('')
            (pred / 'meta.json').write_text(json.dumps({'args': args, 'human_refresh_signature': signature}))
            self.assertEqual(prediction_problem(run, args), '')
            for changed in [dataset / 'data-00000-of-00001.arrow',
                            corpus / 'data-00000-of-00001.arrow', model / 'model.safetensors']:
                previous = evaluation_signature(args)
                changed.write_text('new contents with a different length')
                self.assertNotEqual(previous, evaluation_signature(args))
                self.assertNotEqual(prediction_problem(run, args), '')


if __name__ == '__main__':
    unittest.main()
