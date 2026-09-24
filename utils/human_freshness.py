"""Input signatures for human retrieval predictions; ignores dataset cache files."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def payload_signature(path, dataset=False):
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    files = (sorted(path.glob('data-*.arrow')) + [path / 'state.json', path / 'dataset_info.json']
             if dataset else sorted(p for p in path.rglob('*') if p.is_file()))
    if not files:
        raise ValueError(f'No payload files in {path}')
    return [[str(p.relative_to(path)), p.stat().st_size, p.stat().st_mtime_ns] for p in files]


def evaluation_signature(args):
    code = ['test.py', 'train.py', 'utils/test_inference.py', 'utils/human_freshness.py']
    inputs = {
        'dataset': str(Path(args['dataset']).absolute()),
        'dataset_payload': payload_signature(args['dataset'], dataset=True),
        'model': str(Path(args['model_path']).absolute()),
        'model_payload': payload_signature(args['model_path']),
        'distractor_dataset': str(Path(args['distractor_dataset']).absolute()),
        'distractor_payload': payload_signature(args['distractor_dataset'], dataset=True),
        'distractor_query_kind': args['distractor_query_kind'],
        'split_seed': args['split_seed'], 'top_k': args['top_k'],
        'modality': args['modality'],
        'code': {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in code},
    }
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def prediction_problem(run_dir, expected_args=None):
    directory = Path(run_dir) / 'preds_human'
    if not all((directory / name).is_file() for name in ['meta.json', 'queries.jsonl', 'corpus.jsonl']):
        return 'missing human predictions'
    meta = json.loads((directory / 'meta.json').read_text())
    if 'human_refresh_signature' not in meta:
        return 'human predictions have no matched-data signature'
    args = meta['args'] if expected_args is None else expected_args
    if not str(args['dataset']).endswith(('_human-matched', '_human-matched-in-context')):
        return 'human predictions do not use a matched dataset'
    if meta['human_refresh_signature'] != evaluation_signature(args):
        return 'human data, retrieval corpus, model, or inference code changed'
    return ''
