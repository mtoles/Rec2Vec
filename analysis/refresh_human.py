"""Pull all human workbooks, build matched sets, and refresh human retrieval scores.

Called at the start of analysis.ipynb. Every call fetches all four spreadsheets.
Predictions are reused only when their matched data, model, corpus, and inference
code signatures agree. Evaluation failures stop the notebook before it plots.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile
from string import Template

import pandas as pd
from datasets import load_from_disk

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'human_study'))
import download_human_labels as original
from combine_human_labels import STUDIES, combine
from sample_human_labels import sample_all
from attr_quota import written_counts
from utils.human_freshness import evaluation_signature, prediction_problem
from utils.paper_analysis import discover_profile_runs, human_conditions, active_dataset_bases
from utils.text_holdout import HUMAN_SOURCE, verify_text_holdout
from utils.training_profile import training_profile
from utils.image_split import active_image_base, garment_id, query_key


def pull_studies(directory):
    directory.mkdir(parents=True, exist_ok=True)
    downloads = []
    for name, config in STUDIES.items():
        for study, sheet in [('first', config['first_sheet']), ('backfill', config['sheet'])]:
            url = original.EXPORT_URL.format(sheet_id=sheet)
            target = directory / f'{sheet}.xlsx'
            for attempt in range(3):
                request_url = url if attempt == 0 else f'{url}&refresh={directory.name}-{attempt}'
                subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error',
                                '--retry', '3', '--retry-all-errors', '--max-time', '180',
                                '--output', str(target), request_url], check=True)
                if zipfile.is_zipfile(target):
                    break
                target.rename(directory / f'{sheet}.incomplete-{attempt}.xlsx')
                print(f'Retrying incomplete workbook: {name}/{study}', flush=True)
            else:
                raise ValueError(f'Export returned an incomplete workbook: {name}/{study}')
            payload = target.read_bytes()
            downloads.append({'modality': name, 'study': study, 'sheet': sheet,
                              'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()})
            print(f'Fetched {name}/{study}: {len(payload):,} bytes', flush=True)
    (directory / 'downloads.json').write_text(json.dumps(downloads, indent=2) + '\n')
    for name, config in STUDIES.items():
        tabs = pd.read_excel(directory / (config['first_sheet'] + '.xlsx'), sheet_name=None)
        original.build(name, config['first_sheet'], config['headers'],
                       ROOT / 'dataset/processed' / config['base'],
                       original.text_pool if name == 'text' else original.image_pool,
                       config['positive'], config['negative'], '_human', 5, tabs=tabs)
        combine(name, config, directory)
    return downloads


def verify_human_image_holdout():
    synthetic = load_from_disk(str(active_image_base()) + '_rephrased-in-context')
    human = load_from_disk(str(ROOT / 'dataset/processed' / (STUDIES['image']['base'] + '_human-matched-in-context')))
    non_test = synthetic.filter(lambda row: row['split'] != 'test')
    garments = {garment_id(value) for column in ['positive_product_id', 'negative_product_id']
                for value in non_test[column]}
    queries = {query_key(value) for column in ['nl_query', 'rephrased_query'] for value in non_test[column]}
    overlap = {garment_id(value) for value in human['positive_product_id']} & garments
    if overlap:
        raise ValueError(f'Human image targets overlap training/validation garments: {sorted(overlap)}')
    if {query_key(value) for value in human['human_query']} & queries:
        raise ValueError('Human image queries overlap training/validation queries')


def human_runs():
    runs = discover_profile_runs()
    runs = runs[(runs.query_kind == 'rephrased') & (runs.rephrase == 'in-context')].copy()
    runs['easy'] = runs.easy.fillna(20).astype(int)
    conditions = human_conditions()
    conditions['V'] = pd.to_numeric(conditions['V'], errors='coerce')
    conditions['easy'] = [next((int(t[5:]) for t in extra.split(',') if t.startswith('easy=')), 20)
                          for extra in conditions['extra']]
    keys = ['modality', 'style', 'query_kind', 'V', 'easy', 'negs', 'mining', 'seed', 'rephrase', 'order']
    matched = conditions.drop(columns='extra').drop_duplicates().merge(runs, on=keys, how='left', validate='one_to_one')
    if matched.run_dir.isna().any():
        raise ValueError('Some declared human models are missing')
    return matched


def evaluation_jobs():
    jobs = []
    for run in human_runs().itertuples():
        config = STUDIES['text' if run.modality == 'text' else 'image']
        synthetic = json.loads((Path(run.run_dir) / 'preds/meta.json').read_text())
        args = {
            'modality': run.modality, 'model_path': str(Path(run.run_dir) / 'final'),
            'dataset': str(ROOT / 'dataset/processed' / (config['base'] + '_human-matched-in-context')),
            'query_kind': 'human', 'run_dir': run.run_dir, 'split': 'test', 'top_k': 100,
            'split_seed': synthetic['args']['split_seed'],
            'distractor_dataset': str((ROOT / synthetic['args']['dataset']).absolute()),
            'distractor_query_kind': synthetic['args']['query_kind'],
        }
        jobs.append(args)
    return jobs


def archive(path, stamp):
    if path.exists():
        target = path.parent / 'old' / stamp / path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        path.rename(target)


def evaluate(job, device, directory):
    run = Path(job['run_dir'])
    stamp = directory.name
    stage = run / ('human_refresh_' + stamp)
    log = directory / (run.name + '.log')
    cmd = [sys.executable, '-B', '-u', str(ROOT / 'test.py')]
    for key, value in job.items():
        cmd.extend(['--' + key.replace('_', '-'), str(stage) if key == 'run_dir' else str(value)])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=device, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
               WANDB_MODE='offline')
    signature = evaluation_signature(job)
    with log.open('w') as handle:
        result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Human inference failed for {run.name}: see {log}\n{log.read_text()[-3000:]}')
    if evaluation_signature(job) != signature:
        raise RuntimeError(f'Inputs changed while evaluating {run.name}; predictions were not published')
    meta_path = stage / 'preds_human/meta.json'
    meta = json.loads(meta_path.read_text())
    meta['args']['run_dir'] = str(run)
    meta['human_refresh_signature'] = signature
    meta_path.write_text(json.dumps(meta, indent=2) + '\n')
    archive(run / 'preds_human', stamp)
    (stage / 'preds_human').rename(run / 'preds_human')
    stage.rmdir()
    problem = prediction_problem(run, job)
    if problem:
        raise RuntimeError(f'{run.name}: {problem}')
    print(f'Updated GPU {device}: {run.name}', flush=True)


def archive_human_outputs(stamp):
    names = {
        'paper/figs': ['text_vs_image_recall.png', 'text_vs_image_recall.pdf',
                       'main_results_table.tex', 'significance_table.tex'],
        'analysis/figs': ['protocol_infonce.png', 'protocol_mse.png', 'protocol_cosent.png',
                          'protocol_siglip.png', 'significance.csv', 'all_results.csv',
                          'text_vs_image_recall.csv', 'mining_baseline.png'],
    }
    for directory, artifacts in names.items():
        for name in artifacts:
            archive(ROOT / directory / name, stamp)


def refresh_predictions(directory, devices=None):
    jobs = evaluation_jobs()
    pending = [job for job in jobs if prediction_problem(job['run_dir'], job)]
    print(f'Human predictions: {len(jobs)-len(pending)} current, {len(pending)} need evaluation', flush=True)
    if pending:
        archive_human_outputs(directory.name)
        if devices is None:
            import torch
            devices = (os.environ['CUDA_VISIBLE_DEVICES'].split(',') if 'CUDA_VISIBLE_DEVICES' in os.environ
                       else [str(i) for i in range(torch.cuda.device_count())])
            if not devices:
                devices = ['']
        groups = [pending[i::len(devices)] for i in range(len(devices))]
        def worker(device, group):
            for job in group:
                evaluate(job, device, directory)
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = [pool.submit(worker, device, group) for device, group in zip(devices, groups)]
            for future in futures:
                future.result()
    for job in jobs:
        problem = prediction_problem(job['run_dir'], job)
        if problem:
            raise RuntimeError(f'{job["run_dir"]}: {problem}')
    return {'expected': len(jobs), 'evaluated': len(pending), 'reused': len(jobs)-len(pending)}


def write_table(stamp):
    summary, replacements = {}, {}
    for name, config in STUDIES.items():
        path = ROOT / 'dataset/processed' / (config['base'] + '_human-matched-in-context')
        rows = load_from_disk(str(path))
        documents = set(rows['positive_example'])
        summary[name] = {
            'dataset': str(path), 'queries': len(set(rows['human_query'])),
            'documents': len(documents),
            'words_per_document': (sum(len(text.split()) for text in documents) / len(documents)
                                   if name == 'text' else None),
            'words_per_query': sum(len(text.split()) for text in rows['human_query']) / len(rows),
            'attributes_per_query': sum(sum(written_counts(row)) for row in rows) / len(rows),
        }
        replacements[name + '_documents'] = str(len(documents))
        for field in ['words_per_document', 'words_per_query', 'attributes_per_query']:
            value = summary[name][field]
            replacements[name + '_' + field] = '---' if value is None else f'{value:.1f}'
    synthetic = load_from_disk(str(active_image_base()) + '_rephrased-in-context')
    rows = list(synthetic.select_columns(['split', 'nl_query', 'rephrased_query', 'positive_example', 'negative_example',
                                         'negative_example_source', 'query_distance',
                                         'selected_pos_features', 'selected_neg_features',
                                         'selected_common_features', 'selected_neither_features']))
    corpus = {r[c] for r in rows for c in ['positive_example', 'negative_example']}
    replacements['image_corpus_documents'] = format(len(corpus), ',').replace(',', '{,}')
    for side, label in [('train', 'training'), ('validation', 'validation'), ('test', 'test')]:
        docs = {r[c] for r in rows if r['split'] == side for c in ['positive_example', 'negative_example']}
        replacements['image_' + label + '_documents'] = format(len(docs), ',').replace(',', '{,}')
    queries = {r['rephrased_query'] for r in rows}
    replacements['image_synthetic_words'] = f"{sum(len(q.split()) for q in queries)/len(queries):.1f}"
    hard = [r for r in rows if r['negative_example_source'] != 'random']
    attributes = ['selected_pos_features', 'selected_neg_features', 'selected_common_features', 'selected_neither_features']
    replacements['image_synthetic_attributes'] = f"{sum(sum(len(r[c]) for c in attributes) for r in hard)/len(hard):.1f}"
    replacements['image_synthetic_differentiating'] = f"{sum(r['query_distance'] for r in hard)/len(hard):.1f}"
    content = Template((ROOT / 'paper/dataset_summary_template.tex').read_text()).substitute(replacements)
    target = ROOT / 'paper/figs/dataset_summary_table.tex'
    if not target.exists() or target.read_text() != content:
        archive(target, stamp)
        target.write_text(content)
    (ROOT / 'analysis/human_matched_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpus', help='Comma-separated GPU identifiers; defaults to all visible GPUs')
    parser.add_argument('--data-only', action='store_true', help='Pull and rebuild datasets without retrieval evaluation')
    args = parser.parse_args()
    os.chdir(ROOT)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    directory = ROOT / 'logs/human_refresh' / stamp
    downloads = pull_studies(directory)
    sampling = sample_all()
    verify_human_image_holdout()
    verify_text_holdout(
        load_from_disk(str(ROOT / "dataset/processed" / (active_dataset_bases()["text"] + "_rephrased-in-context"))),
        load_from_disk(str(ROOT / HUMAN_SOURCE)))
    summary = write_table(stamp)
    status = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'downloads': downloads,
              'sampling': sampling, 'summary': summary, 'log_directory': str(directory)}
    if not args.data_only:
        status['predictions'] = refresh_predictions(directory, None if args.gpus is None else args.gpus.split(','))
    (directory / 'refresh.json').write_text(json.dumps(status, indent=2) + '\n')
    if not args.data_only:
        (ROOT / 'analysis/human_refresh_status.json').write_text(json.dumps(status, indent=2) + '\n')
    print('Human refresh complete: ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    lock_path = ROOT / 'logs/human_refresh/refresh.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
