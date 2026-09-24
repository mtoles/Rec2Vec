"""Combine the saved first human study with collected backfill annotations.

Usage: python human_study/combine_human_labels.py
Optional --workbook-dir reuses XLSX exports named <sheet_id>.xlsx.
Writes *_human-combined and *_human-combined-in-context. The original study and
its fixed rephrasing examples remain the inputs to historical model evaluations.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from datasets import concatenate_datasets, load_from_disk

import download_human_labels as original
from dataset_io import save_version

ROOT = original.REPO_ROOT
STUDIES = {
    'text': {
        'sheet': '1aT9mDmMGltYGf4avmDjO8NwIsVR7d6C4fXBEJwT-8kA',
        'first_sheet': '1DRYeAECYlWF2SGniw7vbGj873Hiz6VDEUnystnpjA8M',
        'base': 'feature-distance-dataset_gemini-2.5-flash_1000000_nolek',
        'raw': original.human_data_text.RAW_DEFAULT,
        'positive': 'positive_id', 'negative': 'negative_id',
        'headers': original.TEXT_HEADERS, 'replacements': {},
    },
    'image': {
        'sheet': '1ZOtTNWVY92cwqFhX7IY-hvS8y6eiK32-aXPVKFCfDYs',
        'first_sheet': '18Ok_AmDiqwMgU5mGXslA-XN3dUNqT_nae4M3XlDbpYw',
        'base': 'deepfashion-inshop-image-triplets_hf_20000',
        'raw': original.human_data_image.RAW_DEFAULT,
        'positive': 'positive_product_id', 'negative': 'negative_product_id',
        'headers': original.IMAGE_HEADERS, 'replacements': {'Lorena': 'Lorena fixed'},
    },
}


def corrected_tabs(tabs, replacements):
    for previous, replacement in replacements.items():
        if previous not in tabs or replacement not in tabs:
            raise ValueError(f'Missing expected correction tabs: {previous}, {replacement}')
        keys = ['Product 1', 'Product 2']
        if not tabs[previous][keys].equals(tabs[replacement][keys]):
            raise ValueError(f'Correction {replacement} no longer covers the same ordered pairs')
    return {name: frame for name, frame in tabs.items() if name not in replacements}


def pair_key(row):
    return tuple(str(row[key]).strip() for key in ('Product 1', 'Product 2'))


def raw_matches(name, raw_path, requested):
    """Resolve workbook product text against raw rows without resampling quotas."""
    first_products = {key[0] for key in requested}
    matches = {key: set() for key in requested}
    render = (original.human_data_text.format_product if name == 'text'
              else original.human_data_image.describe)
    with open(raw_path) as handle:
        for line in handle:
            row = json.loads(line)
            shopping_for = row['original_query'] if name == 'text' else row['item']
            first = render(row['positive_product'], shopping_for=shopping_for).strip()
            if first not in first_products:
                continue
            key = first, render(row['hard_neg_product']).strip()
            if key in matches:
                positive = row['positive_product']['product_id']
                negative = row['hard_neg_product']['product_id']
                matches[key].add((positive, negative, row['item']))
    missing = [key for key, candidates in matches.items() if not candidates]
    if missing:
        raise ValueError(f'{len(missing)} annotated pairs cannot be matched to {raw_path}; '
                         f'first product: {missing[0][0][:180]!r}')
    return matches


def hold_out_examples(dataset, examples, positive_column):
    excluded = {row[positive_column] for row in examples}
    if len(excluded) != len(examples):
        raise ValueError('Fixed in-context examples have repeated positive products')
    return dataset.select([i for i, row in enumerate(dataset)
                           if row[positive_column] not in excluded])


def combine(name, config, workbook_dir):
    print(f'== {name}', flush=True)
    sheet_id = config['sheet']
    if workbook_dir is None:
        tabs = original.fetch_workbook(sheet_id)
    else:
        tabs = pd.read_excel(workbook_dir / f'{sheet_id}.xlsx', sheet_name=None)
    original.check_headers(tabs, config['headers'], sheet_id)
    tabs = corrected_tabs(tabs, config['replacements'])
    kept = original.annotated_rows(tabs)
    requested = {pair_key(row) for _, row in kept}
    matches = raw_matches(name, config['raw'], requested)
    base_path = ROOT / 'dataset/processed' / config['base']
    base = load_from_disk(str(base_path))
    first = load_from_disk(str(base_path) + '_human')
    index = original.base_index(base, config['positive'], config['negative'])
    original_queries = list(base['original_query'])
    selected, extras, absent = [], [], []
    seen = set()
    for tab, row in kept:
        raw_ids = matches[pair_key(row)]
        candidates = {i for ids in raw_ids if ids in index for i in index[ids]}
        if name == 'text':
            source_query = str(row['Product 1']).splitlines()[0].removeprefix('SHOPPING FOR: ')
            candidates = {i for i in candidates if original_queries[i] == source_query}
        if not candidates:
            absent.append({'tab': tab, 'sheet_row': int(row.name) + 2, 'pairs': sorted(raw_ids)})
            continue
        if len(candidates) != 1:
            raise ValueError(f'{name}/{tab}/row {row.name + 2}: ambiguous base rows {sorted(candidates)}')
        row_index = next(iter(candidates))
        annotation = {
            'human_query': str(row['Q3']).strip(),
            'human_query_alt': '' if original.blank(row['Q4']) else str(row['Q4']).strip(),
            'human_pos_attributes': '' if original.blank(row['Q1']) else str(row['Q1']).strip(),
            'human_neg_attributes': '' if original.blank(row['Q2']) else str(row['Q2']).strip(),
            'annotator': tab,
            'source_study': 'backfill', 'source_sheet_id': sheet_id,
            'source_tab': tab, 'source_row': int(row.name) + 2,
        }
        key = row_index, annotation['human_query']
        if key in seen:
            raise ValueError(f'Duplicate annotation at {name}/{tab}/row {row.name + 2}')
        seen.add(key)
        selected.append(row_index)
        extras.append(annotation)
    if not selected:
        raise ValueError(f'No usable new {name} annotations matched the base dataset')
    added = base.select(selected)
    for column in extras[0]:
        added = added.add_column(column, [row[column] for row in extras])
    if 'split' in added.column_names:
        added = added.remove_columns('split')
    added = added.add_column('split', ['test'] * len(added))
    for column, values in {
        'source_study': ['first'] * len(first),
        'source_sheet_id': [config['first_sheet']] * len(first),
        'source_tab': list(first['annotator']), 'source_row': [-1] * len(first),
    }.items():
        first = first.add_column(column, values)
    added = added.select_columns(first.column_names).cast(first.features)
    combined = concatenate_datasets([first, added])
    positive = config['positive']
    example_path = ROOT / 'human_study' / f'in_context_examples_{name}.json'
    examples = json.loads(example_path.read_text())
    evaluation = hold_out_examples(combined, examples, positive)
    report = {
        'modality': name, 'first_study_rows': len(first), 'new_usable_sheet_rows': len(kept),
        'new_matched_rows': len(added), 'combined_rows': len(combined),
        'in_context_rows': len(evaluation), 'held_out_examples': len(combined) - len(evaluation),
        'unique_positive_products': len(set(combined[positive])),
        'overlap_products_between_studies': len(set(first[positive]) & set(added[positive])),
        'unmatched_base_rows': absent, 'superseded_tabs': config['replacements'],
        'source_sheet_url': f'https://docs.google.com/spreadsheets/d/{sheet_id}/edit',
        'first_study_dataset': str(base_path) + '_human',
        'in_context_examples': str(example_path),
        'updated_utc': datetime.now(timezone.utc).isoformat(),
    }
    print(json.dumps(report, indent=2), flush=True)
    for suffix, dataset in [('_human-combined', combined), ('_human-combined-in-context', evaluation)]:
        save_version(dataset, Path(str(base_path) + suffix), report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workbook-dir', type=Path)
    args = parser.parse_args()
    for name, config in STUDIES.items():
        combine(name, config, args.workbook_dir)


if __name__ == '__main__':
    main()
