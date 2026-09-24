"""Cap each human attribute-count bin at the largest bin with 6+ attributes.

Usage: python human_study/sample_human_labels.py [--seed 42]
Writes *_human-matched and *_human-matched-in-context.
Sampling is uniform within each overfull total-attribute bin, separately by
modality. Counts use Q1/Q2 through attr_quota.written_counts. Original combined
rows remain available; fixed style examples are excluded after sampling.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_from_disk

from attr_quota import written_counts
from combine_human_labels import ROOT, STUDIES, hold_out_examples, save_version


def capped_indices(rows, seed=42, reference_min=6):
    bins = defaultdict(list)
    for i, row in enumerate(rows):
        bins[sum(written_counts(row))].append(i)
    reference_sizes = [len(indices) for count, indices in bins.items() if count >= reference_min]
    if not reference_sizes:
        raise ValueError(f'No queries have {reference_min}+ attributes; cannot choose a cap')
    cap = max(reference_sizes)
    rng = random.Random(seed)
    selected = []
    for count, indices in sorted(bins.items()):
        selected.extend(rng.sample(indices, cap) if len(indices) > cap else indices)
    return sorted(selected), cap


def histogram(rows):
    counts = Counter(sum(written_counts(row)) for row in rows)
    return {str(count): n for count, n in sorted(counts.items())}


def sample_study(name, config, seed=42, reference_min=6):
    source = ROOT / 'dataset/processed' / (config['base'] + '_human-combined')
    rows = load_from_disk(str(source))
    selected, cap = capped_indices(rows, seed, reference_min)
    sampled = rows.select(selected)
    examples = json.loads((ROOT / 'human_study' / f'in_context_examples_{name}.json').read_text())
    eligible = hold_out_examples(rows, examples, config['positive'])
    eligible_keys = {(row[config['positive']], row['human_query']) for row in eligible}
    evaluation = sampled.select([i for i, row in enumerate(sampled)
                                 if (row[config['positive']], row['human_query']) in eligible_keys])
    report = {
        'modality': name, 'source_dataset': str(source), 'source_fingerprint': rows._fingerprint,
        'rule': 'cap = max frequency among total-attribute counts >= reference_min; keep min(frequency, cap) per count',
        'seed': seed, 'reference_min': reference_min, 'cap_per_attribute_count': cap,
        'source_rows': len(rows), 'sampled_rows': len(sampled), 'removed_rows': len(rows) - len(sampled),
        'sampled_in_context_rows': len(evaluation),
        'held_out_after_sampling': len(sampled) - len(evaluation),
        'selected_source_indices': selected,
        'source_histogram': histogram(rows), 'sampled_histogram': histogram(sampled),
        'in_context_histogram': histogram(evaluation),
    }
    target = ROOT / 'dataset/processed' / (config['base'] + '_human-matched')
    for suffix, dataset in [('', sampled), ('-in-context', evaluation)]:
        path = Path(str(target) + suffix)
        save_version(dataset, path, report, report_name='sampling_report.json')
    print(f'{name}: cap={cap}, retained={len(sampled)}/{len(rows)}, '
          f'in-context={len(evaluation)}, seed={seed}', flush=True)
    return report


def sample_all(seed=42, reference_min=6):
    return {name: sample_study(name, config, seed, reference_min)
            for name, config in STUDIES.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--reference-min', type=int, default=6)
    args = parser.parse_args()
    sample_all(args.seed, args.reference_min)


if __name__ == '__main__':
    main()
