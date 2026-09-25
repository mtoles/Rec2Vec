"""Compare collected human attribute counts with the current synthetic test queries.

Run human_study/combine_human_labels.py first to import the two backfill sheets.
The combined curve includes every usable collected annotation from both studies;
no quota or projected answer contributes to it. A sampled curve caps each total-
attribute bin at the largest bin among counts 6+, separately by modality (seed 42).
The original study and the full combined distribution are shown too.

Synthetic counts use stored selected_* lists for images and recorded BM25 generation
features for text. Human counts use
Q1/Q2 attribute lists, not attribute extraction from the free-form Q3 query.

Usage: python paper/draw_attr_hist.py [--out tmp/attr_hist]
Writes PDF, PNG, JSON statistics, and CSV joint histograms. Previous outputs are
archived under a nearby old/ directory.
"""

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from datasets import load_from_disk

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / 'human_study'))
import attr_quota
from sample_human_labels import sample_all
from utils.condition_counts import original_condition_counts
from utils.paper_analysis import active_dataset_bases

PANELS = [
    ('Text', 'feature-distance-dataset_gemini-2.5-flash_1000000_nolek', 'positive_id'),
    ('Image', 'deepfashion-inshop-image-triplets_hf_20000', 'positive_product_id'),
]
FEATURE_KEYS = ['selected_pos_features', 'selected_common_features',
                'selected_neg_features', 'selected_neither_features']


def stated_counts(row):
    return (len(row['selected_pos_features']) + len(row['selected_common_features']),
            len(row['selected_neg_features']) + len(row['selected_neither_features']))


def totals(hist):
    out = Counter()
    for (required, excluded), count in hist.items():
        out[required + excluded] += count
    return out


def density(hist, grid):
    return np.array([hist[x] / sum(hist.values()) for x in grid])


def synthetic_histogram(path):
    dataset = load_from_disk(str(path))
    table = dataset.data.table
    usable = pc.and_(pc.not_equal(table['positive_example'], table['negative_example']),
                     pc.not_equal(table['rephrased_query'], ''))
    table = table.filter(usable)
    if 'split' in table.column_names:
        table = table.filter(pc.equal(table['split'], 'test'))
    else:
        from train import seeded_query_split
        queries = list(dict.fromkeys(table['rephrased_query'].to_pylist()))
        _, _, test_queries = seeded_query_split(queries, seed=42)
        table = table.filter(pc.is_in(table['rephrased_query'], value_set=pa.array(sorted(test_queries))))
    wanted = set(table['rephrased_query'].to_pylist())
    return Counter(original_condition_counts(path, wanted).values())


def statistics(hist):
    n = sum(hist.values())
    if not n:
        raise ValueError('Cannot summarize an empty collected dataset')
    values = np.array(list(totals(hist).elements()))
    return {
        'rows': n,
        'mean_required': sum(a * count for (a, b), count in hist.items()) / n,
        'mean_excluded': sum(b * count for (a, b), count in hist.items()) / n,
        'mean_attributes': float(values.mean()), 'median_attributes': float(np.median(values)),
        'min_attributes': int(values.min()), 'max_attributes': int(values.max()),
        'pct_at_most_6': float(100 * np.mean(values <= 6)),
        'pct_no_exclusions': 100 * sum(v for (a, b), v in hist.items() if b == 0) / n,
    }


def total_variation(left, right):
    return 0.5 * sum(abs(left[k] / sum(left.values()) - right[k] / sum(right.values()))
                     for k in set(left) | set(right))


def analyze(title, stem, id_column):
    base = REPO_ROOT / 'dataset/processed' / stem
    modality = 'multimodal' if title == 'Image' else 'text'
    synthetic_base = REPO_ROOT / 'dataset/processed' / active_dataset_bases(REPO_ROOT / 'paper.sh')[modality]
    synthetic = Path(str(synthetic_base) + '_rephrased-in-context')
    first = Path(str(base) + '_human')
    combined = Path(str(base) + '_human-combined')
    evaluation = Path(str(combined) + '-in-context')
    sampled = Path(str(base) + '_human-matched')
    sampled_evaluation = Path(str(sampled) + '-in-context')
    rows = load_from_disk(str(combined))
    histograms = {
        'synthetic_test': synthetic_histogram(synthetic),
        'human_first': attr_quota.labeled(first, id_column)[0],
        'human_backfill': Counter(attr_quota.written_counts(row) for row in rows
                                  if row['source_study'] == 'backfill'),
        'human_combined': Counter(attr_quota.written_counts(row) for row in rows),
        'human_combined_in_context': attr_quota.labeled(evaluation, id_column)[0],
        'human_sampled': attr_quota.labeled(sampled, id_column)[0],
        'human_sampled_in_context': attr_quota.labeled(sampled_evaluation, id_column)[0],
    }
    report = {
        'sources': {'synthetic_test': str(synthetic), 'human_first': str(first),
                    'human_combined': str(combined), 'human_combined_in_context': str(evaluation),
                    'human_sampled': str(sampled), 'human_sampled_in_context': str(sampled_evaluation)},
        'count_method': {'synthetic': 'stored feature lists' if title == 'Image' else
                         'recorded BM25 generation feature lists',
                         'human': 'Q1/Q2 lists via attr_quota.written_counts'},
        'statistics': {name: statistics(hist) for name, hist in histograms.items()},
        'joint_total_variation_from_synthetic': {
            name: total_variation(hist, histograms['synthetic_test'])
            for name, hist in histograms.items() if name != 'synthetic_test'},
        'combination': json.loads((combined / 'combination_report.json').read_text()),
        'sampling': json.loads((sampled / 'sampling_report.json').read_text()),
    }
    return histograms, report


def draw_panel(ax, title, histograms):
    series = [('Synthetic, current test', 'synthetic_test', '#8a8a82'),
              ('Human, study 1', 'human_first', '#2a78d6'),
              ('Human, studies 1+2 collected', 'human_combined', '#eb6834'),
              ('Human, matched evaluation', 'human_sampled_in_context', '#16815d')]
    grid = np.arange(max(max(totals(histograms[key])) for _, key, _ in series) + 1)
    for i, (label, key, color) in enumerate(series):
        hist = totals(histograms[key])
        label += f' (n={sum(hist.values()):,})'
        if i == 0:
            ax.bar(grid, density(hist, grid), width=1, color=color, alpha=0.4, label=label)
        else:
            ax.step(grid, density(hist, grid), where='mid', color=color, linewidth=2, label=label)
        ax.axvline(statistics(histograms[key])['mean_attributes'], color=color,
                   linewidth=1, linestyle=':')
    ax.set_title(title)
    ax.set_xlabel('Attributes stated per query (required + excluded)')
    ax.set_xlim(-0.5, grid[-1] + 0.5)
    ax.set_xticks(np.arange(0, grid[-1] + 1, 4))
    ax.legend(fontsize=8, frameon=False)
    ax.spines[['top', 'right']].set_visible(False)


def generate(out=REPO_ROOT / 'tmp/attr_hist'):
    out = Path(out)
    sample_all()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    report, flat = {}, []
    for ax, (title, stem, id_column) in zip(axes, PANELS):
        histograms, report[title] = analyze(title, stem, id_column)
        draw_panel(ax, title, histograms)
        for name, hist in histograms.items():
            s = report[title]['statistics'][name]
            print(f'{title} {name}: n={s["rows"]}, mean={s["mean_attributes"]:.2f}, '
                  f'median={s["median_attributes"]:.0f}', flush=True)
            for (required, excluded), count in sorted(hist.items()):
                flat.append({'modality': title, 'dataset': name, 'required': required,
                             'excluded': excluded, 'count': count,
                             'fraction': count / sum(hist.values())})
    axes[0].set_ylabel('Fraction of queries')
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = [out.with_suffix(ext) for ext in ('.pdf', '.png', '.json', '.csv')
                if out.with_suffix(ext).exists()]
    if existing:
        archive = out.parent / 'old' / (out.name + '_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f'))
        archive.mkdir(parents=True)
        for path in existing:
            path.rename(archive / path.name)
    fig.savefig(out.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(out.with_suffix('.png'), dpi=180, bbox_inches='tight')
    out.with_suffix('.json').write_text(json.dumps(report, indent=2) + '\n')
    with out.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    return report, fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=REPO_ROOT / 'tmp/attr_hist')
    args = parser.parse_args()
    generate(args.out)


if __name__ == '__main__':
    main()
