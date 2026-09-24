"""Paired permutation tests between two groups of runs scored on the same queries.

Every run of one modality and query kind is scored on the same queries against the same
corpus (the mined, mixed and random siblings rewrite the train split only), so two groups of
runs form a paired design with the query as the unit: one per-query difference in recall@k.
Each group is averaged over its training seeds before differencing, so the verdict is
conditional on the seeds that were trained; `compare` also tests every matched seed pairing
on its own, and the largest of those p-values says whether the pooled verdict rests on one
pairing.

The null is that the group label carries no information, under which the sign of every
paired difference is exchangeable. Text test queries share positive products (up to 30
queries per product), so a product's differences are flipped together -- the stratified
(clustered) sign-flip test. Image and human queries are one per product, where the cluster
is the query and this is the plain paired permutation test. p-values are Monte Carlo with
the add-one correction, (1 + #{T* >= T}) / (M + 1), which is never zero and is valid at any M.
"""

import functools
import os

import numpy as np
import pandas as pd

from utils.paper_analysis import _read_jsonl, load_run_preds


@functools.lru_cache(maxsize=None)
def _run_recall(run_dir, k, preds_subdir):
    """((query_id, positives) per query, recall@k per query) in the run's own query order."""
    queries = _read_jsonl(os.path.join(run_dir, preds_subdir, "queries.jsonl"))
    key = tuple((q["query_id"], tuple(sorted(q["positive_corpus_ids"]))) for q in queries)
    frame = load_run_preds(run_dir, ks=(k,), preds_subdir=preds_subdir)
    return key, frame[f"recall@{k}"].to_numpy(dtype=float)


def per_query_recall(run_dirs, k, preds_subdir="preds"):
    """(n_runs, n_queries) recall@k and the cluster id (positive product) of every query.

    Columns follow the first run's query order; every other run must score the same
    query_ids with the same positives, which is what makes the columns paired.
    """
    reference, _ = _run_recall(run_dirs[0], k, preds_subdir)
    rows = []
    for run_dir in run_dirs:
        key, recall = _run_recall(run_dir, k, preds_subdir)
        if key != reference:
            raise ValueError(f"{run_dir} scores different queries than {run_dirs[0]}")
        rows.append(recall)
    clusters = pd.factorize(pd.Series([positives for _, positives in reference]))[0]
    return np.stack(rows), clusters


def sign_flip_test(diff, clusters, n_perm=10000, seed=0, chunk=2000):
    """Mean paired difference and its Monte Carlo p-values under cluster-wise sign flips.

    Returns (delta, p_greater, p_two_sided): p_greater is for the one-sided alternative that
    the first group is better, p_two_sided for a difference in either direction. The flips
    are drawn per cluster and applied to every query in it.
    """
    n_queries = len(diff)
    cluster_sums = np.bincount(clusters, weights=diff)
    observed = cluster_sums.sum() / n_queries
    rng = np.random.default_rng(seed)
    permuted = np.empty(n_perm)
    for start in range(0, n_perm, chunk):
        size = min(chunk, n_perm - start)
        signs = rng.integers(0, 2, size=(size, len(cluster_sums)), dtype=np.int8) * 2 - 1
        permuted[start:start + size] = signs @ cluster_sums / n_queries
    # The identity flip reproduces the observed statistic up to summation order, so the
    # comparison carries a tolerance rather than relying on bit-equal floats.
    tolerance = 1e-12
    p_greater = (1 + int(np.sum(permuted >= observed - tolerance))) / (n_perm + 1)
    p_two_sided = (1 + int(np.sum(np.abs(permuted) >= abs(observed) - tolerance))) / (n_perm + 1)
    return float(observed), p_greater, p_two_sided


def compare(runs_a, runs_b, k, preds_subdir="preds", n_perm=10000, seed=0):
    """Group A against group B, both {training seed: run_dir} over the same seeds.

    The main verdict averages each group over its seeds before differencing. Each matched
    seed pairing (A's seed s against B's seed s) is then tested on its own.
    """
    if set(runs_a) != set(runs_b):
        raise ValueError(f"seed sets differ: {sorted(runs_a)} vs {sorted(runs_b)}")
    seeds = sorted(runs_a)
    run_dirs = [runs_a[s] for s in seeds] + [runs_b[s] for s in seeds]
    recall, clusters = per_query_recall(run_dirs, k, preds_subdir)
    a, b = recall[:len(seeds)], recall[len(seeds):]
    delta, p_greater, p_two_sided = sign_flip_test(a.mean(0) - b.mean(0), clusters, n_perm, seed)
    pairings = [(s, *sign_flip_test(a[i] - b[i], clusters, n_perm, seed))
                for i, s in enumerate(seeds)]
    return {
        "n_queries": int(recall.shape[1]),
        "n_clusters": int(clusters.max()) + 1,
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "delta": delta,
        "p_greater": p_greater,
        "p_two_sided": p_two_sided,
        "seed_pairings": pairings,
        "delta_pairing_min": min(d for _, d, _, _ in pairings),
        "delta_pairing_max": max(d for _, d, _, _ in pairings),
        "p_greater_pairing_max": max(p for _, _, p, _ in pairings),
        "p_two_sided_pairing_max": max(p for _, _, _, p in pairings),
    }


def holm_adjust(pvalues):
    """Holm step-down adjusted p-values, in the input order; valid under any dependence."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(np.argsort(p)):
        running = max(running, (m - rank) * p[i])
        adjusted[i] = min(1.0, running)
    return adjusted
