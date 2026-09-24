"""Shared helpers for the offline test-set inference scripts (test_text.py, test_multimodal.py).

Each test script reconstructs its trainer's exact test split, encodes the corpus and
queries with one model, and dumps predictions to jsonl files inside the model's run
directory so all paper analysis can happen offline:

    <run_dir>/preds/
        corpus.jsonl    {"corpus_id", "text" | "image_path"}
        queries.jsonl   {"query_id", "query", "positive_corpus_ids", "top_k": [[corpus_id, score], ...]}
        triplets.jsonl  one row per test example: metadata + sim_pos/sim_neg (corpus ids, not full texts)
        meta.json       model/dataset/git provenance, args, quick accuracy/recall sanity metrics
"""

import glob
import json
import os
import subprocess
from datetime import datetime, timezone

import numpy as np
import torch


def first_seen_unique(items):
    """Deterministic de-dup preserving first-appearance order (unlike list(set(...)))."""
    return list(dict.fromkeys(items))


def build_corpus_and_queries(dataset, anchor_key="anchor", pos_key="positive", neg_key="negative",
                             distractors=()):
    """Corpus of unique positive+negative items (plus `distractors`, items that are relevant to
    no query) and per-query relevant corpus ids.

    A dataset with no negative column contributes positives only: the human sets carry no hard
    negative, since none was verified against the query the annotator wrote.
    """
    negatives = list(dataset[neg_key]) if neg_key in dataset.column_names else []
    corpus = first_seen_unique(list(distractors) + list(dataset[pos_key]) + negatives)
    corpus_to_idx = {item: i for i, item in enumerate(corpus)}

    queries = []
    query_to_qid = {}
    positives = []  # qid -> ordered unique positive corpus ids
    for anchor, pos in zip(dataset[anchor_key], dataset[pos_key]):
        if anchor not in query_to_qid:
            query_to_qid[anchor] = len(queries)
            queries.append(anchor)
            positives.append([])
        pid = corpus_to_idx[pos]
        if pid not in positives[query_to_qid[anchor]]:
            positives[query_to_qid[anchor]].append(pid)
    return corpus, corpus_to_idx, queries, query_to_qid, positives


def normalize(embeddings):
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def rank_top_k(query_embeddings, corpus_embeddings, top_k, chunk_size=512):
    """Cosine top-k per query (embeddings must be L2-normalized). Returns (ids, scores) numpy arrays."""
    top_k = min(top_k, corpus_embeddings.shape[0])
    all_ids, all_scores = [], []
    for start in range(0, query_embeddings.shape[0], chunk_size):
        scores = query_embeddings[start:start + chunk_size] @ corpus_embeddings.T
        values, ids = torch.topk(scores, k=top_k, dim=1)
        all_ids.append(ids.cpu().numpy())
        all_scores.append(values.cpu().numpy())
    return np.concatenate(all_ids), np.concatenate(all_scores)


def pair_similarities(query_embeddings, item_embeddings):
    """Row-wise cosine similarity for (query, item) pairs of equal length."""
    return (query_embeddings * item_embeddings).sum(dim=1).cpu().numpy()


def quick_metrics(top_ids, positives, ks=(1, 5, 10, 20, 100)):
    """Accuracy@k (any positive in top-k) and Recall@k, averaged over queries."""
    metrics = {}
    for k in ks:
        if k > top_ids.shape[1]:
            continue
        accs, recs = [], []
        for row, pos in zip(top_ids, positives):
            hits = len(set(row[:k].tolist()) & set(pos))
            accs.append(1.0 if hits else 0.0)
            recs.append(hits / len(pos))
        metrics[f"acc@{k}"] = float(np.mean(accs))
        metrics[f"recall@{k}"] = float(np.mean(recs))
    return metrics


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows):,} rows -> {path}")


def git_sha():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
        ).stdout.strip() or None
    except OSError:
        return None


def path_mtime(path):
    return os.path.getmtime(path) if os.path.exists(path) else None


def dataset_mtime(dataset_dir):
    """Newest mtime among a saved dataset's payload files.

    The directory's own mtime is useless as provenance: `datasets` drops
    cache-*.arrow files in there on every load, so it tracks the last read
    rather than the last regeneration. paper.sh's staleness check uses the
    same definition.
    """
    payload = glob.glob(os.path.join(dataset_dir, "data-*.arrow"))
    payload += [os.path.join(dataset_dir, n) for n in ("dataset_info.json", "state.json")]
    mtimes = [os.path.getmtime(p) for p in payload if os.path.exists(p)]
    return max(mtimes) if mtimes else None


def write_meta(preds_dir, args, extra):
    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": git_sha(),
        "args": vars(args),
        "model_mtime": path_mtime(os.path.join(str(args.model_path), "modules.json")),
        "dataset_mtime": dataset_mtime(args.dataset),
        **extra,
    }
    path = os.path.join(preds_dir, "meta.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Wrote {path}")
    return meta
