"""Offline test-set inference for text models.

Reconstructs the exact test split used by train_text.py (leakage-free `split` column
when present, otherwise the legacy 80/10/10 query split with seed 42), encodes the
test corpus and queries with one model, and writes predictions to <run-dir>/preds/
so all analysis can run offline. See utils/test_inference.py for the file formats.
"""

import argparse
import os

import pandas as pd
import torch
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split

from utils.test_inference import (
    build_corpus_and_queries,
    normalize,
    pair_similarities,
    quick_metrics,
    rank_top_k,
    write_jsonl,
    write_meta,
)

# Metadata columns copied verbatim into triplets.jsonl when the dataset has them.
PASSTHROUGH_COLUMNS = ["query_distance", "negative_example_source", "item", "positive_id", "negative_id"]


def load_test_split(dataset_path, query_key):
    """Mirror of train_text.py's dataset preparation: rename, filter, split, take test."""
    dataset = load_from_disk(dataset_path)
    dataset = dataset.rename_column(query_key, "query")
    for column in ["original_query", "nl_query", "rephrased_query"]:
        if column in dataset.column_names:
            dataset = dataset.remove_columns([column])
    dataset = dataset.rename_column("query", "anchor")
    dataset = dataset.rename_column("positive_example", "positive")
    dataset = dataset.rename_column("negative_example", "negative")
    dataset = dataset.filter(lambda x: x["positive"] != x["negative"])

    if "split" in dataset.column_names:
        return dataset.filter(lambda x: x["split"] == "test")

    # Legacy query-only split (same seeds/sizes as train_text.py)
    unique_queries = pd.Series(dataset["anchor"]).drop_duplicates().tolist()
    _, temp_queries = train_test_split(unique_queries, test_size=0.2, random_state=42)
    _, test_queries = train_test_split(temp_queries, test_size=0.5, random_state=42)
    test_queries_set = set(test_queries)
    return dataset.filter(lambda x: x["anchor"] in test_queries_set)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Model dir (…/final) or HF model name")
    parser.add_argument("--dataset", type=str, required=True, help="Processed dataset directory")
    parser.add_argument("--query-kind", choices=["original", "synthetic", "rephrased"], default="original")
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory; preds are written to <run-dir>/preds")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    query_key = {"original": "original_query", "synthetic": "nl_query", "rephrased": "rephrased_query"}[args.query_kind]

    test_dataset = load_test_split(args.dataset, query_key)
    print(f"Test split: {len(test_dataset):,} rows")

    corpus, corpus_to_idx, queries, query_to_qid, positives = build_corpus_and_queries(test_dataset)
    print(f"Corpus: {len(corpus):,} unique product texts | Queries: {len(queries):,}")

    print(f"Loading model {args.model_path}")
    model = SentenceTransformer(args.model_path, device=device)
    encode = lambda texts: normalize(
        model.encode(texts, convert_to_tensor=True, batch_size=args.batch_size, show_progress_bar=True)
    )
    corpus_embeddings = encode(corpus)
    query_embeddings = encode(queries)

    top_ids, top_scores = rank_top_k(query_embeddings, corpus_embeddings, args.top_k)

    # Per-row triplet similarities reuse the corpus/query embeddings via index lookup.
    anchor_ids = torch.tensor([query_to_qid[a] for a in test_dataset["anchor"]])
    pos_ids = [corpus_to_idx[p] for p in test_dataset["positive"]]
    neg_ids = [corpus_to_idx[n] for n in test_dataset["negative"]]
    sim_pos = pair_similarities(query_embeddings[anchor_ids], corpus_embeddings[torch.tensor(pos_ids)])
    sim_neg = pair_similarities(query_embeddings[anchor_ids], corpus_embeddings[torch.tensor(neg_ids)])

    preds_dir = os.path.join(args.run_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)

    write_jsonl(os.path.join(preds_dir, "corpus.jsonl"),
                [{"corpus_id": i, "text": text} for i, text in enumerate(corpus)])
    write_jsonl(os.path.join(preds_dir, "queries.jsonl"), [
        {
            "query_id": qid,
            "query": query,
            "positive_corpus_ids": positives[qid],
            "top_k": [[int(c), round(float(s), 5)] for c, s in zip(top_ids[qid], top_scores[qid])],
        }
        for qid, query in enumerate(queries)
    ])

    passthrough = [c for c in PASSTHROUGH_COLUMNS if c in test_dataset.column_names]
    write_jsonl(os.path.join(preds_dir, "triplets.jsonl"), [
        {
            "query_id": int(anchor_ids[i]),
            "positive_corpus_id": pos_ids[i],
            "negative_corpus_id": neg_ids[i],
            "sim_pos": round(float(sim_pos[i]), 5),
            "sim_neg": round(float(sim_neg[i]), 5),
            **{c: test_dataset[i][c] for c in passthrough},
        }
        for i in range(len(test_dataset))
    ])

    metrics = quick_metrics(top_ids, positives)
    print("Quick metrics:", metrics)
    write_meta(preds_dir, args, {
        "modality": "text",
        "n_test_rows": len(test_dataset),
        "n_corpus": len(corpus),
        "n_queries": len(queries),
        "metrics": metrics,
    })


if __name__ == "__main__":
    main()
