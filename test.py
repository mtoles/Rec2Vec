"""Offline test-set inference for both modalities.

Reconstructs the exact test split used by train.py -- the split logic is imported from
train.py rather than copied, so it cannot drift -- encodes the test corpus and queries
with one model, and writes predictions to <run-dir>/preds/ so all analysis can run
offline. See utils/test_inference.py for the file formats.

Supersedes test_text.py and test_multimodal.py.
"""

import argparse
import os

import torch
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer

from train import QUERY_COLUMNS, encode_documents, split_dataset
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
PASSTHROUGH_COLUMNS = [
    "query_distance", "negative_example_source", "item",
    "positive_id", "negative_id",
    "positive_product_id", "negative_product_id", "positive_category", "negative_category",
]

DEFAULT_BATCH_SIZES = {"text": 256, "multimodal": 64}

# corpus.jsonl names its payload field by content type, matching what the old
# per-modality scripts wrote and what existing preds consumers read.
CORPUS_FIELDS = {"text": "text", "multimodal": "image_path"}


def load_test_split(dataset_path, query_key, split_seed):
    """Mirror of train.py's dataset preparation: rename, filter, split, take test."""
    dataset = load_from_disk(dataset_path)
    if query_key not in dataset.column_names:
        raise ValueError(f"Requested query field '{query_key}' not found in columns: {dataset.column_names}")

    dataset = dataset.rename_column(query_key, "anchor")
    for column in QUERY_COLUMNS.values():
        if column in dataset.column_names:
            dataset = dataset.remove_columns([column])
    dataset = dataset.rename_column("positive_example", "positive")
    dataset = dataset.rename_column("negative_example", "negative")
    dataset = dataset.filter(lambda x: x["positive"] != x["negative"] and bool(x["anchor"]))

    _, _, test_dataset = split_dataset(dataset, seed=split_seed)
    return test_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["text", "multimodal"], required=True)
    parser.add_argument("--model-path", type=str, required=True, help="Model dir (.../final) or HF model name")
    parser.add_argument("--dataset", type=str, required=True, help="Processed dataset directory")
    parser.add_argument("--query-kind", choices=["original", "synthetic", "rephrased"], required=True)
    parser.add_argument("--run-dir", type=str, required=True, help="Run directory; preds are written to <run-dir>/preds")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"Default per modality: {DEFAULT_BATCH_SIZES}")
    parser.add_argument("--split-seed", type=int, default=42)
    args = parser.parse_args()

    modality = args.modality
    batch_size = args.batch_size if args.batch_size is not None else DEFAULT_BATCH_SIZES[modality]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_dataset = load_test_split(args.dataset, QUERY_COLUMNS[args.query_kind], args.split_seed)
    print(f"Test split: {len(test_dataset):,} rows")

    corpus, corpus_to_idx, queries, query_to_qid, positives = build_corpus_and_queries(test_dataset)
    print(f"Corpus: {len(corpus):,} unique items | Queries: {len(queries):,}")

    print(f"Loading model {args.model_path}")
    model = SentenceTransformer(
        args.model_path, device=device, trust_remote_code=(modality == "multimodal")
    )
    corpus_embeddings = normalize(
        encode_documents(model, corpus, modality, batch_size, show_progress_bar=True)
    )
    query_embeddings = normalize(
        model.encode(queries, convert_to_tensor=True, batch_size=batch_size, show_progress_bar=True)
    )

    top_ids, top_scores = rank_top_k(query_embeddings, corpus_embeddings, args.top_k)

    # Per-row triplet similarities reuse the corpus/query embeddings via index lookup.
    anchor_ids = torch.tensor([query_to_qid[a] for a in test_dataset["anchor"]])
    pos_ids = [corpus_to_idx[p] for p in test_dataset["positive"]]
    neg_ids = [corpus_to_idx[n] for n in test_dataset["negative"]]
    sim_pos = pair_similarities(query_embeddings[anchor_ids], corpus_embeddings[torch.tensor(pos_ids)])
    sim_neg = pair_similarities(query_embeddings[anchor_ids], corpus_embeddings[torch.tensor(neg_ids)])

    preds_dir = os.path.join(args.run_dir, "preds")
    os.makedirs(preds_dir, exist_ok=True)

    corpus_field = CORPUS_FIELDS[modality]
    write_jsonl(os.path.join(preds_dir, "corpus.jsonl"),
                [{"corpus_id": i, corpus_field: item} for i, item in enumerate(corpus)])
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
        "modality": modality,
        "n_test_rows": len(test_dataset),
        "n_corpus": len(corpus),
        "n_queries": len(queries),
        "metrics": metrics,
    })


if __name__ == "__main__":
    main()
