"""Offline held-out inference for both modalities.

Reconstructs the exact split used by train.py -- the split logic is imported from
train.py rather than copied, so it cannot drift -- encodes that split's corpus and queries
with one model, and writes predictions to <run-dir>/preds/ (test) or <run-dir>/preds_val/
(validation) so all analysis can run offline. See utils/test_inference.py for the formats.

--query-kind human scores the human-written queries (human_study/download_human_labels.py)
and writes to <run-dir>/preds_human/. The human dataset is evaluation-only: every row is
split=test, and train.py has no `human` query kind. Its corpus is the test-split corpus of
--distractor-dataset / --distractor-query-kind (the dataset and query kind the model was
trained on, so the same corpus as its preds/) plus the human pairs' own products; the human
set alone is a few hundred products and saturates every cutoff. paper.sh runs it for every
test row.

--split validation is what every hyperparameter sweep must read: selecting V or the
easy-negative value on test numbers tunes on the reported set. The main grid reports test;
the ablations select on val.

Supersedes test_text.py and test_multimodal.py.
"""

import argparse
import os

import torch
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer

from train import QUERY_COLUMNS, encode_documents, split_dataset

# Query kinds test.py can score: the training kinds plus the evaluation-only human queries.
EVAL_QUERY_COLUMNS = {**QUERY_COLUMNS, "human": "human_query"}
PREDS_SUBDIRS = {"test": "preds", "validation": "preds_val", "human": "preds_human"}
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


def load_eval_split(dataset_path, query_key, split_seed, split="test"):
    """Mirror of train.py's dataset preparation: rename, filter, split, take one split."""
    dataset = load_from_disk(dataset_path)
    if query_key not in dataset.column_names:
        raise ValueError(f"Requested query field '{query_key}' not found in columns: {dataset.column_names}")

    dataset = dataset.rename_column(query_key, "anchor")
    for column in EVAL_QUERY_COLUMNS.values():
        if column in dataset.column_names:
            dataset = dataset.remove_columns([column])
    dataset = dataset.rename_column("positive_example", "positive")
    # The human sets carry no hard negative (download_human_labels.py drops it as unverified
    # against the query the annotator wrote), so everything keyed on it is conditional.
    if "negative_example" in dataset.column_names:
        dataset = dataset.rename_column("negative_example", "negative")
        dataset = dataset.filter(lambda x: x["positive"] != x["negative"] and bool(x["anchor"]))
    else:
        dataset = dataset.filter(lambda x: bool(x["anchor"]))

    _, val_dataset, test_dataset = split_dataset(dataset, seed=split_seed)
    return {"validation": val_dataset, "test": test_dataset}[split]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["text", "multimodal"], required=True)
    parser.add_argument("--model-path", type=str, required=True, help="Model dir (.../final) or HF model name")
    parser.add_argument("--dataset", type=str, required=True, help="Processed dataset directory")
    parser.add_argument("--query-kind", choices=list(EVAL_QUERY_COLUMNS), required=True,
                        help="human reads the *_human dataset and is test-split only")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Run directory; preds go to <run-dir>/preds (test), <run-dir>/preds_val "
                             "(validation) or <run-dir>/preds_human (human queries)")
    parser.add_argument("--split", choices=["validation", "test"], default="test",
                        help="Which held-out split to score. Sweeps must use validation.")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"Default per modality: {DEFAULT_BATCH_SIZES}")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--distractor-dataset", type=str, default=None,
                        help="human only: dataset whose test-split corpus is added as distractors")
    parser.add_argument("--distractor-query-kind", choices=list(QUERY_COLUMNS), default=None,
                        help="human only: query kind that defines the distractor dataset's test split")
    args = parser.parse_args()
    if args.query_kind == "human" and args.split != "test":
        raise ValueError("--query-kind human is test-split only")
    has_distractors = args.distractor_dataset is not None or args.distractor_query_kind is not None
    if (args.query_kind == "human") != has_distractors or (
            has_distractors and None in (args.distractor_dataset, args.distractor_query_kind)):
        raise ValueError("--distractor-dataset and --distractor-query-kind are required with "
                         "--query-kind human and not accepted otherwise")

    modality = args.modality
    batch_size = args.batch_size if args.batch_size is not None else DEFAULT_BATCH_SIZES[modality]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    test_dataset = load_eval_split(args.dataset, EVAL_QUERY_COLUMNS[args.query_kind], args.split_seed,
                                   split=args.split)
    print(f"{args.split} split: {len(test_dataset):,} rows")

    distractors = []
    if args.query_kind == "human":
        distractor_split = load_eval_split(args.distractor_dataset,
                                           QUERY_COLUMNS[args.distractor_query_kind], args.split_seed)
        distractors = list(distractor_split["positive"]) + list(distractor_split["negative"])
        print(f"Distractors: {len(set(distractors)):,} unique items from the "
              f"{args.distractor_query_kind} test split of {args.distractor_dataset}")
    corpus, corpus_to_idx, queries, query_to_qid, positives = build_corpus_and_queries(
        test_dataset, distractors=distractors)
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
    has_negatives = "negative" in test_dataset.column_names
    sim_pos = pair_similarities(query_embeddings[anchor_ids], corpus_embeddings[torch.tensor(pos_ids)])
    if has_negatives:
        neg_ids = [corpus_to_idx[n] for n in test_dataset["negative"]]
        sim_neg = pair_similarities(query_embeddings[anchor_ids], corpus_embeddings[torch.tensor(neg_ids)])

    preds_dir = os.path.join(args.run_dir, PREDS_SUBDIRS["human" if args.query_kind == "human" else args.split])
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
    # triplets.jsonl is one row per (query, positive, hard negative); a dataset with no
    # negative has no triplet to write, and the win-rate analysis reads this file only for
    # the splits that have one.
    if has_negatives:
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
        "split": args.split,
        "n_test_rows": len(test_dataset),
        "n_corpus": len(corpus),
        "n_queries": len(queries),
        "metrics": metrics,
    })


if __name__ == "__main__":
    main()
