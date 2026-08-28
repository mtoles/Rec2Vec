"""Query Recall vs. Attribute Count and Distance Evaluation.

Evaluates Accuracy/Recall@5/10/20 on the test split of the Rec2Vec dataset as a
function of:
  1. Number of Attributes: total count of all features specified in the query
     (positive + negative + common + neither).
  2. Query Distance: the number of differentiating attributes (the dataset's
     existing `query_distance` field).

Methodology
  1. Load the raw dataset and replicate the exact query-level 80/10/10
     train/val/test split used during training, isolating the 10% test queries.
  2. Build the retrieval corpus from all unique positive, hard negative, and easy
     negative products in the test split, mirroring `train_fork_val.py`.
  3. For each model, encode corpus and queries, batch-compute cosine similarity
     rankings, and measure Accuracy and Recall at K=5, 10, 20. Group and plot
     performance against both query attribute count and query distance.

Outputs are written to ./query_recall_by_attributes_and_distance/
"""

import os

# Set CUDA device to 5 as requested by the user
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

pd.set_option("display.max_columns", None)
sns.set_theme(style="whitegrid")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_PATH = "dataset/feature-distance-dataset_gemini-2.5-flash_1000000_fixed_distance.jsonl"
OUTPUT_DIR = os.path.splitext(os.path.basename(__file__))[0]

# One entry per model to score: (display name, model path).
MODELS = [
    ("Pre-trained (baseline)", "all-mpnet-base-v2"),
    ("Fine-tuned (Triplet)", "models/schachaf/baseline-triplet_keep"),
    ("Fine-tuned (Classic MSE)", "models/schachaf/classic-mse_best"),
    # 2026-08-04 run: train_fork.py --training-style baseline-infonce (base: microsoft/mpnet-base)
    # Disabled 2026-08-28: models/infonce-all-mpnet-base-v2 deleted to reclaim disk.
    # Re-enable against the paper run: models/text__all-mpnet-base-v2__infonce__
    #   feature-distance-dataset_gemini-2.5-flash_1000000_nolek__synthetic__note-paper/final
    # ("Fine-tuned (InfoNCE)", "models/infonce-all-mpnet-base-v2/final"),
]


def load_dataset(path):
    print(f"Loading dataset from {path}...")
    entries = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    print(f"Loaded {len(entries):,} total entries.")
    return entries


def split_test_entries(entries):
    """Replicate the query-level train/val/test split using random state 42."""
    unique_queries = pd.Series([e["nl_query"] for e in entries]).drop_duplicates().tolist()
    print(f"Total unique queries in dataset: {len(unique_queries):,}")

    train_queries, temp_queries = train_test_split(unique_queries, test_size=0.2, random_state=42)
    eval_queries, test_queries = train_test_split(temp_queries, test_size=0.5, random_state=42)
    test_queries_set = set(test_queries)

    print(
        f"Split sizes: {len(train_queries):,} train, {len(eval_queries):,} val, "
        f"{len(test_queries):,} test unique queries."
    )

    test_entries = [e for e in entries if e["nl_query"] in test_queries_set]
    print(f"Filtered test split entries: {len(test_entries):,}")
    return test_entries


def build_corpus(test_entries):
    """Build the retrieval corpus and the per-query attribute/distance mappings."""
    corpus_texts = list(set(
        [e["positive_product"]["product_text"] for e in test_entries] +
        [e["hard_neg_product"]["product_text"] for e in test_entries] +
        [e["easy_neg_product"]["product_text"] for e in test_entries]
    ))
    print(f"Retrieval corpus size: {len(corpus_texts):,} unique product texts.")
    corpus_to_idx = {txt: i for i, txt in enumerate(corpus_texts)}

    query_to_positives = {}
    query_to_num_attributes = {}
    query_to_distance = {}

    for e in test_entries:
        q = e["nl_query"]
        pos_txt = e["positive_product"]["product_text"]

        if q not in query_to_positives:
            query_to_positives[q] = set()
        query_to_positives[q].add(corpus_to_idx[pos_txt])

        # 1. Total number of attributes is the sum of all selected feature categories
        query_to_num_attributes[q] = (
            len(e["selected_pos_features"]) +
            len(e["selected_neg_features"]) +
            len(e["selected_common_features"]) +
            len(e["selected_neither_features"])
        )

        # 2. Query distance (existing differentiation field, which equals
        #    len(pos_features) + len(neg_features))
        query_to_distance[q] = e["query_distance"]

    unique_test_queries = list(query_to_positives.keys())
    relevant_corpus_ids = [list(query_to_positives[q]) for q in unique_test_queries]
    query_attr_counts = [query_to_num_attributes[q] for q in unique_test_queries]
    query_distances = [query_to_distance[q] for q in unique_test_queries]

    print(f"Evaluation queries count: {len(unique_test_queries):,}")
    return corpus_texts, unique_test_queries, relevant_corpus_ids, query_attr_counts, query_distances


def encode_texts(model, texts, batch_size=256):
    embeddings = model.encode(
        texts,
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=batch_size,
    )
    # L2 normalize so dot product equals cosine similarity
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy()


def score_model(name, path, corpus_texts, unique_test_queries, relevant_corpus_ids,
                query_attr_counts, query_distances):
    """Encode corpus + queries with one model and return its per-query metrics."""
    print(f"Loading {name} from {path}...")
    model = SentenceTransformer(path, device=DEVICE)

    print(f"Encoding corpus for {name}...")
    corpus_embeddings = encode_texts(model, corpus_texts)

    print(f"Encoding queries for {name}...")
    query_embeddings = encode_texts(model, unique_test_queries)

    print(f"Computing retrieval ranks and metrics for {name}...")
    batch_size = 256
    accuracies = {5: [], 10: [], 20: []}
    recalls = {5: [], 10: [], 20: []}

    for i in tqdm(range(0, len(query_embeddings), batch_size)):
        q_batch = query_embeddings[i:i + batch_size]
        scores = q_batch @ corpus_embeddings.T  # [batch_size, corpus_size]

        for idx in range(len(q_batch)):
            positives_set = set(relevant_corpus_ids[i + idx])
            ranked_indices = np.argsort(-scores[idx])

            for k in [5, 10, 20]:
                hits_in_k = positives_set.intersection(set(ranked_indices[:k]))
                # Accuracy@K (Hit@K) is 1.0 if at least one positive is retrieved
                accuracies[k].append(1.0 if len(hits_in_k) > 0 else 0.0)
                # Recall@K is the fraction of relevant positives retrieved
                recalls[k].append(len(hits_in_k) / len(positives_set))

    # Free the weights before the next model loads; they do not all fit on one GPU.
    del model, corpus_embeddings, query_embeddings
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    print(f"Done: {name}.")
    return pd.DataFrame({
        "num_attributes": query_attr_counts,
        "query_distance": query_distances,
        "acc@5": accuracies[5],
        "acc@10": accuracies[10],
        "acc@20": accuracies[20],
        "recall@5": recalls[5],
        "recall@10": recalls[10],
        "recall@20": recalls[20],
        "model": name,
    })


def plot_results(results_df, out_path):
    """Side-by-side recall plots grouped by attribute count and query distance."""
    grouped_attrs = (
        results_df.groupby(["model", "num_attributes"])[["acc@5", "acc@10", "acc@20"]]
        .mean().reset_index()
    )
    grouped_dist = (
        results_df.groupby(["model", "query_distance"])[["acc@5", "acc@10", "acc@20"]]
        .mean().reset_index()
    )

    fig, axes = plt.subplots(3, 2, figsize=(18, 18), sharey=True)
    metrics_k = [5, 10, 20]

    for idx, k in enumerate(metrics_k):
        # Left column: Accuracy/Recall vs. number of attributes
        ax_left = axes[idx, 0]
        sns.lineplot(
            data=grouped_attrs,
            x="num_attributes",
            y=f"acc@{k}",
            hue="model",
            marker="o",
            linewidth=2,
            ax=ax_left,
        )
        ax_left.set_title(f"Accuracy/Recall@{k} vs. Number of Attributes", fontsize=12, fontweight="bold")
        ax_left.set_ylabel(f"Recall@{k}", fontsize=10)
        ax_left.set_xlabel("Number of Attributes", fontsize=10)
        ax_left.set_ylim(0.0, 1.05)
        ax_left.legend(title="Model", loc="lower right", fontsize=8)

        # Right column: Accuracy/Recall vs. query distance
        ax_right = axes[idx, 1]
        sns.lineplot(
            data=grouped_dist,
            x="query_distance",
            y=f"acc@{k}",
            hue="model",
            marker="o",
            linewidth=2,
            ax=ax_right,
        )
        ax_right.set_title(f"Accuracy/Recall@{k} vs. Query Distance", fontsize=12, fontweight="bold")
        ax_right.set_ylabel(f"Recall@{k}", fontsize=10)
        ax_right.set_xlabel("Query Distance (Diff Features)", fontsize=10)
        ax_right.set_ylim(0.0, 1.05)
        ax_right.legend(title="Model", loc="lower right", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Wrote {out_path}")
    return grouped_attrs, grouped_dist


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Using device: {DEVICE}")
    print(f"Writing outputs to {OUTPUT_DIR}/")

    entries = load_dataset(DATASET_PATH)
    test_entries = split_test_entries(entries)
    (corpus_texts, unique_test_queries, relevant_corpus_ids,
     query_attr_counts, query_distances) = build_corpus(test_entries)

    model_results = {}
    for name, path in MODELS:
        model_results[name] = score_model(
            name, path, corpus_texts, unique_test_queries, relevant_corpus_ids,
            query_attr_counts, query_distances,
        )

    results_df = pd.concat(model_results.values(), ignore_index=True)
    print(f"{len(model_results)} models in the plot: {', '.join(model_results)}")
    print(f"{len(results_df):,} per-query rows total.")

    per_query_path = os.path.join(OUTPUT_DIR, "per_query_results.csv")
    results_df.to_csv(per_query_path, index=False)
    print(f"Wrote {per_query_path}")

    grouped_attrs, grouped_dist = plot_results(
        results_df, os.path.join(OUTPUT_DIR, "recall_by_attributes_and_distance.png")
    )
    grouped_attrs.to_csv(os.path.join(OUTPUT_DIR, "grouped_by_num_attributes.csv"), index=False)
    grouped_dist.to_csv(os.path.join(OUTPUT_DIR, "grouped_by_query_distance.csv"), index=False)

    print("Global Mean Accuracy/Recall by Model:")
    summary = results_df.groupby("model")[["acc@5", "acc@10", "acc@20"]].mean().reset_index()
    print(summary.to_string(index=False))
    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
