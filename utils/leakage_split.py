"""Build a train/val/test split of the text dataset with no leakage across queries or items.

Leakage sources this removes:
  1. the same query appearing in two splits,
  2. the same product appearing in two splits in any role (positive, hard negative, easy negative),
  3. products of the same ESCI subject (``item``) being spread across splits.

Recipe (see the survey in tmp/retrieval_dataset_survey.md):
  1. group rows by ``item``, unioning items that share an ``original_query``,
  2. hash each group into train/val/test,
  3. drop val/test rows whose positive or hard negative also appears in another split,
  4. re-draw every easy negative from its own split's product pool,
  5. assert the query and product id sets are pairwise disjoint.

Usage:
    python -m utils.leakage_split \
        --raw dataset/feature-distance-dataset_gemini-2.5-flash_1000000_fixed_distance.jsonl \
        --out dataset/processed/feature-distance-dataset_gemini-2.5-flash_1000000_nolek
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict

from datasets import Dataset as HFDataset

from utils.esci_candidates import filter_raw_pairs

# A random negative's violation count was never measured, so query_distance is left null and
# negative_example_source is the only marker. See utils/distance_labels.py.
EASY_NEGATIVE_DISTANCE = None


def _find(parent, x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _union(parent, a, b):
    parent.setdefault(a, a)
    parent.setdefault(b, b)
    ra, rb = _find(parent, a), _find(parent, b)
    if ra != rb:
        parent[ra] = rb


def item_groups(rows):
    """Union items that share an original_query, so a query never straddles two groups."""
    parent = {}
    query_to_item = {}
    for r in rows:
        parent.setdefault(r["item"], r["item"])
        prev = query_to_item.setdefault(r["original_query"], r["item"])
        if prev != r["item"]:
            _union(parent, prev, r["item"])
    return {item: _find(parent, item) for item in parent}


def assign_side(group, seed, val_frac, test_frac):
    h = hashlib.md5(f"{seed}:{group}".encode()).hexdigest()[:8]
    u = int(h, 16) / 0xFFFFFFFF
    if u < test_frac:
        return "test"
    if u < test_frac + val_frac:
        return "validation"
    return "train"


def build(raw_path, val_frac, test_frac, seed):
    rows = filter_raw_pairs([json.loads(line) for line in open(raw_path)])
    groups = item_groups(rows)
    side_of = {g: assign_side(g, seed, val_frac, test_frac) for g in set(groups.values())}
    for r in rows:
        r["_split"] = side_of[groups[r["item"]]]

    # Products that land on more than one side via a positive or hard-negative role.
    doc_sides = defaultdict(set)
    for r in rows:
        doc_sides[r["positive_product"]["product_id"]].add(r["_split"])
        doc_sides[r["hard_neg_product"]["product_id"]].add(r["_split"])
    conflicts = {d for d, s in doc_sides.items() if len(s) > 1}

    # Keep every train row; drop the val/test rows that reach into another split.
    kept = []
    dropped = 0
    for r in rows:
        touches = {r["positive_product"]["product_id"], r["hard_neg_product"]["product_id"]}
        if r["_split"] != "train" and touches & conflicts:
            dropped += 1
            continue
        kept.append(r)

    # Split conflict removal can turn a validation/test pool into a singleton.
    eligible_kept = filter_raw_pairs(kept)
    dropped += len(kept) - len(eligible_kept)
    kept = eligible_kept

    # Easy negatives are random catalog draws, so re-draw them inside each split.
    pool = defaultdict(dict)  # split -> {product_id: product_text}
    for r in kept:
        for key in ("positive_product", "hard_neg_product"):
            p = r[key]
            if p["product_id"] not in conflicts:
                pool[r["_split"]][p["product_id"]] = p["product_text"]
    pool_ids = {s: sorted(d) for s, d in pool.items()}

    rng = random.Random(seed)
    for r in kept:
        ids = pool_ids[r["_split"]]
        banned = {r["positive_product"]["product_id"], r["hard_neg_product"]["product_id"]}
        for _ in range(10):
            pid = ids[rng.randrange(len(ids))]
            if pid not in banned:
                break
        r["_easy_id"] = pid
        r["_easy_text"] = pool[r["_split"]][pid]

    return kept, dropped, conflicts


def to_hf_rows(kept):
    """Two rows per triple (hard and easy negative), matching preprocess_text.py's schema."""
    out = []
    for r in kept:
        base = {
            "original_query": r["original_query"],
            "nl_query": r["nl_query"],
            "rephrased_query": r.get("rephrased_query", ""),
            "positive_example": r["positive_product"]["product_text"],
            "item": r["item"],
            "positive_id": r["positive_product"]["product_id"],
            "split": r["_split"],
        }
        out.append({
            **base,
            "negative_example": r["hard_neg_product"]["product_text"],
            "negative_example_source": r["negative_example_source"],
            "negative_id": r["hard_neg_product"]["product_id"],
            "query_distance": float(r["query_distance"]),
        })
        out.append({
            **base,
            "negative_example": r["_easy_text"],
            "negative_example_source": "random",
            "negative_id": r["_easy_id"],
            "query_distance": EASY_NEGATIVE_DISTANCE,
        })
    return out


def verify(hf_rows):
    queries = defaultdict(set)
    docs = defaultdict(set)
    for r in hf_rows:
        queries[r["split"]].add(r["original_query"])
        docs[r["split"]].add(r["positive_id"])
        docs[r["split"]].add(r["negative_id"])
    splits = sorted(queries)
    for i, a in enumerate(splits):
        for b in splits[i + 1:]:
            assert not queries[a] & queries[b], f"query leak {a}/{b}"
            assert not docs[a] & docs[b], f"product leak {a}/{b}: {len(docs[a] & docs[b])}"
    return {s: (len(queries[s]), len(docs[s])) for s in splits}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="raw feature-distance jsonl")
    ap.add_argument("--out", required=True, help="output HF dataset directory")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    kept, dropped, conflicts = build(args.raw, args.val_frac, args.test_frac, args.seed)
    hf_rows = to_hf_rows(kept)
    stats = verify(hf_rows)

    counts = defaultdict(int)
    for r in kept:
        counts[r["_split"]] += 1
    print(f"conflicting products: {len(conflicts)}")
    print(f"dropped val/test triples: {dropped}")
    for s in sorted(counts):
        q, d = stats[s]
        print(f"{s:11s} {counts[s]:7d} triples  {2 * counts[s]:7d} rows  {q:6d} queries  {d:7d} products")

    HFDataset.from_list(hf_rows).save_to_disk(args.out)
    print(f"saved to {args.out}")


if __name__ == "__main__":
    main()
