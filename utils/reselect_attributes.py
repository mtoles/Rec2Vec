"""Re-run attribute selection and query rendering over an already-extracted raw dataset.

The expensive part of preprocess_text.py is the LLM feature extraction, and its output
(full_common_features / full_unique_pos_features / full_unique_neg_features /
full_neither_features) is already stored on every raw row. Changing how features are
*selected* from those pools therefore needs no LLM calls at all -- just a re-roll of the
selection and a re-render of nl_query.

preprocess_text.py cannot do this itself: it early-exits on its cached jsonl and reuses the
stored nl_query, so a new selection rule would never fire.

    python -m utils.reselect_attributes --raw <in.jsonl> --out <out.jsonl> [--seed 42]
"""

import argparse
import json
import random
from collections import Counter

from utils.query_render import choose_feature_counts, render_query


def reselect(row, rng, max_distance):
    pos = row["full_unique_pos_features"]
    neg = row["full_unique_neg_features"]
    common = row["full_common_features"]
    neither = row["full_neither_features"]

    query_distance = rng.randint(1, max_distance)
    n_pos, n_neg, n_common, n_neither = choose_feature_counts(
        query_distance, len(pos), len(neg), len(common), len(neither), rng.randint)

    selected_pos = rng.sample(pos, n_pos)
    selected_neg = rng.sample(neg, n_neg)
    selected_common = rng.sample(common, n_common)
    selected_neither = rng.sample(neither, n_neither)

    row["selected_pos_features"] = selected_pos
    row["selected_neg_features"] = selected_neg
    row["selected_common_features"] = selected_common
    row["selected_neither_features"] = selected_neither
    # The differentiating count is what the label regresses onto: only the features that
    # actually distinguish the two products count, never the common/neither context.
    row["query_distance"] = len(selected_pos) + len(selected_neg)
    row["nl_query"] = render_query(
        row["item"], selected_pos + selected_common, selected_neg + selected_neither)
    # Rephrasings are derived from nl_query, so any stored one is now stale.
    row["rephrased_query"] = ""
    # The image pipeline carries the easy negative's distance on the same row; it was never
    # measured either, so null it here too rather than leave the old 20.0 marker.
    if "easy_negative_query_distance" in row:
        row["easy_negative_query_distance"] = None
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, help="Input raw jsonl with full_* feature pools")
    parser.add_argument("--out", required=True, help="Output raw jsonl")
    parser.add_argument("--max-distance", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    attributes, distances = Counter(), Counter()
    n = 0
    dropped = 0
    with open(args.raw, encoding="utf-8") as src, open(args.out, "w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            if not (row["full_unique_pos_features"] or row["full_unique_neg_features"]):
                # No query can separate this pair.
                dropped += 1
                continue
            row = reselect(row, rng, args.max_distance)
            total = (len(row["selected_pos_features"]) + len(row["selected_neg_features"])
                     + len(row["selected_common_features"]) + len(row["selected_neither_features"]))
            attributes[total] += 1
            distances[row["query_distance"]] += 1
            dst.write(json.dumps(row) + "\n")
            n += 1

    print(f"Rewrote {n:,} rows -> {args.out}  (dropped {dropped:,} undifferentiable)")
    print(f"  attributes per query: min={min(attributes)} max={max(attributes)} "
          f"| n(1)={attributes.get(1, 0):,} n(2)={attributes.get(2, 0):,}")
    print(f"  query_distance:       min={min(distances)} max={max(distances)} "
          f"| n(1)={distances.get(1, 0):,}")


if __name__ == "__main__":
    main()
