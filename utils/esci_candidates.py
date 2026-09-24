"""ESCI comparison pools shared by construction, migration, and Baseline sampling."""

from collections import defaultdict

ESCI_NEGATIVE_LABELS = frozenset({"Substitute", "Irrelevant"})
MIN_CANDIDATES = 2


def eligible_query_ids(frame):
    """Cheap source-data gate before pair selection or any generative calls.

    Count distinct products across Substitute AND Irrelevant, never annotation rows.
    Language checks and generation failures are handled by a second gate on the pairs.
    """
    negatives = frame[frame["esci_label"].isin(ESCI_NEGATIVE_LABELS)]
    counts = negatives.groupby("query_id")["product_id"].nunique()
    return set(counts[counts >= MIN_CANDIDATES].index)


def group_key(row):
    """Keep candidates within one split, original query, and positive product."""
    positive = row.get("positive_product")
    positive_id = positive["product_id"] if positive is not None else row["positive_id"]
    query = row.get("original_query", row.get("query"))
    if not query:
        raise ValueError("ESCI candidate pools require the original query")
    return (row.get("split", row.get("_split", "")), query, positive_id)


def pair_key(row):
    return (*group_key(row), row["nl_query"])


def comparison(row):
    """(negative product id, document), or None for an ineligible comparison."""
    if row["negative_example_source"] not in ESCI_NEGATIVE_LABELS:
        return None
    positive = row.get("positive_product")
    if positive is not None:
        negative = row["hard_neg_product"]
        pos_id, pos_text = positive["product_id"], positive["product_text"]
        neg_id, neg_text = negative["product_id"], negative["product_text"]
    else:
        pos_id, pos_text = row["positive_id"], row["positive_example"]
        neg_id, neg_text = row["negative_id"], row["negative_example"]
    if neg_id == pos_id or not neg_text or neg_text == pos_text:
        return None
    return neg_id, neg_text


def candidate_pools(rows):
    """Distinct usable negative products; duplicate rows do not enlarge the pool."""
    pools = defaultdict(dict)
    for row in rows:
        candidate = comparison(row)
        if candidate is not None:
            product_id, document = candidate
            pools[group_key(row)][product_id] = document
    return dict(pools)


def eligible_groups(pools):
    # Identical descriptions are one retrievable document even if ESCI assigns two IDs.
    return {key for key, pool in pools.items()
            if len(pool) >= MIN_CANDIDATES and len(set(pool.values())) >= MIN_CANDIDATES}


def filter_raw_pairs(rows):
    """Apply the same eligibility gate before/after generation and after splitting."""
    eligible = eligible_groups(candidate_pools(rows))
    return [row for row in rows if comparison(row) is not None and group_key(row) in eligible]


def filter_processed_dataset(dataset):
    """Retain eligible comparison rows and their existing easy-negative twins.

    Queries, splits, rephrasings, distances, and common easy negatives are preserved.
    This is the no-API migration path for already-generated GOLD datasets. Mined or
    Baseline siblings must be rebuilt from this filtered GOLD base, not filtered alone.
    """
    wanted = ("split", "original_query", "nl_query", "positive_id", "negative_id",
              "positive_example", "negative_example", "negative_example_source")
    rows = dataset.select_columns([c for c in wanted if c in dataset.column_names]).to_list()
    unknown = {r["negative_example_source"] for r in rows} - ESCI_NEGATIVE_LABELS - {"random"}
    if unknown:
        raise ValueError(f"Filter the GOLD base before constructing negative variants: {unknown}")
    pools = candidate_pools(rows)
    eligible = eligible_groups(pools)
    hard_indices = {i for i, r in enumerate(rows)
                    if comparison(r) is not None and group_key(r) in eligible}
    keys = {pair_key(rows[i]) for i in hard_indices}
    indices = [i for i, r in enumerate(rows)
               if i in hard_indices or (r["negative_example_source"] == "random" and pair_key(r) in keys)]
    if not hard_indices:
        raise ValueError("No examples have two eligible Substitute/Irrelevant candidates")
    report = {
        "minimum_candidates": MIN_CANDIDATES,
        "candidate_labels": sorted(ESCI_NEGATIVE_LABELS),
        "input_rows": len(dataset), "output_rows": len(indices),
        "retained_groups": len(eligible), "removed_groups": len(pools) - len(eligible),
        "retained_comparison_rows": len(hard_indices),
    }
    return dataset.select(indices), report
