"""Shared attribute selection and query rendering for the text and image pipelines.

Both preprocessors grew their own copy of this logic and drifted (see utils/distance_labels
for what that cost last time), so it lives in one place now.

The selection rule: a query needs at least one stated attribute, not one of each polarity.
The old guard forced a desired *and* an unwanted attribute into every query, which made 2 the
hard floor on attribute count -- single-attribute queries, the most constrained and most
interesting regime, were never generated at all.
"""

# The first clause is introduced with "that", any following clause with "and", so a
# single-polarity query reads correctly on its own ('that does not have: X', never
# 'and does not have: X').
WANT_CLAUSE = "has"
AVOID_CLAUSE = "does not have"


class UndifferentiableError(ValueError):
    """No query over this pair can tell the two products apart.

    Raised when both differentiating pools (unique_pos, unique_neg) are empty. A query built
    only from common/neither features has query_distance 0 -- it does not identify the
    positive, and training on it regresses a positive and a hard negative to the same score.
    Callers drop the pair."""


def choose_feature_counts(
    query_distance,
    n_unique_pos,
    n_unique_neg,
    n_common,
    n_neither,
    randint,
):
    """How many features to take from each pool, given a target differentiating distance.

    `query_distance` (= pos + neg) is the differentiating count and is preserved exactly when
    the pools allow it; common/neither features are context and do not count toward it.
    Returns (n_pos, n_neg, n_common, n_neither).
    """
    n_pos = min(randint(0, query_distance), n_unique_pos)
    n_neg = min(query_distance - n_pos, n_unique_neg)
    n_common = min(randint(0, query_distance), n_common)
    n_neither = min(randint(0, query_distance), n_neither)

    # At least one *differentiating* feature, always. Common and neither features are shared
    # by both products, so a query built only from those separates nothing: the positive is
    # not identified and query_distance comes out 0. One differentiating attribute is
    # enough, and its polarity does not matter.
    if not (n_pos or n_neg):
        if n_unique_pos:
            n_pos = 1
        elif n_unique_neg:
            n_neg = 1
        else:
            raise UndifferentiableError(
                "both differentiating pools are empty; no query can separate this pair")

    return n_pos, n_neg, n_common, n_neither


def render_query(item, want_features, avoid_features):
    """Render the natural-language query, omitting a clause that has no features.

    Emitting an empty clause ('... and does not have: ') is what the old template did once a
    single-polarity query became possible, so each clause is conditional here.
    """
    clauses = []
    if want_features:
        clauses.append(f"{WANT_CLAUSE}: {', '.join(want_features)}")
    if avoid_features:
        clauses.append(f"{AVOID_CLAUSE}: {', '.join(avoid_features)}")
    if not clauses:
        raise ValueError(f"Query for {item!r} has no attributes on either side")
    connectives = ["that"] + ["and"] * (len(clauses) - 1)
    body = "; ".join(f"{c} {clause}" for c, clause in zip(connectives, clauses))
    return f'I am looking for: "{item}" {body}'


def parse_query(query):
    """Inverse of render_query: the (want_features, avoid_features) a canonical query states.

    Only the canonical `nl_query` rendering is accepted; a rephrased query has no fixed form
    and shares its row's nl_query anyway. Raises on anything that does not round-trip.
    """
    prefix = 'I am looking for: "'
    if not query.startswith(prefix):
        raise ValueError(f"Not a canonical query: {query!r}")
    body = query[len(prefix):]
    item, sep, rest = body.partition('" ')
    if not sep:
        raise ValueError(f"No item close quote: {query!r}")
    want, avoid = [], []
    for i, clause in enumerate(rest.split("; ")):
        connective = "that" if i == 0 else "and"
        head, sep, features = clause.partition(": ")
        if not sep:
            raise ValueError(f"Clause without features: {clause!r} in {query!r}")
        if head == f"{connective} {WANT_CLAUSE}":
            want = features.split(", ")
        elif head == f"{connective} {AVOID_CLAUSE}":
            avoid = features.split(", ")
        else:
            raise ValueError(f"Unknown clause head {head!r} in {query!r}")
    if render_query(item, want, avoid) != query:
        raise ValueError(f"Query does not round-trip: {query!r}")
    return want, avoid
