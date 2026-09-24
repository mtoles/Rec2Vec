"""Canonical mapping from raw constraint-violation counts to MSE training labels.

`query_distance` answers one question: how many of the query's constraints does this
negative violate? For a *random* negative that count was never measured -- it is absent,
not small -- so it must never be encoded as a number in the same column.

Both preprocessors used to encode it as one anyway, and disagreed: `preprocess_text.py`
wrote -1, `preprocess_images.py` wrote `max_distance * 2` (= 20). The train-time remap keyed
on -1, so it fired on text (relabelling to 15) and silently no-opped on images, which then
trained toward 20/V. Three different targets, none of them chosen.

The rule here is: **easy negatives are identified by `negative_example_source`, never by
their number.** Anything that disagrees raises instead of being quietly absorbed.
"""

from utils.distance_transform import transform_normalized_distance

EASY_NEGATIVE_SOURCE = "random"
# A retrieval-mined negative (mine_hard_negs.py). Its distance to the query was never
# measured, so `query_distance` is null on these rows until a labeling pass fills it in.
MINED_NEGATIVE_SOURCE = "mined"
# The comparison slot must remain distinct from its random easy twin: CoSENT keeps
# a positive pair from this slot. Its constraint-violation count is unmeasured.
BASELINE_NEGATIVE_SOURCE = "baseline"
UNMEASURED_NEGATIVE_SOURCES = frozenset({MINED_NEGATIVE_SOURCE, BASELINE_NEGATIVE_SOURCE})

# The --max-distance both preprocessors are invoked with. Only used to derive the default
# easy-negative distance; to_training_labels re-checks it against the data every run, so a
# drift here surfaces as an error rather than a silently wrong label.
DEFAULT_MAX_DISTANCE = 10
SOURCE_COLUMN = "negative_example_source"

# Historical markers, accepted only by the migration script. Training never looks at these.
LEGACY_EASY_SENTINELS = (-1.0, 20.0)


def default_easy_negative_distance(max_distance: int) -> float:
    """Easy negatives sit beyond every measurable distance.

    A hard negative's distance is `randint(1, max_distance)`, so `max_distance` is the
    ceiling of the measured scale and twice it is unambiguously outside. Derived rather
    than hardcoded so it cannot drift if `max_distance` changes.
    """
    return 2.0 * max_distance


def to_training_labels(
    dataset,
    V: float,
    easy_negative_distance: float,
    transform: str,
    transform_alpha: float = 5.0,
    distance_column: str = "query_distance",
    label_column: str = "label",
    unmeasured_as_easy: bool = False,
):
    """Rewrite `distance_column` into a `[0, 1]` training label, and report what it did.

    Returns `(dataset, stats)`. Raises when the data cannot support the mapping:
      - `negative_example_source` missing -- easy negatives would be unidentifiable
      - a hard negative with a null or negative distance -- a leftover sentinel
      - `easy_negative_distance` not strictly above every hard distance -- an easy negative
        would be indistinguishable from, or nearer than, a measured one

    `unmeasured_as_easy` admits mined or metadata-sampled Baseline negatives whose
    distance is null and labels them at the easy label. That is only correct for a loss that
    targets every non-positive at easy_label regardless of the label column (mse-mined);
    a graded loss must leave it False so unmeasured rows raise.
    """
    if SOURCE_COLUMN not in dataset.column_names:
        raise ValueError(
            f"Dataset has no '{SOURCE_COLUMN}' column, so easy negatives cannot be identified. "
            "Rebuild it with the current preprocessor, or migrate it with "
            "utils/migrate_easy_negative_labels.py.")
    if distance_column not in dataset.column_names:
        raise ValueError(f"Dataset has no '{distance_column}' column")

    sources = dataset[SOURCE_COLUMN]
    distances = dataset[distance_column]
    is_easy = [s == EASY_NEGATIVE_SOURCE for s in sources]
    is_unmeasured = [s in UNMEASURED_NEGATIVE_SOURCES and d is None for s, d in zip(sources, distances)]

    n_easy = sum(is_easy)
    n_unmeasured = sum(is_unmeasured)
    n_hard = len(is_easy) - n_easy - n_unmeasured
    if n_unmeasured and not unmeasured_as_easy:
        raise ValueError(
            f"{n_unmeasured} mined/baseline negatives have no measured {distance_column}. A graded "
            "loss cannot train on them; label them first, or use an ungraded style.")

    hard_distances = [d for d, easy, unmeasured in zip(distances, is_easy, is_unmeasured)
                      if not easy and not unmeasured]
    bad = [d for d in hard_distances if d is None or d < 0]
    if bad:
        raise ValueError(
            f"{len(bad)} hard negatives carry a null or negative {distance_column} "
            f"(e.g. {bad[:5]}). A negative distance is a leftover sentinel, not a count.")
    if not hard_distances and not (unmeasured_as_easy and n_unmeasured):
        raise ValueError(f"No measured or permitted unmeasured comparison negatives found")

    max_hard = max(hard_distances, default=0.0)
    if easy_negative_distance < max_hard:
        raise ValueError(
            f"easy_negative_distance={easy_negative_distance} sits below the largest "
            f"measured hard-negative distance ({max_hard}). Easy negatives must not be "
            "nearer than a measured negative.")
    # Equality is permitted so the easy sweep can reach the bottom of the measured scale
    # (easy=10 with distances 1..10). The easy label then equals the label of the most
    # distant hard negatives, so a loss that recovers "is random" by comparing a label to
    # easy_label can no longer tell them apart. Losses that only FILL cross-row cells with
    # easy_label are unaffected; GradedInfoNCELoss compares, and train.py refuses the
    # combination rather than training on silently mislabeled rows.
    easy_collides = easy_negative_distance == max_hard

    def to_label(row):
        unmeasured = row[SOURCE_COLUMN] in UNMEASURED_NEGATIVE_SOURCES and row[distance_column] is None
        raw = (easy_negative_distance if row[SOURCE_COLUMN] == EASY_NEGATIVE_SOURCE or unmeasured
               else row[distance_column])
        return {label_column: transform_normalized_distance(raw / V, transform, transform_alpha)}

    dataset = dataset.map(to_label, desc="Building distance labels")
    if distance_column != label_column and distance_column in dataset.column_names:
        dataset = dataset.remove_columns([distance_column])

    stats = {
        "n_easy": n_easy,
        "n_hard": n_hard,
        "n_unmeasured": n_unmeasured,
        "max_hard_distance": max_hard,
        "easy_collides": easy_collides,
        "easy_negative_distance": easy_negative_distance,
        "easy_label": transform_normalized_distance(
            easy_negative_distance / V, transform, transform_alpha),
        "V": V,
        "transform": transform,
    }
    print(f"Labels: {n_hard:,} hard (distance 1..{max_hard:g}) + {n_easy:,} easy "
          f"(distance {easy_negative_distance:g} -> label {stats['easy_label']:.4f})"
          + (f" + {n_unmeasured:,} unmeasured mined/baseline at the easy label" if n_unmeasured else "")
          + f", V={V}")
    return dataset, stats
