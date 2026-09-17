"""Download the human-study annotations and write them as processed datasets.

The two workbooks are the ones handed to annotators by `human_data_text.py` and
`human_data_image.py`. Each has one tab per annotator; the Q3/Q4 columns hold the queries a
person wrote for that product pair. This script pulls both workbooks, resolves every
annotated row back to the dataset row it was built from, and writes one processed dataset per
modality:

    dataset/processed/feature-distance-dataset_gemini-2.5-flash_1000000_nolek_human
    dataset/processed/deepfashion-inshop-image-triplets_hf_20000_human

Each output is a row-subset of the corresponding base processed dataset -- minus the
hard-negative columns -- plus the human columns (`human_query`, `human_query_alt`,
`human_pos_attributes`, `human_neg_attributes`, `annotator`). `human_query` is the column the
`human` query kind reads in test.py, mirroring `nl_query` for synthetic and `rephrased_query`
for rephrased.

The pair's hard negative is dropped: an annotator writes a free-form query, and nothing
establishes that the product the synthetic pipeline picked as that pair's negative fails the
query the annotator actually wrote. Its `query_distance` counts violations of the synthetic
query, not of the human one. Carrying either column would invite scoring a human query against
an unverified negative, so the columns are removed rather than left for a caller to skip.

Every output row is `split == "test"`: the human set is evaluation-only, so test.py's split
logic keeps all of it and train.py never sees it (train.py has no `human` query kind).

Rows are dropped when `Q3` is blank (nobody wrote a query) or when `Problem?` is non-blank
(the annotator flagged the pair as unusable).

Resolving a sheet row back to a dataset row: the workbooks carry rendered product text, not
ids. Both generators are deterministic given their seed, so this script rebuilds the same
pool and keys it by the rendered (Product 1, Product 2) text. That is exact rather than fuzzy,
and it survives annotators reordering rows or tabs -- which two of them did. The
"SHOPPING FOR: ..." header line of Product 1 is not part of the key: one annotator corrected
a typo in it, and the product body below it is unique per pool row anyway (the generators
never show a product twice).

Usage:
    python human_study/download_human_labels.py
    python human_study/download_human_labels.py --out-suffix _human_pilot
"""

import argparse
import io
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd
from datasets import load_from_disk

import human_data_image
import human_data_text

REPO_ROOT = Path(__file__).resolve().parent.parent
# The generators sit next to this file and are importable directly; utils/ lives at the repo
# root, which is not on the path when this script is run as human_study/<name>.py.
sys.path.insert(0, str(REPO_ROOT))

from utils.distance_labels import EASY_NEGATIVE_SOURCE  # noqa: E402

EXPORT_URL = "https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=xlsx"

# Hard-coded on purpose. These are the headers the workbooks have today; if a column is
# renamed, added, removed or reordered, every downstream assumption about which cell holds
# which answer is void, so the run stops instead of silently mapping the wrong column.
TEXT_HEADERS = ["Product 1", "Product 2", "n_1", "Q1", "n_2", "Q2", "Q3", "Q4", "Problem?"]
IMAGE_HEADERS = ["Product 1", "Product 1 Image", "Product 2", "Product 2 Image",
                 "n_1", "Q1", "n_2", "Q2", "Q3", "Q4", "Problem?"]

# The generator defaults the workbooks were built with. The pool is only reproducible at
# these exact values, so they are pinned here rather than re-derived.
SHEETS = 10
PER_SHEET = 25
SEED = 0
TEXT_MIN_CHARS = 250
TEXT_MAX_CHARS = 2500
IMAGE_CATEGORY_CAP = 6


def blank(value):
    return pd.isna(value) or not str(value).strip()


def product_body(cell):
    """The rendered product text of a sheet cell without its 'SHOPPING FOR:' header line."""
    text = str(cell).strip()
    if text.startswith("SHOPPING FOR:"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
    return text.strip()


def fetch_workbook(sheet_id):
    """Every tab of the sheet, as {tab_name: DataFrame}.

    The xlsx export is what carries all tabs; the csv export returns only the first one,
    which in both workbooks is the unannotated 'Annotator 1' template.
    """
    url = EXPORT_URL.format(sheet_id=sheet_id)
    with urllib.request.urlopen(url) as response:
        payload = response.read()
    tabs = pd.read_excel(io.BytesIO(payload), sheet_name=None)
    print(f"  fetched {len(tabs)} tabs ({len(payload):,} bytes) from {sheet_id}")
    return tabs


def check_headers(tabs, expected, sheet_id):
    for tab_name, frame in tabs.items():
        found = list(frame.columns)
        if found != expected:
            raise ValueError(
                f"Sheet {sheet_id} tab {tab_name!r} has unexpected headers.\n"
                f"  expected: {expected}\n"
                f"  found:    {found}\n"
                "The column layout changed. Update TEXT_HEADERS/IMAGE_HEADERS here and "
                "re-check which column each answer now lives in before re-running."
            )


def annotated_rows(tabs):
    """(tab_name, row) for rows a person actually filled in and did not flag."""
    kept, no_query, flagged = [], 0, 0
    for tab_name, frame in tabs.items():
        for _, row in frame.iterrows():
            if blank(row["Q3"]):
                no_query += 1
                continue
            if not blank(row["Problem?"]):
                flagged += 1
                continue
            kept.append((tab_name, row))
    print(f"  {len(kept)} annotated rows kept | {no_query} without a query | "
          f"{flagged} flagged in 'Problem?'")
    return kept


def text_pool():
    """{(product 1 text, product 2 text): raw row} for the text workbook."""
    rows = human_data_text.load_rows(str(human_data_text.RAW_DEFAULT),
                                     SHEETS * PER_SHEET, SEED, TEXT_MIN_CHARS, TEXT_MAX_CHARS)
    pool = {}
    for row in rows:
        key = (human_data_text.format_product(row["positive_product"]).strip(),
               human_data_text.format_product(row["hard_neg_product"]).strip())
        pool[key] = (row["positive_product"]["product_id"],
                     row["hard_neg_product"]["product_id"], row["item"])
    return pool


def image_pool():
    """{(product 1 text, product 2 text): raw row} for the image workbook."""
    rows = human_data_image.load_pool(str(human_data_image.RAW_DEFAULT),
                                      SHEETS * PER_SHEET, SEED)
    pool = {}
    for row in rows:
        key = (human_data_image.describe(row["positive_product"]).strip(),
               human_data_image.describe(row["hard_neg_product"]).strip())
        pool[key] = (row["positive_product_id"], row["hard_negative_product_id"],
                     row["item"])
    return pool


def base_index(base, positive_column, negative_column):
    """{(positive id, negative id, item): [row index]} over the base's hard negatives.

    Easy negatives are excluded: the study only ever showed the hard pair.

    The product pair alone is not unique -- the same two products recur under near-duplicate
    ESCI queries ("honda lawn mower" / "honda lawnmower") -- so `item` is part of the key.
    nl_query is deliberately *not*: the study workbooks were built from the raw
    `_fixed_distance` jsonl, whose per-row feature sampling differs from the `_nolek`
    processed dataset, so the two disagree on query text for the same pair.

    Values are lists rather than single indices because a handful of (pair, item) keys still
    collide across the full base. That is only a problem if a key the study actually needs is
    ambiguous, which `build` checks at lookup time.
    """
    index = {}
    positives = base[positive_column]
    negatives = base[negative_column]
    items = base["item"]
    sources = base["negative_example_source"]
    for i, (pos, neg, item, source) in enumerate(zip(positives, negatives, items, sources)):
        if source == EASY_NEGATIVE_SOURCE:
            continue
        index.setdefault((pos, neg, item), []).append(i)
    return index


IN_CONTEXT_SUFFIX = "-in-context"


def in_context_examples(out, n_examples):
    """Row indices of the first kept row of each of the first n_examples annotators.

    Rows are in workbook order (tab, then sheet row), so the first row of an annotator's
    block is their first non-problematic pair. These rows are the style examples in
    rephrase_dataset.py --in-context and are held out of the *-in-context human set so
    that models trained on in-context rephrasings are not scored on the queries their
    rephraser saw."""
    first = {}
    for i, annotator in enumerate(out["annotator"]):
        if annotator not in first:
            first[annotator] = i
    annotators = list(first)[:n_examples]
    if len(annotators) < n_examples:
        raise ValueError(f"only {len(annotators)} annotators, cannot take {n_examples} examples")
    return [first[a] for a in annotators]


def build(name, sheet_id, headers, base_path, pool_fn, positive_column, negative_column,
          out_suffix, n_examples):
    print(f"\n== {name}")
    tabs = fetch_workbook(sheet_id)
    check_headers(tabs, headers, sheet_id)
    kept = annotated_rows(tabs)

    pool = pool_fn()
    base = load_from_disk(str(base_path))
    index = base_index(base, positive_column, negative_column)
    print(f"  base dataset {base_path.name}: {len(base)} rows, "
          f"{len(index)} hard pairs")

    selected, extras, unmatched = [], [], []
    for tab_name, row in kept:
        key = (product_body(row["Product 1"]), product_body(row["Product 2"]))
        if key not in pool:
            raise ValueError(
                f"Tab {tab_name!r} has a product pair that is not in the regenerated pool. "
                "The workbook was not built from the pinned generator defaults, or the raw "
                "jsonl changed; the row cannot be resolved to a dataset row."
            )
        ids = pool[key]
        if ids not in index:
            # The workbooks were sampled from the raw jsonl; the processed base is a
            # leak-filtered subset of it, so some study pairs are simply not in the base.
            # Expected, but never silent -- the count is reported below.
            unmatched.append((tab_name, ids))
            continue
        candidates = index[ids]
        if len(candidates) != 1:
            raise ValueError(f"Pair {ids} from tab {tab_name!r} matches {len(candidates)} "
                             f"hard-negative rows of {base_path.name}; the human query "
                             "cannot be attached to one row unambiguously.")
        selected.append(candidates[0])
        extras.append({
            "human_query": str(row["Q3"]).strip(),
            "human_query_alt": "" if blank(row["Q4"]) else str(row["Q4"]).strip(),
            "human_pos_attributes": "" if blank(row["Q1"]) else str(row["Q1"]).strip(),
            "human_neg_attributes": "" if blank(row["Q2"]) else str(row["Q2"]).strip(),
            "annotator": tab_name,
        })

    if unmatched:
        print(f"  {len(unmatched)} annotated rows dropped: pair absent from "
              f"{base_path.name} (leak-filtered out of the base). First few: "
              f"{[ids for _, ids in unmatched[:3]]}")
    if not selected:
        raise ValueError(f"No annotated row of {name} could be matched to {base_path.name}.")

    out = base.select(selected).flatten_indices()
    for column in extras[0]:
        out = out.add_column(column, [e[column] for e in extras])
    # The hard negative is not a verified negative for the query the annotator wrote (see the
    # module docstring), so drop it and everything derived from it.
    unverified = [c for c in (negative_column, "negative_example", "negative_example_source",
                              "negative_category", "query_distance", "distance_source")
                  if c in out.column_names]
    out = out.remove_columns(unverified)
    print(f"  dropped hard-negative columns: {', '.join(unverified)}")
    # Evaluation-only: every row is test, whatever split the base assigned the pair.
    if "split" in out.column_names:
        out = out.remove_columns(["split"])
    out = out.add_column("split", ["test"] * len(out))

    out_path = REPO_ROOT / "dataset/processed" / f"{base_path.name}{out_suffix}"
    out.save_to_disk(str(out_path))
    print(f"  wrote {out_path}: {len(out)} rows, {len(set(out['annotator']))} annotators")

    # The in-context variant: the same set minus the rows used as style examples.
    examples = in_context_examples(out, n_examples)
    example_rows = [{"annotator": out[i]["annotator"], "human_query": out[i]["human_query"],
                     "nl_query": out[i]["nl_query"], positive_column: out[i][positive_column]}
                    for i in examples]
    examples_path = Path(__file__).resolve().parent / f"in_context_examples_{name}.json"
    with open(examples_path, "w") as fh:
        json.dump(example_rows, fh, indent=2)
    rest = out.select([i for i in range(len(out)) if i not in set(examples)]).flatten_indices()
    in_context_path = REPO_ROOT / "dataset/processed" / f"{base_path.name}{out_suffix}{IN_CONTEXT_SUFFIX}"
    rest.save_to_disk(str(in_context_path))
    print(f"  wrote {examples_path}: {len(example_rows)} style examples "
          f"({', '.join(r['annotator'] for r in example_rows)})")
    print(f"  wrote {in_context_path}: {len(rest)} rows (examples held out)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-suffix", default="_human",
                    help="suffix appended to each base dataset name")
    ap.add_argument("--in-context-examples", type=int, default=5,
                    help="first kept row of this many annotators become the rephrasing style "
                         "examples and are held out of the <suffix>-in-context set")
    args = ap.parse_args()

    build("text", "1DRYeAECYlWF2SGniw7vbGj873Hiz6VDEUnystnpjA8M", TEXT_HEADERS,
          REPO_ROOT / "dataset/processed/feature-distance-dataset_gemini-2.5-flash_1000000_nolek",
          text_pool, "positive_id", "negative_id", args.out_suffix, args.in_context_examples)

    build("image", "18Ok_AmDiqwMgU5mGXslA-XN3dUNqT_nae4M3XlDbpYw", IMAGE_HEADERS,
          REPO_ROOT / "dataset/processed/deepfashion-inshop-image-triplets_hf_20000",
          image_pool, "positive_product_id", "negative_product_id", args.out_suffix,
          args.in_context_examples)


if __name__ == "__main__":
    main()
