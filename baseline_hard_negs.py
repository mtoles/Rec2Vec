"""Build the metadata-matched Baseline while retaining the shared easy-negative rows.

Images: draw a training image from the positive's category, excluding every view/colorway
of each known positive garment. Text: draw from distinct Substitute/Irrelevant products
paired with the same original ESCI query and positive, in the training split. Both labels
form one pool, which must contain at least two eligible products/documents.

The original comparison negative remains eligible. No attribute labels or teacher scores
enter the draw. Validation/test rows and existing easy-negative rows are unchanged.
Outputs use _baseline-<kind>, so old _random-<kind> results cannot be reused as Baseline.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from datasets import load_from_disk

from mine_hard_negs import ID_COLUMNS, apply, columns, row_splits
from train import QUERY_COLUMNS
from utils.distance_labels import BASELINE_NEGATIVE_SOURCE, EASY_NEGATIVE_SOURCE
from utils.esci_candidates import ESCI_NEGATIVE_LABELS, candidate_pools, eligible_groups, group_key
from utils.image_split import garment_id, verify_image_splits


def draw(dataset, modality, kind, seed=42):
    anchor = QUERY_COLUMNS[kind]
    pos_id_col, neg_id_col = ID_COLUMNS[modality]
    splits = row_splits(dataset, anchor, seed)
    cols = columns(dataset, [anchor, "original_query", "split", "negative_example_source",
                             "positive_example", "negative_example", pos_id_col, neg_id_col,
                             "positive_category", "negative_category"])
    train_rows = [i for i, side in enumerate(splits) if side == "train"]
    hard_rows = [i for i in train_rows if cols["negative_example_source"][i] != EASY_NEGATIVE_SOURCE]
    if not hard_rows:
        raise ValueError("No usable training comparison rows")
    positives_of, positive_garments = defaultdict(set), defaultdict(set)
    item_id, item_category = {}, {}
    for i in train_rows:
        query = cols[anchor][i]
        positives_of[query].add(cols["positive_example"][i])
        for prefix, id_col in (("positive", pos_id_col), ("negative", neg_id_col)):
            document = cols[prefix + "_example"][i]
            item_id[document] = cols[id_col][i]
            if modality == "multimodal":
                item_category[document] = cols[prefix + "_category"][i]
        if modality == "multimodal":
            positive_garments[query].add(garment_id(cols[pos_id_col][i]))

    row_metadata, text_pools, image_pools = {}, {}, defaultdict(list)
    if modality == "text":
        unknown = {cols["negative_example_source"][i] for i in hard_rows} - ESCI_NEGATIVE_LABELS
        if unknown:
            raise ValueError(f"Construct Baseline from GOLD's ESCI pairs, not a replaced variant: {unknown}")
        for i in hard_rows:
            row_metadata[i] = {key: values[i] for key, values in cols.items()}
            row_metadata[i]["split"] = "train"
        text_pools = candidate_pools(row_metadata.values())
        eligible = eligible_groups(text_pools)
        bad = [i for i in hard_rows if group_key(row_metadata[i]) not in eligible]
        if bad:
            raise ValueError(f"{len(bad)} training comparisons lack two eligible ESCI candidates; "
                             "rebuild/filter the GOLD base with utils.esci_candidates first")
    else:
        for document, category in item_category.items():
            image_pools[category].append(document)

    rng = random.Random(seed)
    records = {}
    for i in hard_rows:
        query = cols[anchor][i]
        if modality == "text":
            products = text_pools[group_key(row_metadata[i])]
            # Uniform over distinct eligible product IDs, excluding known positive documents.
            candidates = [(pid, doc) for pid, doc in products.items()
                          if doc not in positives_of[query]]
            if len(candidates) < 2 or len({doc for _, doc in candidates}) < 2:
                raise ValueError(f"Row {i} has fewer than two candidates after excluding positives")
        else:
            category = cols["positive_category"][i]
            candidates = [(item_id[doc], doc) for doc in image_pools[category]
                          if doc not in positives_of[query]
                          and garment_id(item_id[doc]) not in positive_garments[query]]
            if not candidates:
                raise ValueError(f"No different-garment training candidate in category {category!r}")
        product_id, document = rng.choice(candidates)
        records[i] = {
            "row": i, "query": query, "positive_id": cols[pos_id_col][i],
            "labeled_negative_id": cols[neg_id_col][i],
            "labeled_source": cols["negative_example_source"][i],
            "mined_id": product_id, "mined_item": document,
            "candidate_count": len(candidates),
        }
    return records, item_id, item_category


def build(dataset, modality, kind, seed=42):
    records, item_id, item_category = draw(dataset, modality, kind, seed)
    out = apply(dataset, records, modality, item_id, item_category, source=BASELINE_NEGATIVE_SOURCE)
    if modality == "multimodal":
        verify_image_splits(out)
    report = {
        "query_kind": kind, "seed": seed, "n_replaced": len(records),
        "n_output_rows": len(out), "original_negative_eligible": True,
        "sampling": ("same_category_different_garment" if modality == "multimodal"
                     else "same_original_query_positive_substitute_or_irrelevant"),
        "n_selected_original": sum(r["mined_id"] == r["labeled_negative_id"] for r in records.values()),
        "min_candidate_count": min(r["candidate_count"] for r in records.values()),
        "original_sources": dict(Counter(r["labeled_source"] for r in records.values())),
    }
    return out, report, records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", help="GOLD dataset before negative replacement")
    parser.add_argument("--modality", choices=list(ID_COLUMNS), required=True)
    parser.add_argument("--query-kinds", nargs="+", choices=list(QUERY_COLUMNS), required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--out-root", type=Path)
    args = parser.parse_args()
    source = Path(args.dataset)
    dataset = load_from_disk(str(source))
    for kind in args.query_kinds:
        destination = (args.out_root or source.parent) / f"{source.name}_baseline-{kind}"
        if destination.exists():
            raise FileExistsError(destination)
        out, report, records = build(dataset, args.modality, kind, args.split_seed)
        out.save_to_disk(str(destination))
        report.update(input=str(source), output=str(destination))
        (destination / "baseline_report.json").write_text(json.dumps(report, indent=2) + "\n")
        with (destination / "baseline_rows.jsonl").open("w") as handle:
            for row in records.values():
                handle.write(json.dumps(row) + "\n")
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
