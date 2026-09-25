"""Validate image dataset separation across queries, files, and garment designs."""

import re
import unicodedata
from collections import defaultdict


def garment_id(product_id):
    """One design identity across image views, SKUs, and colorways."""
    product_id = re.sub(r"_\d+_(front|side|back|full|flat|additional)$", "", str(product_id))
    return re.sub(r"_\d+$", "", product_id)


def query_key(query):
    return " ".join(unicodedata.normalize("NFKC", query).casefold().split())


def verify_image_splits(dataset):
    if "split" not in dataset.column_names:
        raise ValueError("Image datasets require a precomputed document-disjoint split")
    columns = [name for name in (
        "split", "positive_example", "negative_example", "positive_product_id",
        "negative_product_id", "nl_query", "rephrased_query", "human_query", "anchor",
    ) if name in dataset.column_names]
    owners = {kind: {} for kind in ("query", "image", "garment")}
    counts = defaultdict(int)
    for row in dataset.select_columns(columns):
        side = row["split"]
        if side not in ("train", "validation", "test"):
            raise ValueError(f"Invalid image split: {side!r}")
        counts[side] += 1
        values = {
            "query": [query_key(row[c]) for c in ("nl_query", "rephrased_query", "human_query", "anchor")
                      if c in row and row[c]],
            "image": [row[c] for c in ("positive_example", "negative_example") if c in row],
            "garment": [garment_id(row[c]) for c in ("positive_product_id", "negative_product_id") if c in row],
        }
        for kind, items in values.items():
            for value in items:
                previous = owners[kind].setdefault(value, side)
                if previous != side:
                    raise ValueError(f"Cross-split {kind} overlap: {previous}/{side}: {value!r}")
    return dict(counts)


def active_image_base():
    from pathlib import Path
    from utils.training_profile import training_profile
    root = Path(__file__).resolve().parents[1]
    profile = training_profile()
    if profile:
        return Path(profile['bases']['multimodal'])
    match = re.search(r'IMG_DATASET=\$\{IMG_DATASET:-([^}]+)\}', (root / 'paper.sh').read_text())
    if match is None:
        raise ValueError('paper.sh does not declare an image dataset')
    return root / match[1]
