"""Filter an existing GOLD text dataset to two-candidate ESCI pools without API calls.

Example:
  python -m utils.build_candidate_text dataset/processed/<old>_rephrased-in-context \
    --out dataset/processed/<old>_candidates2_rephrased-in-context \
    --base-out dataset/processed/<old>_candidates2

Run this once on the shared GOLD base, then build Baseline/mined variants from the result.
Existing predictions belong to the old dataset and must not be renamed to this condition.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

from datasets import load_from_disk

from utils.esci_candidates import filter_processed_dataset
from utils.leakage_split import verify


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-out", type=Path, help="also save the shared base under this path")
    args = parser.parse_args()
    destinations = [args.out] + ([args.base_out] if args.base_out else [])
    for destination in destinations:
        if destination.exists():
            raise FileExistsError(destination)
    dataset = load_from_disk(str(args.dataset))
    out, report = filter_processed_dataset(dataset)
    report["input"] = str(args.dataset)
    report["rows_by_split"] = dict(Counter(out["split"]))
    report["comparison_sources"] = dict(Counter(s for s in out["negative_example_source"] if s != "random"))
    report["split_query_and_product_counts"] = verify(out.select_columns(
        ["split", "original_query", "positive_id", "negative_id"]))
    for destination in destinations:
        out.save_to_disk(str(destination))
        (destination / "candidate_filter_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
