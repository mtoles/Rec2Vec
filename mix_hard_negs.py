"""Mix labeled (ours) and retrieval-mined hard negatives in one training set.

Two processed datasets in, one out. The labeled dataset carries our synthesized hard
negative with its measured query_distance on every row; its mine_hard_negs.py sibling
carries the retrieval-mined negative (query_distance null) on every train-split hard row
and is otherwise row-for-row identical. The output takes a seeded random `--mined-fraction`
of those train-split hard rows from the mined sibling and every other row from the labeled
dataset, so half the training negatives are graded and half are mined and ungraded.
Validation and test rows are untouched, so a model trained on the output is evaluated on
exactly the corpus and queries of the input.

Which rows are eligible is read off the mined sibling: every row whose
negative_example_source is MINED_NEGATIVE_SOURCE. Rows the miner left alone (easy rows,
val/test rows, rows train.py filters out) stay labeled.

Output: <labeled>_mixed-<kind>[_<variant>]/, the mined sibling's name with `mined`
replaced by `mixed`, plus mixing_report.json and mixing_rows.jsonl (row index, source).

    python mix_hard_negs.py dataset/processed/<dataset> dataset/processed/<dataset>_mined-<kind>_<variant>
"""

import argparse
import json
import os

import numpy as np
from datasets import load_from_disk

from mine_hard_negs import columns
from utils.distance_labels import MINED_NEGATIVE_SOURCE, SOURCE_COLUMN


def output_dir(labeled_dir, mined_dir, out_root):
    labeled = os.path.basename(labeled_dir.rstrip("/"))
    mined = os.path.basename(mined_dir.rstrip("/"))
    prefix = f"{labeled}_mined-"
    if not mined.startswith(prefix):
        raise ValueError(f"{mined!r} is not a mine_hard_negs.py sibling of {labeled!r} "
                         f"(expected the prefix {prefix!r})")
    root = out_root or os.path.dirname(labeled_dir.rstrip("/"))
    return os.path.join(root, f"{labeled}_mixed-{mined[len(prefix):]}")


def check_aligned(labeled, mined, anchor_columns):
    if len(labeled) != len(mined):
        raise ValueError(f"row counts differ: labeled {len(labeled):,}, mined {len(mined):,}; "
                         "the mined sibling dropped rows (mine without --fallback none)")
    if labeled.column_names != mined.column_names:
        raise ValueError(f"columns differ: {labeled.column_names} vs {mined.column_names}")
    a = columns(labeled, anchor_columns)
    b = columns(mined, anchor_columns)
    for name in anchor_columns:
        if a[name] != b[name]:
            raise ValueError(f"column {name!r} differs between the two datasets; rows are not aligned")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labeled", help="processed dataset with our labeled negatives")
    ap.add_argument("mined", help="its mine_hard_negs.py sibling, <labeled>_mined-<kind>[_<variant>]")
    ap.add_argument("--mined-fraction", type=float, default=0.5,
                    help="fraction of the mined sibling's mined rows taken from it; the rest stay labeled")
    ap.add_argument("--seed", type=int, default=42, help="which rows are taken from the mined sibling")
    ap.add_argument("--out-root", default=None, help="write the output here instead of beside the input")
    args = ap.parse_args()
    if not 0 < args.mined_fraction < 1:
        raise ValueError(f"--mined-fraction must be in (0, 1), got {args.mined_fraction}")

    out_dir = output_dir(args.labeled, args.mined, args.out_root)
    if os.path.exists(out_dir):
        raise FileExistsError(f"{out_dir} exists; move it aside first")

    labeled = load_from_disk(args.labeled)
    mined = load_from_disk(args.mined)
    print(f"labeled {args.labeled}: {len(labeled):,} rows")
    print(f"mined   {args.mined}: {len(mined):,} rows")
    check_aligned(labeled, mined, ["nl_query", "positive_example"])

    # Arrow -> Python keeps a null query_distance null; a pandas round trip would make it NaN.
    mined_cols = mined.data.to_pydict()
    eligible = np.flatnonzero(np.array(mined_cols[SOURCE_COLUMN]) == MINED_NEGATIVE_SOURCE)
    n_take = int(round(len(eligible) * args.mined_fraction))
    rng = np.random.default_rng(args.seed)
    take = set(rng.choice(eligible, size=n_take, replace=False).tolist())
    print(f"{len(eligible):,} mined rows in the sibling; taking {n_take:,} ({args.mined_fraction:g}) from it")

    def swap(batch, indices):
        batch = {k: list(v) for k, v in batch.items()}
        for r, idx in enumerate(indices):
            if idx not in take:
                continue
            for name in batch:
                batch[name][r] = mined_cols[name][idx]
        return batch

    out = labeled.map(swap, with_indices=True, batched=True, features=labeled.features,
                      desc="Mixing negatives")
    out.save_to_disk(out_dir)

    sources = out.to_pandas()[SOURCE_COLUMN].value_counts().to_dict()
    report = {
        "labeled": args.labeled,
        "mined": args.mined,
        "output": out_dir,
        "mined_fraction": args.mined_fraction,
        "seed": args.seed,
        "n_rows": len(out),
        "n_eligible": int(len(eligible)),
        "n_taken_from_mined": n_take,
        "sources": {str(k): int(v) for k, v in sources.items()},
    }
    with open(os.path.join(out_dir, "mixing_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(out_dir, "mixing_rows.jsonl"), "w") as f:
        for idx in eligible.tolist():
            f.write(json.dumps({"row": idx, "source": "mined" if idx in take else "labeled"}) + "\n")
    print(f"saved {len(out):,} rows to {out_dir}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
