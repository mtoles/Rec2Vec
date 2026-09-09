"""Measure query_distance for retrieval-mined negatives.

mine_hard_negs.py swaps each train-split hard negative for the product a frozen teacher
retrieves, and leaves query_distance null: nothing measured how many of the query's
constraints that product violates, so the graded losses refuse the rows
(utils/distance_labels.to_training_labels). This pass fills the number in.

The labeled (ours) rows carry query_distance by construction: the query states
`has: [selected_pos + selected_common]` and `does not have: [selected_neg + selected_neither]`,
and the synthesized negative violates exactly the selected_pos and selected_neg ones
(preprocess_text.generate_example). For a mined product the same quantity has to be
measured. Each constraint is put to the labeling model as a yes/no question about the
product; a violated constraint is a wanted feature the product lacks or an unwanted one it
has, and query_distance is the count. Zero is a legal answer and the interesting one: a
retrieved product that satisfies the whole query is a false negative, and the graded losses
give it near-positive target mass instead of pushing it away.

Constraints are the row's selected_* columns on image rows; text rows lost those in
preprocessing, so they are parsed back out of the canonical nl_query
(utils/query_render.parse_query), the one rendering every query kind shares. Measured
distances are clipped to --max-distance, the ceiling of the constructed scale, so the
easy-negative label stays beyond every hard one; the raw count is kept in labeling_rows.

Output: <dataset>_graded/, the input with query_distance filled on every mined row (rows
whose call failed are dropped and counted), plus labeling_report.json and
labeling_rows.jsonl (per-constraint verdicts, for audit). A sibling rather than an in-place
write so the ungraded dataset's mtime, and with it paper.sh's freshness rule for the models
already trained on it, stays untouched.

    python label_mined_negs.py dataset/processed/<dataset>_mined-<kind> --modality text
    python label_mined_negs.py ... --calibrate 500   # score labeled rows, compare to construction
    python label_mined_negs.py ... --dry-run          # print the rendered prompts, no API calls
"""

import argparse
import json
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from datasets import load_from_disk
from tqdm import tqdm

from preprocess_images import DEFAULT_MODEL_ID
from utils.distance_labels import DEFAULT_MAX_DISTANCE, EASY_NEGATIVE_SOURCE, MINED_NEGATIVE_SOURCE
from utils.query_render import parse_query
from utils.retry import print_cost_report, retry_vlm_with_fallback, retry_with_fallback

ID_COLUMNS = {
    "text": "negative_id",
    "multimodal": "negative_product_id",
}

# One prompt for both modalities; only the product presentation differs. Text products are
# the description string the dataset trains on; image products are the image plus the
# catalog category, the only metadata the pool carries.
PROMPT = """### Product

{product}

### Features

{features}

For each feature, decide whether the product has it. Judge only from the product {evidence}; do not assume anything not shown or stated. Answer true if the product has the feature and false if it does not or if it cannot be determined.

Return ONLY JSON: {{"has": [true, false, ...]}} with exactly {n} booleans, one per feature, in the same order as the features above.""".strip()

EVIDENCE = {"text": "description", "multimodal": "image and category"}

# Strict shape check without parsing: a JSON object holding one key "has" whose value is a
# list of booleans. Anything else is rejected and retried.
_RESPONSE = re.compile(r'^\s*\{\s*"has"\s*:\s*\[\s*(?:(?:true|false)\s*(?:,\s*(?:true|false)\s*)*)?\]\s*\}\s*$')


def validate_response(n_features):
    def check(content):
        if not _RESPONSE.match(content):
            return False
        return len(json.loads(content)["has"]) == n_features
    return check


def render_prompt(modality, product, features):
    numbered = "\n".join(f"{i + 1}. {f}" for i, f in enumerate(features))
    return PROMPT.format(product=product, features=numbered, evidence=EVIDENCE[modality],
                         n=len(features))


def product_text(modality, row):
    if modality == "text":
        return row["negative_example"]
    return f"Category: {row['negative_category']}"


def constraints_of(row, modality):
    """(want, avoid) the row's query states.

    Image rows keep the selected_* columns the query was rendered from, so they are read
    directly. Text rows lost them in preprocessing and are parsed back out of nl_query; a
    feature that itself contains a comma ("Boxy, relaxed fit") comes back as two, which the
    constructed distance counted as one. Rare (count_query_attributes measured 5/3000) and
    accepted.
    """
    if modality == "multimodal":
        return (row["selected_pos_features"] + row["selected_common_features"],
                row["selected_neg_features"] + row["selected_neither_features"])
    return parse_query(row["nl_query"])


def ask(modality, model_id, product, image_path, features, max_retries):
    prompt = render_prompt(modality, product, features)
    if modality == "text":
        response = retry_with_fallback(
            messages=[{"role": "user", "content": prompt}],
            validation_func=validate_response(len(features)),
            model_id=model_id, max_retries=max_retries, fallback_value=None)
    else:
        response = retry_vlm_with_fallback(
            prompt=prompt, image_paths=[image_path],
            validation_func=validate_response(len(features)),
            model_id=model_id, max_retries=max_retries, fallback_value=None)
    if response is None:
        return None
    return json.loads(response)["has"]


def distance_of(want, avoid, has):
    """Constraints violated: wanted features the product lacks, unwanted ones it has."""
    n_want = len(want)
    lacks_wanted = sum(1 for h in has[:n_want] if not h)
    has_unwanted = sum(1 for h in has[n_want:] if h)
    return lacks_wanted + has_unwanted


def label_rows(dataset, modality, indices, model_id, workers, max_retries, dry_run, max_distance):
    """Ask once per distinct (product, constraints) and map the verdicts back onto rows.

    Returns {row_index: record}; record["distance"] is None when the call failed.
    "distance" is clipped to max_distance, the ceiling of the constructed scale (a labeled
    negative's distance is randint(1, max_distance)), so the easy-negative label keeps its
    meaning of "beyond every measurable distance"; "distance_raw" keeps the count.
    """
    id_col = ID_COLUMNS[modality]
    keys, key_of = {}, {}
    for i in indices:
        row = dataset[i]
        want, avoid = constraints_of(row, modality)
        key = (row[id_col], tuple(want), tuple(avoid))
        key_of[i] = key
        if key not in keys:
            keys[key] = {"product": product_text(modality, row),
                         "image_path": row["negative_example"] if modality == "multimodal" else None,
                         "want": want, "avoid": avoid}
    print(f"{len(indices):,} rows -> {len(keys):,} distinct (product, constraints) calls")

    if dry_run:
        for key, spec in list(keys.items())[:3]:
            print("-" * 70)
            print(render_prompt(modality, spec["product"], spec["want"] + spec["avoid"]))
            if spec["image_path"]:
                print(f"[image: {spec['image_path']}]")
        print("-" * 70)
        print("dry run: no API calls made")
        return {}

    def work(key):
        spec = keys[key]
        has = ask(modality, model_id, spec["product"], spec["image_path"],
                  spec["want"] + spec["avoid"], max_retries)
        return key, has

    verdicts = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for key, has in tqdm(pool.map(work, keys), total=len(keys), desc="Labeling"):
            verdicts[key] = has

    records = {}
    for i in indices:
        key = key_of[i]
        has = verdicts[key]
        want, avoid = list(key[1]), list(key[2])
        raw = distance_of(want, avoid, has) if has is not None else None
        records[i] = {
            "row": i,
            "negative_id": key[0],
            "want": want,
            "avoid": avoid,
            "has": has,
            "distance_raw": raw,
            "distance": min(raw, max_distance) if raw is not None else None,
        }
    return records


def apply(dataset, records, model_id):
    """Fill query_distance on the labeled mined rows; drop the ones whose call failed."""
    def fill(batch, indices):
        batch = {k: list(v) for k, v in batch.items()}
        for r, idx in enumerate(indices):
            if idx not in records:
                continue
            batch["query_distance"][r] = records[idx]["distance"]
            if "distance_source" in batch:
                batch["distance_source"][r] = f"llm:{model_id}"
        return batch

    failed = {i for i, rec in records.items() if rec["distance"] is None}
    out = dataset.map(fill, with_indices=True, batched=True, features=dataset.features,
                      desc="Filling query_distance")
    out = out.filter(lambda _, indices: [i not in failed for i in indices], with_indices=True,
                     batched=True, desc="Dropping unlabeled rows")
    return out


def report(records, model_id, args):
    recs = list(records.values())
    labeled = [r for r in recs if r["distance"] is not None]
    distances = Counter(r["distance"] for r in labeled)
    n_constraints = Counter(len(r["want"]) + len(r["avoid"]) for r in recs)
    out = {
        "model": model_id,
        "n_mined_rows": len(recs),
        "n_labeled": len(labeled),
        "n_failed": len(recs) - len(labeled),
        "max_distance": args.max_distance,
        "n_clipped_to_max_distance": sum(1 for r in labeled if r["distance_raw"] > args.max_distance),
        "n_distance_zero": distances[0],
        "mean_distance": sum(r["distance"] for r in labeled) / max(len(labeled), 1),
        "distance_hist": {str(k): v for k, v in sorted(distances.items())},
        "n_constraints_hist": {str(k): v for k, v in sorted(n_constraints.items())},
        "mean_fraction_violated": sum(r["distance"] / (len(r["want"]) + len(r["avoid"]))
                                      for r in labeled) / max(len(labeled), 1),
    }
    print(json.dumps(out, indent=2))
    return out


def calibrate(dataset, modality, n, model_id, workers, max_retries, dry_run, seed):
    """Score n labeled (ours) hard rows and compare the measured distance to the constructed one.

    The labeled negative's distance is known by construction, so this is the only ground
    truth the labeling prompt can be checked against before it is trusted on mined rows.
    """
    import random
    labeled = [i for i, s in enumerate(dataset["negative_example_source"])
               if s not in (EASY_NEGATIVE_SOURCE, MINED_NEGATIVE_SOURCE)]
    indices = sorted(random.Random(seed).sample(labeled, min(n, len(labeled))))
    records = label_rows(dataset, modality, indices, model_id, workers, max_retries, dry_run,
                         DEFAULT_MAX_DISTANCE)
    if dry_run:
        return
    truth = dataset.select(indices)["query_distance"]
    pairs = [(t, records[i]["distance"]) for i, t in zip(indices, truth)
             if records[i]["distance"] is not None]
    exact = sum(1 for t, m in pairs if t == m)
    within1 = sum(1 for t, m in pairs if abs(t - m) <= 1)
    mae = sum(abs(t - m) for t, m in pairs) / max(len(pairs), 1)
    bias = sum(m - t for t, m in pairs) / max(len(pairs), 1)
    print(f"calibration on {len(pairs)} labeled rows ({len(indices) - len(pairs)} failed): "
          f"exact {exact / len(pairs):.3f} | within 1 {within1 / len(pairs):.3f} | "
          f"MAE {mae:.3f} | bias (measured - constructed) {bias:+.3f}")
    confusion = Counter((t, m) for t, m in pairs)
    print("constructed -> measured (count):")
    for (t, m), c in sorted(confusion.items()):
        print(f"  {t} -> {m}: {c}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="a _mined-<kind> dataset directory from mine_hard_negs.py")
    ap.add_argument("--modality", choices=["text", "multimodal"], required=True)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--workers", type=int, default=int(os.getenv("MAX_WORKERS", "32")))
    ap.add_argument("--max-retries", type=int, default=3)
    ap.add_argument("--max-distance", type=int, default=DEFAULT_MAX_DISTANCE,
                    help="clip measured distances here, the ceiling of the constructed scale")
    ap.add_argument("--limit", type=int, default=None, help="label only the first N mined rows")
    ap.add_argument("--calibrate", type=int, default=None,
                    help="score N labeled rows against their constructed distance instead")
    ap.add_argument("--seed", type=int, default=42, help="row sample for --calibrate")
    ap.add_argument("--dry-run", action="store_true", help="print the rendered prompts and exit")
    ap.add_argument("--out-root", default=None, help="write outputs here instead of beside the input")
    args = ap.parse_args()

    dataset = load_from_disk(args.dataset)
    print(f"{args.dataset}: {len(dataset):,} rows")

    if args.calibrate is not None:
        calibrate(dataset, args.modality, args.calibrate, args.model_id, args.workers,
                  args.max_retries, args.dry_run, args.seed)
        print_cost_report()
        return

    mined = [i for i, s in enumerate(dataset["negative_example_source"]) if s == MINED_NEGATIVE_SOURCE]
    if args.limit is not None:
        mined = mined[:args.limit]
    records = label_rows(dataset, args.modality, mined, args.model_id, args.workers,
                         args.max_retries, args.dry_run, args.max_distance)
    if args.dry_run:
        return

    base = os.path.basename(args.dataset.rstrip("/"))
    root = args.out_root or os.path.dirname(args.dataset.rstrip("/"))
    out_dir = os.path.join(root, f"{base}_graded")
    out = apply(dataset, records, args.model_id)
    out.save_to_disk(out_dir)
    summary = report(records, args.model_id, args)
    summary["input"] = args.dataset
    summary["output"] = out_dir
    summary["n_output_rows"] = len(out)
    with open(os.path.join(out_dir, "labeling_report.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(out_dir, "labeling_rows.jsonl"), "w") as f:
        for rec in records.values():
            f.write(json.dumps(rec) + "\n")
    print(f"saved {len(out):,} rows to {out_dir}")
    print_cost_report()


if __name__ == "__main__":
    main()
