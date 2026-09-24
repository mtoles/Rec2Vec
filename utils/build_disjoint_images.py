"""Rebuild image pairs after assigning documents and garment designs to splits."""

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from datasets import load_from_disk
from PIL import Image, ImageOps

from preprocess_images import (
    generate_visual_distance_example, get_visual_common_and_differentiating_features,
    make_raw_example, save_processed_dataset,
)
from rephrase_dataset import product_description, rephrase_query
from utils.image_split import garment_id, query_key, verify_image_splits

OLD_BASE = Path("dataset/processed/deepfashion-inshop-image-triplets_hf_20000")
NEW_BASE = Path(str(OLD_BASE) + "_disjoint")
RAW = Path("dataset/deepfashion-inshop-image-triplets_hf_20000_repaired_scored.jsonl")
WORK = Path("analysis/migrations/image_disjoint_20260921")
FEATURES = (
    "selected_pos_features", "selected_neg_features", "selected_common_features", "selected_neither_features",
    "full_common_features", "full_unique_pos_features", "full_unique_neg_features", "full_neither_features",
)


def digest_image(path):
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        payload = str(image.size).encode() + image.tobytes()
    return path, hashlib.sha256(payload).hexdigest()


def load_inputs(workers):
    raw = [json.loads(line) for line in RAW.open()]
    old = load_from_disk(str(OLD_BASE) + "_rephrased-in-context")
    hard = {r["positive_product_id"]: r for r in old if r["negative_example_source"] != "random"}
    assert len(hard) == len(raw)
    records = {}
    for row in raw:
        prior = hard[row["positive_product_id"]]
        assert prior["negative_product_id"] == row["hard_negative_product_id"]
        for key in FEATURES + ("nl_query", "rephrased_query", "query_distance"):
            row[key] = prior[key]
        for role in ("positive_product", "hard_neg_product", "easy_neg_product"):
            record = row[role]
            records[record["image_path"]] = record
    humans = load_from_disk(str(OLD_BASE) + "_human-combined-in-context")
    examples = json.loads(Path("human_study/in_context_examples_image.json").read_text())
    test_garments = {garment_id(r["positive_product_id"]) for r in humans}
    train_garments = {garment_id(r["positive_product_id"]) for r in examples}
    assert not test_garments & train_garments
    assert {r["positive_example"] for r in humans} <= records.keys()
    hashes_path = WORK / "image_hashes.json"
    if hashes_path.exists():
        hashes = json.loads(hashes_path.read_text())
        assert set(hashes) == set(records)
    else:
        print(json.dumps({"phase": "hash_images", "images": len(records)}), flush=True)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            hashes = dict(pool.map(digest_image, sorted(records)))
        hashes_path.write_text(json.dumps(hashes, sort_keys=True) + "\n")
    return raw, records, hashes, humans, examples, test_garments, train_garments


def assign_splits(records, hashes, held_out, examples, seed):
    parent = {garment_id(r["product_id"]): garment_id(r["product_id"]) for r in records.values()}

    def root(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_hash = {}
    for path, record in sorted(records.items()):
        g = garment_id(record["product_id"])
        previous = by_hash.setdefault(hashes[path], g)
        a, b = sorted((root(g), root(previous)))
        parent[b] = a
    groups = {g: root(g) for g in parent}
    test_groups = {groups[g] for g in held_out}
    train_groups = {groups[g] for g in examples}
    if test_groups & train_groups:
        raise ValueError("Human evaluation and rephrasing examples contain duplicate images")
    categories = defaultdict(set)
    for record in records.values():
        categories[groups[garment_id(record["product_id"])]].add(record["category2"])
    strata = defaultdict(list)
    for group, cats in categories.items():
        strata[tuple(sorted(cats))].append(group)
    owner = {}
    rng = random.Random(seed)
    for cats, members in sorted(strata.items()):
        members = sorted(members)
        rng.shuffle(members)
        forced_test = [g for g in members if g in test_groups]
        forced_train = [g for g in members if g in train_groups]
        free = [g for g in members if g not in test_groups | train_groups]
        n_test = max(len(forced_test), 2, round(len(members) * .1)) if len(members) >= 6 or forced_test else 0
        n_val = max(2, round(len(members) * .1)) if len(members) >= 6 else 0
        n_test = min(n_test, len(members) - len(forced_train))
        n_val = min(n_val, max(0, len(members) - n_test - max(2, len(forced_train))))
        if n_val == 1:
            n_val = 0
        need_test = n_test - len(forced_test)
        for g in forced_test + free[:need_test]: owner[g] = "test"
        for g in free[need_test:need_test + n_val]: owner[g] = "validation"
        for g in forced_train + free[need_test + n_val:]: owner[g] = "train"
    return {g: owner[group] for g, group in groups.items()}


def prepare(raw, records, owners, seed):
    by_side_category = defaultdict(list)
    for record in sorted(records.values(), key=lambda r: r["image_path"]):
        side = owners[garment_id(record["product_id"])]
        by_side_category[side, record["category2"]].append(record)
    easy_pools = {(side, cat): [r for (s, c), values in by_side_category.items()
                              if s == side and c != cat for r in values]
                  for side, cat in by_side_category}
    candidate_garments = {key: {garment_id(r["product_id"]) for r in values}
                         for key, values in by_side_category.items()}
    pairs, changed = [], 0
    for index, original in enumerate(raw):
        row = dict(original)
        p = row["positive_product"]
        g = garment_id(p["product_id"])
        side = owners[g]
        rng = random.Random(seed + index)
        candidates = by_side_category[side, p["category2"]]
        if not candidate_garments[side, p["category2"]] - {g}:
            raise ValueError(f"No same-category negative for {g} in {side}")
        hard = row["hard_neg_product"]
        regenerate = (owners[garment_id(hard["product_id"])] != side
                      or garment_id(hard["product_id"]) == g
                      or hard["category2"] != p["category2"])
        if regenerate:
            hard = rng.choice(candidates)
            while garment_id(hard["product_id"]) == g:
                hard = rng.choice(candidates)
            row = make_raw_example(p, hard, row["easy_neg_product"], row["category_key"])
            changed += 1
        easy = rng.choice(easy_pools[side, p["category2"]])
        while garment_id(easy["product_id"]) == g:
            easy = rng.choice(easy_pools[side, p["category2"]])
        row.update(split=side, easy_neg_product=easy, easy_negative_product_id=easy["product_id"],
                   easy_negative_category=easy["category2"], easy_negative_query_distance=None)
        pairs.append((index, row, regenerate))
    return pairs, by_side_category, changed


def generate(index, row, regenerate, pools, examples, seed):
    if not regenerate:
        return index, row
    rng = random.Random(seed + index)
    p = row["positive_product"]
    candidates = pools[row["split"], p["category2"]]
    for attempt in range(5):
        features = get_visual_common_and_differentiating_features(p, row["hard_neg_product"], model_id="gemini-2.5-flash")
        if features is not None and (features[1] or features[2]):
            break
        hard = rng.choice(candidates)
        while garment_id(hard["product_id"]) == garment_id(p["product_id"]):
            hard = rng.choice(candidates)
        row.update(hard_neg_product=hard, hard_negative_product_id=hard["product_id"],
                   hard_negative_category=hard["category2"])
    else:
        raise RuntimeError(f"No differentiating attributes for row {index}")
    common, unique_pos, unique_neg, neither = features
    row.update(generate_visual_distance_example(row["item"], common, unique_pos, unique_neg, neither, 10, rng))
    row["hard_negative_distance_source"] = "gemini-2.5-flash"
    rephrased = rephrase_query(row["nl_query"], product_description(row), "gemini-2.5-flash", examples)
    if not rephrased:
        raise RuntimeError(f"Rephrasing failed for row {index}")
    row["rephrased_query"] = rephrased
    return index, row


def separate_queries(rows, humans, examples, seed):
    seen = {query_key(r["human_query"]): "test" for r in humans}
    for r in examples:
        seen[query_key(r["human_query"])] = "train"
    rewritten = 0
    for index, row in enumerate(rows):
        rng = random.Random(seed + index + 100000)
        for attempt in range(30):
            keys = [query_key(row[c]) for c in ("nl_query", "rephrased_query")]
            if all(k not in seen or seen[k] == row["split"] for k in keys):
                for k in keys: seen[k] = row["split"]
                break
            row.update(generate_visual_distance_example(
                row["item"], row["full_common_features"], row["full_unique_pos_features"],
                row["full_unique_neg_features"], row["full_neither_features"], 10, rng))
            value = rephrase_query(row["nl_query"], product_description(row), "gemini-2.5-flash",
                                   [r["human_query"] for r in examples])
            if not value: raise RuntimeError(f"Collision rephrasing failed for row {index}")
            row["rephrased_query"] = value
            rewritten += 1
        else:
            raise RuntimeError(f"Could not separate query for row {index}")
    return rewritten


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    WORK.mkdir(parents=True, exist_ok=True)
    raw, records, hashes, humans, examples, held_out, in_context = load_inputs(args.workers)
    owners = assign_splits(records, hashes, held_out, in_context, args.seed)
    pairs, pools, changed = prepare(raw, records, owners, args.seed)
    plan = {"seed": args.seed, "raw_triples": len(pairs), "regenerate": changed,
            "retain": len(pairs) - changed, "triples_by_split": dict(Counter(r["split"] for _, r, _ in pairs)),
            "human_test_garments": len(held_out), "in_context_train_garments": len(in_context)}
    (WORK / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (WORK / "garment_splits.json").write_text(json.dumps(owners, sort_keys=True) + "\n")
    print(json.dumps(plan), flush=True)
    if args.plan_only: return
    journal = WORK / "generated.jsonl"
    done = {}
    if journal.exists():
        for line in journal.open():
            saved = json.loads(line)
            done[saved["index"]] = saved["row"]
    todo = [(i, r, new) for i, r, new in pairs if i not in done]
    with journal.open("a") as handle, ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(generate, i, r, new, pools, [e["human_query"] for e in examples], args.seed)
                   for i, r, new in todo]
        for future in as_completed(futures):
            index, row = future.result()
            done[index] = row
            handle.write(json.dumps({"index": index, "row": row}) + "\n")
            handle.flush()
            if len(done) % 100 == 0 or len(done) == len(pairs):
                print(json.dumps({"phase": "generate", "complete": len(done), "total": len(pairs)}), flush=True)
    rows = [done[i] for i in range(len(pairs))]
    collisions = separate_queries(rows, humans, examples, args.seed)
    hash_owner = {}
    for row in rows:
        for role in ("positive_product", "hard_neg_product", "easy_neg_product"):
            record = row[role]
            assert owners[garment_id(record["product_id"])] == row["split"]
            digest = hashes[record["image_path"]]
            assert hash_owner.setdefault(digest, row["split"]) == row["split"]
    raw_out = WORK / "disjoint_raw.jsonl"
    with raw_out.open("w") as handle:
        for row in rows: handle.write(json.dumps(row) + "\n")
    for path in (NEW_BASE, Path(str(NEW_BASE) + "_rephrased-in-context")):
        if path.exists(): raise FileExistsError(path)
        save_processed_dataset(rows, str(path))
        verify_image_splits(load_from_disk(str(path)))
        (path / "image_hashes.json").write_text(json.dumps(hashes, sort_keys=True) + "\n")
    for source in sorted(OLD_BASE.parent.glob(OLD_BASE.name + "_human*")):
        if not source.is_dir(): continue
        data = load_from_disk(str(source))
        if "in-context" not in source.name: continue
        assert all(owners[garment_id(p)] == "test" for p in data["positive_product_id"])
        dest = Path(str(NEW_BASE) + source.name[len(OLD_BASE.name):])
        data.save_to_disk(str(dest))
    plan.update(phase="complete", query_collision_rewrites=collisions, output=str(NEW_BASE),
                image_hash_overlap=0, garment_overlap=0, query_overlap=0)
    (WORK / "audit.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(json.dumps(plan), flush=True)


if __name__ == "__main__":
    main()
