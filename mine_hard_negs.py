"""Replace the labeled hard negative of every training row with a retrieval-mined one.

Processed dataset in, processed dataset out. The output keeps the input's schema and rows,
except that on the TRAIN split every hard-negative row's `negative_example` is replaced by
the product a frozen teacher embedder retrieves for the query under the positive-aware rule
of NV-Retriever (Moreira et al., CIKM 2025), and hard rows with no admissible candidate are
dropped. Validation and test rows are untouched, so a model trained on the output is
evaluated on exactly the corpus and queries of the input.

Mining rule, per (query, positive) training pair:
  1. score every product in the query's own split pool with the teacher (cosine);
  2. take the top `--top-k` candidates, excluding every labeled positive of that query;
  3. with --filter percpos, discard candidates scoring above (1 - relative_margin) x the
     positive's score, the TopK-PercPos filter at its published optimum, 95% of the positive;
     --filter none keeps every candidate (NV-Retriever's naive top-k baseline);
  4. the best surviving candidate is the mined negative. No survivor: the row is dropped.

The pool is the split's own products (positives and negatives of its rows), never another
split's, so the leakage-free split of the text dataset is preserved. Image datasets
require a precomputed document-disjoint split.

`query_distance` is null on mined rows (source "mined"): the distance was never measured.
Graded losses refuse those rows until a labeling pass fills them (utils/distance_labels).

Output: <dataset>_mined-<query_kind>/ with mining_report.json and mining_rows.jsonl inside.
The teacher never trains; it runs one forward pass over the pool and the queries.
"""

import argparse
import hashlib
import json
import os
import time
from collections import Counter, defaultdict

import torch
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from tqdm import tqdm

from train import QUERY_COLUMNS, is_usable_row, load_rgb_image, seeded_query_split
from utils.distance_labels import EASY_NEGATIVE_SOURCE, MINED_NEGATIVE_SOURCE

DEFAULT_TEACHERS = {
    "text": "intfloat/e5-mistral-7b-instruct",
    "multimodal": "sentence-transformers/clip-ViT-L-14",
}
# e5-mistral scores queries with a task instruction and documents bare.
E5_QUERY_PROMPT = ("Instruct: Given a product search query, retrieve products that satisfy "
                   "the query\nQuery: ")
ID_COLUMNS = {
    "text": ("positive_id", "negative_id"),
    "multimodal": ("positive_product_id", "negative_product_id"),
}
# Annotations that describe the labeled negative relative to the positive. They say
# nothing about a mined product, so they are blanked on mined rows.
PAIR_COLUMNS = ("full_common_features", "full_unique_pos_features", "full_unique_neg_features",
                "full_neither_features", "distance_source")
CACHE_DIR = "dataset/processed/_teacher_cache"


def columns(dataset, names):
    """Plain Python lists of the named columns. `dataset[name]` is a lazy column in this
    datasets version, and indexing it per row fetches one Arrow row at a time."""
    frame = dataset.select_columns([n for n in names if n in dataset.column_names]).to_pandas()
    return {n: frame[n].tolist() for n in frame.columns}


def row_splits(dataset, anchor_column, seed):
    """Split name per row: the `split` column when present, else train.py's seeded split.

    Rows train.py would filter out get None and are left alone.
    """
    if "positive_product_id" in dataset.column_names:
        from utils.image_split import verify_image_splits
        verify_image_splits(dataset)
    cols = columns(dataset, [anchor_column, "positive_example", "negative_example", "split"])
    anchors = cols[anchor_column]
    usable = [is_usable_row({"anchor": a, "positive": p, "negative": n})
              for a, p, n in zip(anchors, cols["positive_example"], cols["negative_example"])]
    if "split" in cols:
        return [s if u else None for s, u in zip(cols["split"], usable)]
    unique = list(dict.fromkeys(a for a, u in zip(anchors, usable) if u))
    train_q, eval_q, test_q = seeded_query_split(unique, seed)
    out = []
    for a, u in zip(anchors, usable):
        if not u:
            out.append(None)
        elif a in train_q:
            out.append("train")
        elif a in eval_q:
            out.append("validation")
        else:
            assert a in test_q, a
            out.append("test")
    return out


def load_teacher(name, modality, max_seq_length, device):
    kwargs = {"trust_remote_code": modality == "multimodal"}
    if modality == "text":
        kwargs["model_kwargs"] = {"torch_dtype": torch.float16}
    model = SentenceTransformer(name, device=device, **kwargs)
    if modality == "text":
        model.max_seq_length = max_seq_length
    return model


class _ImagePaths(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return load_rgb_image(self.paths[i])


def encode_pool(model, pool, modality, batch_size, workers, encode_devices=None):
    """Text goes straight to the encoder. Images are decoded by worker processes, since
    decoding on the main thread is the bottleneck on a loaded machine."""
    if modality == "text":
        return model.encode(pool, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True,
                            **({"device": encode_devices} if encode_devices else {}))
    loader = DataLoader(_ImagePaths(pool), batch_size=batch_size, num_workers=workers, collate_fn=list)
    chunks = [model.encode(images, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
              for images in tqdm(loader, desc="Encoding images")]
    return torch.cat(chunks, dim=0)


def pool_embeddings(model, pool, modality, teacher, batch_size, workers, device, encode_devices=None):
    """Teacher embeddings of the pool, cached on disk by (teacher, exact pool)."""
    key = hashlib.sha1((teacher + "\n" + "\n".join(pool)).encode()).hexdigest()
    path = os.path.join(CACHE_DIR, f"{key}.pt")
    if os.path.exists(path):
        print(f"pool embeddings: cache hit {path}")
        return torch.load(path).to(device)
    start = time.time()
    emb = encode_pool(model, pool, modality, batch_size, workers, encode_devices)
    emb = torch.nn.functional.normalize(emb.float(), dim=1).half()
    os.makedirs(CACHE_DIR, exist_ok=True)
    torch.save(emb.cpu(), path)
    print(f"pool embeddings: {len(pool):,} items in {time.time() - start:.0f}s -> {path}")
    return emb.to(device)


def query_embeddings(model, queries, teacher, prompt, batch_size, device, encode_devices=None):
    key_data = {'teacher': teacher, 'max_seq_length': model.max_seq_length,
                'prompt': prompt, 'queries': queries, 'batch_size': batch_size}
    key = hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    path = os.path.join(CACHE_DIR, f'queries-{key}.pt')
    if os.path.exists(path):
        print(f'query embeddings: cache hit {path}', flush=True)
        return torch.load(path, map_location=device, weights_only=True)
    embeddings = model.encode([prompt + query for query in queries], batch_size=batch_size,
                              convert_to_tensor=True, normalize_embeddings=True,
                              show_progress_bar=True,
                              **({"device": encode_devices} if encode_devices else {})).half()
    os.makedirs(CACHE_DIR, exist_ok=True)
    temporary = path + f'.{os.getpid()}.tmp'
    torch.save(embeddings.cpu(), temporary)
    os.replace(temporary, path)
    return embeddings.to(device)


def mine(dataset, modality, kind, teacher, model, args):
    anchor_column = QUERY_COLUMNS[kind]
    pos_id_col, neg_id_col = ID_COLUMNS[modality]
    splits = row_splits(dataset, anchor_column, args.split_seed)
    cols = columns(dataset, [anchor_column, "negative_example_source", "positive_example",
                             "negative_example", pos_id_col, neg_id_col, "query_distance",
                             "positive_category", "negative_category"])
    sources = cols["negative_example_source"]
    anchors = cols[anchor_column]
    positives = cols["positive_example"]
    negatives = cols["negative_example"]
    pos_ids = cols[pos_id_col]
    neg_ids = cols[neg_id_col]
    distances = cols["query_distance"]
    has_category = "positive_category" in cols
    pos_cats = cols["positive_category"] if has_category else None
    neg_cats = cols["negative_category"] if has_category else None

    train_rows = [i for i, s in enumerate(splits) if s == "train"]
    pool = list(dict.fromkeys([positives[i] for i in train_rows] + [negatives[i] for i in train_rows]))
    pool_index = {item: j for j, item in enumerate(pool)}
    item_id = {}
    item_category = {}
    for i in train_rows:
        item_id[positives[i]] = pos_ids[i]
        item_id[negatives[i]] = neg_ids[i]
        if has_category:
            item_category[positives[i]] = pos_cats[i]
            item_category[negatives[i]] = neg_cats[i]
    positives_of = defaultdict(set)
    for i in train_rows:
        positives_of[anchors[i]].add(pool_index[positives[i]])

    hard_rows = [i for i in train_rows if sources[i] != EASY_NEGATIVE_SOURCE]
    if args.limit:
        hard_rows = hard_rows[:args.limit]
    print(f"[{kind}] train rows {len(train_rows):,} | hard rows to mine {len(hard_rows):,} "
          f"| pool {len(pool):,} | unique queries {len(positives_of):,}")

    pool_emb = pool_embeddings(model, pool, modality, teacher, args.batch_size, args.workers, args.device, args.encode_devices)
    unique_anchors = list(dict.fromkeys(anchors[i] for i in hard_rows))
    prompt = args.query_prompt if modality == "text" else ""
    start = time.time()
    anchor_emb = query_embeddings(model, unique_anchors, teacher, prompt, args.query_batch_size, args.device, args.encode_devices)
    print(f"[{kind}] {len(unique_anchors):,} queries encoded in {time.time() - start:.0f}s")
    anchor_index = {a: j for j, a in enumerate(unique_anchors)}

    keep_below = 1.0 - args.relative_margin if args.filter == "percpos" else None
    records = {}
    for start in tqdm(range(0, len(hard_rows), args.chunk_size), desc="Mining hard negatives"):
        chunk = hard_rows[start:start + args.chunk_size]
        q = anchor_emb[[anchor_index[anchors[i]] for i in chunk]]
        scores = (q @ pool_emb.T).float()
        top_scores, top_idx = scores.topk(min(args.top_k, scores.shape[1]), dim=1)
        top_scores, top_idx = top_scores.cpu().tolist(), top_idx.cpu().tolist()
        for r, i in enumerate(chunk):
            pos_score = scores[r, pool_index[positives[i]]].item()
            threshold = keep_below * pos_score if keep_below is not None else float("inf")
            excluded = positives_of[anchors[i]]
            rank = 0
            filtered = 0
            mined = None
            mined_score = None
            mined_source = None
            mined_rank = None
            weakest = None  # (cand, score, rank) of the lowest-scoring non-positive candidate seen
            survivors = 0
            for cand, score in zip(top_idx[r], top_scores[r]):
                if cand in excluded:
                    continue
                rank += 1
                weakest = (cand, score, rank)
                if score >= threshold:
                    filtered += 1
                    continue
                # --skip-survivors N takes the (N+1)-th survivor (NV-Retriever's top-k shifted
                # variant); a row with fewer survivors keeps its deepest one.
                mined, mined_score, mined_rank, mined_source = cand, score, rank, "rule"
                survivors += 1
                if survivors > args.skip_survivors:
                    break
            if mined is not None:
                rank = mined_rank
            # top_k candidates all inside the margin: the rule drops the row. --fallback weakest
            # keeps it with the lowest-scoring candidate retrieved -- the weakest hard negative
            # available, and the one least likely to be a false negative -- so the mined train set
            # stays the same size as the labeled one. Tagged so the report separates the two.
            if mined is None and args.fallback == "weakest" and weakest is not None:
                mined, mined_score, rank = weakest
                mined_source = "fallback-weakest"
            records[i] = {
                "row": i,
                "query": anchors[i],
                "positive_id": pos_ids[i],
                "labeled_negative_id": neg_ids[i],
                "labeled_source": sources[i],
                "labeled_distance": distances[i],
                "positive_score": pos_score,
                "mined_id": item_id[pool[mined]] if mined is not None else None,
                "mined_item": pool[mined] if mined is not None else None,
                "mined_score": mined_score,
                "mined_rank": rank if mined is not None else None,
                "mined_source": mined_source,
                "survivors_seen": survivors,
                "filtered_above_threshold": filtered,
            }
    return records, item_id, item_category


def apply(dataset, records, modality, item_id, item_category, source=MINED_NEGATIVE_SOURCE):
    pos_id_col, neg_id_col = ID_COLUMNS[modality]
    blank = [c for c in PAIR_COLUMNS if c in dataset.column_names]

    def swap(batch, indices):
        batch = {k: list(v) for k, v in batch.items()}
        for r, idx in enumerate(indices):
            rec = records[idx] if idx in records else None
            if rec is None or rec["mined_id"] is None:
                continue
            batch["negative_example"][r] = rec["mined_item"]
            batch[neg_id_col][r] = rec["mined_id"]
            batch["negative_example_source"][r] = source
            batch["query_distance"][r] = None
            for c in blank:
                batch[c][r] = None
            if "negative_category" in batch:
                batch["negative_category"][r] = item_category[rec["mined_item"]]
        return batch

    dropped = {i for i, rec in records.items() if rec["mined_id"] is None}
    out = dataset.map(swap, with_indices=True, batched=True, features=dataset.features,
                      desc="Swapping negatives")
    out = out.filter(lambda _, indices: [i not in dropped for i in indices], with_indices=True,
                     batched=True, desc="Dropping unmined rows")
    return out


def report(records, kind, args):
    recs = list(records.values())
    mined = [r for r in recs if r["mined_id"] is not None]
    by_source = Counter(r["labeled_source"] for r in recs)
    dropped_by_source = Counter(r["labeled_source"] for r in recs if r["mined_id"] is None)
    ranks = [r["mined_rank"] for r in mined]
    out = {
        "query_kind": kind,
        "teacher": args.teacher,
        "top_k": args.top_k,
        "filter": args.filter,
        "relative_margin": args.relative_margin if args.filter == "percpos" else None,
        "n_hard_train_rows": len(recs),
        "fallback": args.fallback,
        "skip_survivors": args.skip_survivors,
        "n_short_of_skip": sum(1 for r in mined if r["mined_source"] == "rule"
                               and r["survivors_seen"] <= args.skip_survivors),
        "n_mined": len(mined),
        "n_mined_by_rule": sum(r["mined_source"] == "rule" for r in mined),
        "n_mined_by_fallback": sum(r["mined_source"] == "fallback-weakest" for r in mined),
        "n_dropped_no_survivor": len(recs) - len(mined),
        "n_top1_above_threshold": sum(r["filtered_above_threshold"] > 0 for r in recs),
        "n_mined_equals_labeled": sum(r["mined_id"] == r["labeled_negative_id"] for r in mined),
        "mean_filtered_above_threshold": sum(r["filtered_above_threshold"] for r in recs) / max(len(recs), 1),
        "mean_mined_rank": sum(ranks) / max(len(ranks), 1),
        "median_mined_rank": sorted(ranks)[len(ranks) // 2] if ranks else None,
        "mean_positive_score": sum(r["positive_score"] for r in recs) / max(len(recs), 1),
        "mean_mined_score": sum(r["mined_score"] for r in mined) / max(len(mined), 1),
        "hard_rows_by_labeled_source": dict(by_source),
        "dropped_by_labeled_source": dict(dropped_by_source),
    }
    print(json.dumps(out, indent=2))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dataset", help="processed HuggingFace dataset directory")
    ap.add_argument("--modality", choices=["text", "multimodal"], required=True)
    ap.add_argument("--query-kinds", nargs="+", choices=list(QUERY_COLUMNS), required=True,
                    help="one output dataset per kind; the pool embeddings are shared")
    ap.add_argument("--teacher", default=None, help="frozen embedder that ranks the pool")
    ap.add_argument("--query-prompt", default=E5_QUERY_PROMPT, help="prefix for text queries")
    ap.add_argument("--top-k", type=int, default=100, help="candidates retrieved before filtering")
    ap.add_argument("--filter", choices=["percpos", "none"], default="percpos",
                    help="percpos: TopK-PercPos score filter; none: naive top-k, no filter")
    ap.add_argument("--relative-margin", type=float, default=0.05,
                    help="percpos: discard candidates scoring above (1 - margin) x positive score")
    ap.add_argument("--skip-survivors", type=int, default=0,
                    help="take the (N+1)-th survivor instead of the first (top-k shifted)")
    ap.add_argument("--variant", default=None,
                    help="suffix for the output dir, <dataset>_mined-<kind>_<variant>, so mining "
                         "sweeps keep their datasets apart; the default config has none")
    ap.add_argument("--fallback", choices=["none", "weakest"], default="none",
                    help="no survivor: drop the row (none) or keep the lowest-scoring retrieved "
                         "candidate as a weaker hard negative (weakest)")
    ap.add_argument("--split-seed", type=int, default=42, help="train.py --split-seed")
    ap.add_argument("--max-seq-length", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=32, help="pool encoding batch")
    ap.add_argument("--workers", type=int, default=16, help="image decoding processes")
    ap.add_argument("--query-batch-size", type=int, default=128)
    ap.add_argument("--chunk-size", type=int, default=512, help="queries scored per matmul")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--encode-devices", nargs="+", help="Devices for shared multi-process text encoding")
    ap.add_argument("--limit", type=int, default=None, help="dry run: mine only the first N hard rows")
    ap.add_argument("--out-root", default=None, help="write outputs here instead of beside the input")
    ap.add_argument("--output-dir", help="Explicit destination for a single query kind")
    args = ap.parse_args()
    if args.output_dir and (args.out_root or len(args.query_kinds) != 1):
        ap.error('--output-dir requires one query kind and cannot be combined with --out-root')
    args.teacher = args.teacher or DEFAULT_TEACHERS[args.modality]

    dataset = load_from_disk(args.dataset)
    print(f"{args.dataset}: {len(dataset):,} rows, columns {dataset.column_names}")
    model = load_teacher(args.teacher, args.modality, args.max_seq_length,
                         "cpu" if args.encode_devices else args.device)

    base = os.path.basename(args.dataset.rstrip("/"))
    root = args.out_root or os.path.dirname(args.dataset.rstrip("/"))
    for kind in args.query_kinds:
        records, item_id, item_category = mine(dataset, args.modality, kind, args.teacher, model, args)
        out_dir = args.output_dir or os.path.join(root, f"{base}_mined-{kind}" + (f"_{args.variant}" if args.variant else ""))
        out = apply(dataset, records, args.modality, item_id, item_category)
        out.save_to_disk(out_dir)
        summary = report(records, kind, args)
        summary["input"] = args.dataset
        summary["output"] = out_dir
        summary["n_output_rows"] = len(out)
        with open(os.path.join(out_dir, "mining_report.json"), "w") as f:
            json.dump(summary, f, indent=2)
        with open(os.path.join(out_dir, "mining_rows.jsonl"), "w") as f:
            for rec in records.values():
                f.write(json.dumps(rec) + "\n")
        print(f"[{kind}] saved {len(out):,} rows to {out_dir}")


if __name__ == "__main__":
    main()
