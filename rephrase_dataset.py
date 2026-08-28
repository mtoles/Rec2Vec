"""Rewrite every nl_query in a dataset as a `rephrased_query`, using the positive product as context.

Takes either a raw jsonl or a processed HuggingFace dataset directory. For a processed dataset the
column is filled in place and each distinct nl_query is rephrased once, so the two rows of a triple
share one call. Any existing rephrased_query is overwritten. Output goes to the input path with
`_rephrased` appended.

API calls go through utils.retry, so joblib caches every request under .cache/: a re-run after a
crash replays finished rows from disk instead of paying for them again. Rows are also appended to
the output as they finish, and --resume skips the ones already written.

Usage:
    python rephrase_dataset.py dataset/feature-distance-dataset_gemini-2.5-flash_1000000_fixed_distance.jsonl
    python rephrase_dataset.py <path> --limit 5 --show      # sample and print prompts + answers
    python rephrase_dataset.py dataset/processed/<dir> --descriptions <raw.jsonl>

--descriptions supplies the product text when the processed rows do not carry it (the image
dataset stores a file path in positive_example); it is looked up by nl_query.

The run prints measured token counts and cost from the API responses; to price a full pass, run a
sample with --limit and multiply by the row count.
"""

import argparse
import json
import logging
import os
from multiprocessing import Pool
from pathlib import Path

from datasets import load_from_disk

from typing import Optional

from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from jsonschema import ValidationError, validate

from utils.retry import (
    get_cost_summary,
    reset_cost_tracking,
    retry_with_fallback,
    update_cost_from_summary,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "gemini-2.5-flash"


REPHRASE_PROMPT = """
    Original query: '{query}'

    Context: The user will be happy with the following product:

    {product_description}

    Task: Rephrase the query so it keeps the same meaning but sounds more natural and uses synonyms.

    Rules:
    - Preserve intent exactly (positive stays positive, negative stays negative).
    - You MUST rephrase at least half of the comma-separated attributes.
    - Do NOT force rewording if it sounds unnatural.
    - Do NOT change the product/item name unless a very natural alternative exists.
    - Do NOT change measurement units or numeric values.
    - Keep the query fluent and realistic.

    Return ONLY a JSON object: {{"rephrased_query": "..."}}"""


def rephrase_query(nl_query: str, product_description: str, model_id: str) -> Optional[str]:
    """
    Rephrase a natural language query using the LLM to mean the same thing while avoiding keywords.

    product_description is the text of the positive product, given to the model as context so
    the rewording stays consistent with the item the query is meant to retrieve.
    """
    prompt = REPHRASE_PROMPT


    schema = {
        "type": "object",
        "properties": {
            "rephrased_query": {"type": "string"},
        },
        "required": ["rephrased_query"],
        "additionalProperties": False,
    }

    def validate_response(content: str) -> bool:
        try:
            parsed = json.loads(content)
            validate(instance=parsed, schema=schema)
            return True
        except (json.JSONDecodeError, ValidationError):
            return False

    messages = [
        {
            "role": "user",
            "content": prompt.format(query=nl_query, product_description=product_description),
        }
    ]
    
    response = retry_with_fallback(
        messages=messages,
        validation_func=validate_response,
        max_retries=3,
        fallback_value=None,
        model_id=model_id,
    )
    
    if response is None:
        return None
        
    try:
        parsed = json.loads(response)
        return parsed["rephrased_query"]
    except json.JSONDecodeError:
        return None



def product_description(row):
    """The positive product's text. Key differs between the text and image pipelines."""
    product = row.get("positive_product") or {}
    for key in ("product_text", "text", "product_title"):
        value = product.get(key)
        if value:
            return value
    return ""


REPHRASED_SUFFIX = "_rephrased"


DATA_SUFFIXES = {".jsonl", ".json"}


def output_path(input_path):
    """Same path with _rephrased appended: a.jsonl -> a_rephrased.jsonl, dir -> dir_rephrased.

    Only .jsonl/.json count as extensions. Dataset directory names contain dots of their own
    (gemini-2.5-flash), and Path.stem would happily treat ".5-flash_1000000_nolek" as one.

    Refuses an input that already carries the suffix, which would otherwise silently produce a
    _rephrased_rephrased dataset built by rephrasing rephrasings.
    """
    path = Path(input_path.rstrip("/"))
    extension = path.suffix if path.suffix in DATA_SUFFIXES else ""
    stem = path.name[:-len(extension)] if extension else path.name
    if stem.endswith(REPHRASED_SUFFIX):
        raise SystemExit(
            f"{input_path} already ends in {REPHRASED_SUFFIX}; rephrase the original dataset "
            f"instead (rephrasing a rephrasing is not what you want)"
        )
    return str(path.with_name(stem + REPHRASED_SUFFIX + extension))


def description_map(jsonl_path):
    """nl_query -> positive product text, for datasets whose rows do not carry the text."""
    mapping = {}
    with open(jsonl_path) as fh:
        for line in fh:
            row = json.loads(line)
            query = row.get("nl_query")
            if query:
                mapping[query] = product_description(row)
    return mapping


def rephrase_processed(args, out_path):
    """Fill rephrased_query on a processed dataset, one call per distinct nl_query."""
    dataset = load_from_disk(args.dataset)
    descriptions = description_map(args.descriptions) if args.descriptions else {}

    queries = list(dict.fromkeys(dataset["nl_query"]))
    if args.limit:
        queries = queries[:args.limit]
    # A processed row already holds the product text unless it holds an image path.
    if not descriptions:
        for query, positive in zip(dataset["nl_query"], dataset["positive_example"]):
            descriptions.setdefault(query, positive)

    print(f"{len(dataset)} rows, {len(queries)} distinct queries to rephrase")
    rows = [{"nl_query": q, "positive_product": {"product_text": descriptions.get(q, "")}}
            for q in queries]

    reset_cost_tracking()
    results = run_rows(rows, args)
    mapping = {r["nl_query"]: r["rephrased_query"] for r in results}
    failed = sum(1 for r in results if r.get("rephrase_failed"))

    dataset = dataset.map(
        lambda row: {"rephrased_query": mapping.get(row["nl_query"], "")},
        desc="Writing rephrased_query",
    )
    dataset.save_to_disk(out_path)
    summary = get_cost_summary()
    print(f"\nwrote {len(dataset)} rows -> {out_path}")
    print(f"failed rephrasings: {failed}")
    print(f"api calls: {summary['total_api_calls']}  tokens: {summary['total_tokens_used']:,}  "
          f"cost: ${summary['total_cost']:.4f}")


def load_rows(path, limit=None):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if limit and len(rows) == limit:
                break
    return rows


def rephrase_row(row, model_id):
    """Overwrite rephrased_query. On failure the row keeps its original query, flagged."""
    query = row.get("nl_query")
    if not query:
        row["rephrased_query"] = ""
        row["rephrase_failed"] = True
        return row
    rephrased = rephrase_query(query, product_description(row), model_id)
    row["rephrased_query"] = rephrased if rephrased else query
    row["rephrase_failed"] = not rephrased
    return row


def _worker(args_tuple):
    reset_cost_tracking()
    row, model_id = args_tuple
    return rephrase_row(row, model_id), get_cost_summary()


def run_rows(rows, args, on_row=None):
    """Rephrase rows with a worker pool that ramps up while chunks come back clean and halves
    when they do not, which together with the backoff in utils.retry keeps a long pass alive.
    Returns the processed rows; on_row is called with each as it completes."""
    done_rows = []
    failed = 0
    progress = tqdm(total=len(rows), desc="Rephrasing", unit="row", dynamic_ncols=True,
                    mininterval=0.1, smoothing=0.05, leave=True)
    with logging_redirect_tqdm():
        if args.max_workers > 1 and len(rows) > 1:
            workers = max(1, args.workers)
            index = 0
            while index < len(rows):
                chunk = rows[index:index + workers * 4]
                with Pool(processes=workers) as pool:
                    results = pool.imap(_worker, [(r, args.model_id) for r in chunk], chunksize=1)
                    chunk_failed = 0
                    for row, summary in results:
                        update_cost_from_summary(summary)
                        chunk_failed += bool(row.get("rephrase_failed"))
                        failed += bool(row.get("rephrase_failed"))
                        done_rows.append(row)
                        if on_row:
                            on_row(row)
                        progress.update(1)
                        progress.set_postfix(workers=workers, failed=failed,
                                             cost=f"${get_cost_summary()['total_cost']:.4f}",
                                             refresh=True)
                index += len(chunk)
                rate = chunk_failed / len(chunk)
                if rate > args.fail_threshold:
                    workers = max(1, workers // 2)
                else:
                    workers = min(args.max_workers, max(workers + 1, int(workers * 1.5)))
        else:
            for row in rows:
                row = rephrase_row(row, args.model_id)
                failed += bool(row.get("rephrase_failed"))
                done_rows.append(row)
                if on_row:
                    on_row(row)
                progress.update(1)
                progress.set_postfix(failed=failed,
                                     cost=f"${get_cost_summary()['total_cost']:.4f}",
                                     refresh=True)
    progress.close()
    return done_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="input jsonl, or a processed HuggingFace dataset directory")
    ap.add_argument("--model-id", default=DEFAULT_MODEL)
    ap.add_argument("--out", default=None, help="override output path")
    ap.add_argument("--descriptions", default=None,
                    help="jsonl holding the product text, when the processed rows do not "
                         "(joined on nl_query)")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N rows")
    ap.add_argument("--workers", type=int, default=int(os.getenv("N_WORKERS", "1")),
                    help="starting worker count; the pool ramps up from here")
    ap.add_argument("--max-workers", type=int, default=int(os.getenv("MAX_WORKERS", "32")),
                    help="ceiling for the ramp")
    ap.add_argument("--fail-threshold", type=float, default=0.05,
                    help="chunk failure rate above which the worker count is halved")
    ap.add_argument("--resume", action="store_true", help="skip rows already in the output file")
    ap.add_argument("--show", action="store_true", help="print each prompt and answer in full")
    args = ap.parse_args()

    # Validate the input name even when --out overrides the destination.
    default_out = output_path(args.dataset)
    out_path = args.out or default_out

    if os.path.isdir(args.dataset):
        rephrase_processed(args, out_path)
        return

    rows = load_rows(args.dataset, limit=args.limit)
    done = 0
    if args.resume and os.path.exists(out_path):
        done = sum(1 for _ in open(out_path))
        print(f"resuming: {done} rows already in {out_path}")
    todo = rows[done:]

    reset_cost_tracking()
    with open(out_path, "a" if done else "w") as out:
        def emit(row):
            out.write(json.dumps(row) + "\n")
            out.flush()
            if args.show:
                print("\n" + "=" * 100)
                print(REPHRASE_PROMPT.format(
                    query=row.get("nl_query", ""), product_description=product_description(row)))
                print("-" * 100)
                print("REPHRASED:", row["rephrased_query"])

        results = run_rows(todo, args, on_row=emit)

    failed = sum(1 for r in results if r.get("rephrase_failed"))
    summary = get_cost_summary()
    print(f"\nwrote {len(results)} rows -> {out_path}")
    print(f"failed rephrasings: {failed}")
    print(f"api calls: {summary['total_api_calls']}  tokens: {summary['total_tokens_used']:,}  "
          f"cost: ${summary['total_cost']:.4f}")


if __name__ == "__main__":
    main()
