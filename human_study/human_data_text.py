"""Build the Excel workbook for the Columbia Advanced Search Study.

Creates one workbook with N sheets (one per annotator), each pre-filled with N_PER_SHEET
product pairs drawn from the text dataset. Annotators fill in Q1-Q4; n_1 and n_2 tell them
how many attributes to list.

n_1 and n_2 are not random: they are the number of positive / negative constraints in the
synthetic query generated for the same product pair.

Usage (paths resolve relative to the repo root, so it runs from anywhere):
    python human_study/human_data_text.py                    # 10 sheets x 25 -> human_study/human_study.xlsx
    python human_study/human_data_text.py --sheets 4 --per-sheet 10 --out pilot.xlsx
    python human_study/human_data_text.py --backfill         # quota-sampled second study
"""

import argparse
import html
import json
import math
import random
import re
from collections import Counter
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

import attr_quota

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DEFAULT = REPO_ROOT / "dataset/feature-distance-dataset_gemini-2.5-flash_1000000_fixed_distance.jsonl"
OUT_DEFAULT = Path(__file__).resolve().parent / "human_study.xlsx"
LABELED_DEFAULT = REPO_ROOT / "dataset/processed/feature-distance-dataset_gemini-2.5-flash_1000000_nolek_human"

# n_1 / n_2 are the number of attributes the annotator is asked for. They come from
# the synthetic query built for the same pair, so a human query is length-matched to the
# synthetic one it will be compared against.
MAX_ATTRS = 3
# --backfill asks for the counts the synthetic query states rather than the differentiating
# ones the first study capped at MAX_ATTRS, and draws each (n_1, n_2) cell to a quota that makes
# the combined set reproduce the synthetic distribution. See attr_quota for why.
BACKFILL_TARGET_ROWS = 300

COLUMNS = ["Product 1", "Product 2", "n_1", "Q1", "n_2", "Q2", "Q3", "Q4", "Problem?"]
COL_WIDTHS = [70, 70, 6, 34, 6, 34, 40, 40, 16]

# Product texts are title / brand / color / description / bullets, and the description
# still carries raw seller HTML.
BLOCK_TAGS = re.compile(r"</?(?:p|br|li|ul|ol|div|tr)[^>]*>", re.I)
ALL_TAGS = re.compile(r"<[^>]+>")


def clean(text):
    """Turn a raw product_text blob into something a person can read in one cell."""
    text = BLOCK_TAGS.sub("\n", text or "")
    text = ALL_TAGS.sub(" ", text)
    text = html.unescape(text)
    lines = []
    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()
        if line and line.lower() != "none":
            lines.append(line)
    return "\n".join(lines)


def format_product(product, shopping_for=None):
    """First line names the search intent so the annotator has the item in front of them."""
    body = clean(product.get("product_text", ""))
    header = f"SHOPPING FOR: {shopping_for}\n\n" if shopping_for else ""
    return f"{header}{body}"


def attr_counts(row):
    """(n_1, n_2): differentiating constraint counts of this pair's synthetic query."""
    return len(row["selected_pos_features"]), len(row["selected_neg_features"])


def stated_counts(row):
    """(n_1, n_2): every attribute the synthetic query states, per side. The rendered query
    lists selected_pos + selected_common as required and selected_neg + selected_neither as
    excluded, so these are the counts a human query has to match to be length-comparable."""
    return (len(row["selected_pos_features"]) + len(row["selected_common_features"]),
            len(row["selected_neg_features"]) + len(row["selected_neither_features"]))


def readable(row, min_chars, max_chars):
    """Both products short enough to read in a cell and long enough to describe."""
    texts = [clean(row[k].get("product_text", "")) for k in ("positive_product", "hard_neg_product")]
    return all(min_chars <= len(t) <= max_chars for t in texts)


def usable(row, min_chars, max_chars):
    # Counts are capped at MAX_ATTRS so n_1 / n_2 are real counts rather than clamped ones.
    # Zero is allowed: it means the annotator lists nothing on that side.
    return readable(row, min_chars, max_chars) and all(n <= MAX_ATTRS for n in attr_counts(row))


def load_rows(path, needed, seed, min_chars, max_chars):
    """Sample `needed` usable rows, at most one per ESCI item so the sheets stay varied."""
    with open(path) as fh:
        raw = [json.loads(line) for line in fh]
    rng = random.Random(seed)
    rng.shuffle(raw)
    picked, seen_items, seen_products = [], set(), set()
    for row in raw:
        if len(picked) == needed:
            break
        pid = row["positive_product"]["product_id"]
        nid = row["hard_neg_product"]["product_id"]
        if row["item"] in seen_items or pid in seen_products or nid in seen_products:
            continue
        if not usable(row, min_chars, max_chars):
            continue
        seen_items.add(row["item"])
        seen_products.update((pid, nid))
        picked.append(row)
    if len(picked) < needed:
        raise SystemExit(f"only {len(picked)} usable rows found, need {needed}")
    return picked


def load_quota_rows(path, seed, min_chars, max_chars, quota, exclude_products):
    """Sample `quota[(n_1, n_2)]` rows at each stated-attribute cell.

    Same dedup rules as load_rows, plus the products of the first study: a product shown twice
    across the two studies would appear twice in the combined evaluation set.
    """
    with open(path) as fh:
        raw = [json.loads(line) for line in fh]
    rng = random.Random(seed)
    rng.shuffle(raw)
    picked, seen_items, seen_products, taken = [], set(), set(exclude_products), Counter()
    for row in raw:
        cell = stated_counts(row)
        if taken[cell] >= quota[cell]:
            continue
        pid = row["positive_product"]["product_id"]
        nid = row["hard_neg_product"]["product_id"]
        if row["item"] in seen_items or pid in seen_products or nid in seen_products:
            continue
        if not readable(row, min_chars, max_chars):
            continue
        seen_items.add(row["item"])
        seen_products.update((pid, nid))
        taken[cell] += 1
        picked.append(row)
    short = {cell: quota[cell] - taken[cell] for cell in quota if taken[cell] < quota[cell]}
    if short:
        raise SystemExit(f"pool exhausted at {len(short)} of {len(quota)} cells: {short}")
    return picked


def deal_round_robin(rows, sheets, seed):
    """Deal cell by cell so every sheet is an even slice of every (n_1, n_2) cell, then shuffle
    within the sheet so the annotator sees no order in the attribute counts.

    Annotators drop out, and a sheet dealt this way costs the same fraction of each cell when
    left unfinished rather than emptying one corner of the distribution."""
    ordered = sorted(rows, key=stated_counts)
    rng = random.Random(seed)
    dealt = [ordered[i::sheets] for i in range(sheets)]
    for sheet in dealt:
        rng.shuffle(sheet)
    return dealt


def write_sheet(ws, rows, counts):
    ws.append(COLUMNS)
    header_fill = PatternFill("solid", fgColor="D9E1F2")
    for i, _ in enumerate(COLUMNS, start=1):
        cell = ws.cell(row=1, column=i)
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(vertical="center")
        ws.column_dimensions[get_column_letter(i)].width = COL_WIDTHS[i - 1]

    for row in rows:
        n_1, n_2 = counts(row)
        ws.append([
            format_product(row["positive_product"], shopping_for=row["original_query"]),
            format_product(row["hard_neg_product"]),
            n_1,   # attributes Product 1 has that Product 2 lacks
            "",    # Q1
            n_2,   # attributes Product 2 has that Product 1 lacks
            "",    # Q2
            "",    # Q3
            "",    # Q4
            "",    # Problem?
        ])

    for r in range(2, ws.max_row + 1):
        ws.row_dimensions[r].height = 150
        for c in range(1, len(COLUMNS) + 1):
            ws.cell(row=r, column=c).alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW_DEFAULT), help="raw feature-distance jsonl")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--sheets", type=int, default=10)
    ap.add_argument("--per-sheet", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-chars", type=int, default=250, help="skip products with thin text")
    ap.add_argument("--max-chars", type=int, default=2500, help="skip products too long to read")
    ap.add_argument("--backfill", action="store_true",
                    help="ask for every attribute the synthetic query states, quota-sampled so "
                         "the combined set matches the synthetic distribution, into a "
                         "_backfill workbook")
    ap.add_argument("--target-rows", type=int, default=BACKFILL_TARGET_ROWS,
                    help="--backfill: size of the matched evaluation set")
    ap.add_argument("--labeled", default=str(LABELED_DEFAULT),
                    help="--backfill: human dataset of the first study, whose rows fill part "
                         "of the quota")
    ap.add_argument("--oversample", type=float, default=1.5,
                    help="--backfill: hand out this multiple of the quota, so annotator "
                         "dropout and the leak filter still leave every cell full")
    args = ap.parse_args()

    out = Path(args.out)
    if args.backfill and out == OUT_DEFAULT:
        out = out.with_name(f"{out.stem}_backfill{out.suffix}")

    if args.backfill:
        counts = stated_counts
        with open(args.raw) as fh:
            synthetic = Counter(stated_counts(json.loads(line)) for line in fh)
        already, products = attr_quota.labeled(args.labeled, "positive_id")
        quota, keep = attr_quota.build(synthetic, already, args.target_rows)
        print(attr_quota.summarize(synthetic, already, keep, quota))
        handout = Counter({k: math.ceil(v * args.oversample) for k, v in quota.items()})
        rows = load_quota_rows(args.raw, args.seed, args.min_chars, args.max_chars,
                               handout, products)
        sheets = deal_round_robin(rows, math.ceil(len(rows) / args.per_sheet), args.seed)
    else:
        counts = attr_counts
        rows = load_rows(args.raw, args.sheets * args.per_sheet, args.seed,
                         args.min_chars, args.max_chars)
        sheets = [rows[s * args.per_sheet:(s + 1) * args.per_sheet] for s in range(args.sheets)]

    wb = Workbook()
    wb.remove(wb.active)
    for s, sheet_rows in enumerate(sheets, start=1):
        write_sheet(wb.create_sheet(f"Annotator {s}"), sheet_rows, counts)
    wb.save(out)
    print(f"wrote {out}: {len(sheets)} sheets, {len(rows)} pairs")


if __name__ == "__main__":
    main()
