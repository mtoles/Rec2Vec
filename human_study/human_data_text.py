"""Build the Excel workbook for the Columbia Advanced Search Study.

Creates one workbook with N sheets (one per annotator), each pre-filled with N_PER_SHEET
product pairs drawn from the text dataset. Annotators fill in Q1-Q4; n_1 and n_2 tell them
how many attributes to list.

n_1 and n_2 are not random: they are the number of positive / negative constraints in the
synthetic query generated for the same product pair.

Usage (paths resolve relative to the repo root, so it runs from anywhere):
    python human_study/human_data_text.py                    # 10 sheets x 25 -> human_study/human_study.xlsx
    python human_study/human_data_text.py --sheets 4 --per-sheet 10 --out pilot.xlsx
"""

import argparse
import html
import json
import random
import re
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DEFAULT = REPO_ROOT / "dataset/feature-distance-dataset_gemini-2.5-flash_1000000_fixed_distance.jsonl"
OUT_DEFAULT = Path(__file__).resolve().parent / "human_study.xlsx"

# n_1 / n_2 are the number of attributes the annotator is asked for, in 0-3. They come from
# the synthetic query built for the same pair, so a human query is length-matched to the
# synthetic one it will be compared against.
MAX_ATTRS = 3

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
    """(n_1, n_2): positive and negative constraint counts of this pair's synthetic query."""
    return len(row["selected_pos_features"]), len(row["selected_neg_features"])


def usable(row, min_chars, max_chars):
    texts = [clean(row[k].get("product_text", "")) for k in ("positive_product", "hard_neg_product")]
    if not all(min_chars <= len(t) <= max_chars for t in texts):
        return False
    # Only pairs whose synthetic query has at most MAX_ATTRS constraints per side, so n_1 /
    # n_2 are real counts rather than clamped ones. Zero is allowed: it means the annotator
    # lists nothing on that side.
    return all(0 <= n <= MAX_ATTRS for n in attr_counts(row))


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


def write_sheet(ws, rows):
    ws.append(COLUMNS)
    header_fill = PatternFill("solid", fgColor="D9E1F2")
    for i, _ in enumerate(COLUMNS, start=1):
        cell = ws.cell(row=1, column=i)
        cell.font = Font(bold=True)
        cell.fill = header_fill
        cell.alignment = Alignment(vertical="center")
        ws.column_dimensions[get_column_letter(i)].width = COL_WIDTHS[i - 1]

    for row in rows:
        n_1, n_2 = attr_counts(row)
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
    args = ap.parse_args()

    needed = args.sheets * args.per_sheet
    rows = load_rows(args.raw, needed, args.seed, args.min_chars, args.max_chars)

    wb = Workbook()
    wb.remove(wb.active)
    for s in range(args.sheets):
        ws = wb.create_sheet(f"Annotator {s + 1}")
        write_sheet(ws, rows[s * args.per_sheet:(s + 1) * args.per_sheet])
    wb.save(args.out)
    print(f"wrote {args.out}: {args.sheets} sheets x {args.per_sheet} examples = {needed} pairs")


if __name__ == "__main__":
    main()
