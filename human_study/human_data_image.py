"""Build the Excel workbook for the image half of the Columbia Advanced Search Study.

Same layout as human_data_text.py, but the products are DeepFashion In-shop garments, so each
product gets a description cell (categories, color, seller copy) followed by its photo.

n_1 and n_2 are not random: they are the number of positive / negative constraints in the
synthetic query generated for the same product pair, capped at MAX_ATTRS.

Usage (paths resolve relative to the repo root, so it runs from anywhere):
    python human_study/human_data_image.py                 # 10 sheets x 25 -> human_study/human_study_image.xlsx
    python human_study/human_data_image.py --sheets 4 --per-sheet 10 --out pilot.xlsx
    python human_study/human_data_image.py --backfill       # quota-sampled second study
"""

import argparse
import json
import math
import random
import tempfile
from collections import Counter
from pathlib import Path

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from PIL import Image as PILImage

import attr_quota

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DEFAULT = REPO_ROOT / "dataset/deepfashion-inshop-image-triplets_hf_20000.jsonl"
OUT_DEFAULT = Path(__file__).resolve().parent / "human_study_image.xlsx"
LABELED_DEFAULT = REPO_ROOT / "dataset/processed/deepfashion-inshop-image-triplets_hf_20000_human"

COLUMNS = ["Product 1", "Product 1 Image", "Product 2", "Product 2 Image",
           "n_1", "Q1", "n_2", "Q2", "Q3", "Q4", "Problem?"]
COL_WIDTHS = [46, 34, 46, 34, 6, 30, 6, 30, 36, 36, 14]

# n_1 / n_2 are the number of attributes the annotator is asked for, in 0-MAX_ATTRS.
MAX_ATTRS = 3
# --backfill asks for the counts the synthetic query states rather than the differentiating
# ones the first study capped at MAX_ATTRS, and draws each (n_1, n_2) cell to a quota that makes
# the combined set reproduce the synthetic distribution. See attr_quota for why.
BACKFILL_TARGET_ROWS = 300
IMG_PX = 224          # DeepFashion In-shop crops are 224x224
ROW_HEIGHT_PT = 210   # ~280 px: fits the 224 px image and the seller copy beside it
COL_WIDTH_IMG = 34    # ~238 px at the default font


def attr_counts(row):
    """(n_1, n_2): differentiating constraint counts of this pair's synthetic query."""
    return len(row["selected_pos_features"]), len(row["selected_neg_features"])


def stated_counts(row):
    """(n_1, n_2): every attribute the synthetic query states, per side. The rendered query
    lists selected_pos + selected_common as required and selected_neg + selected_neither as
    excluded, so these are the counts a human query has to match to be length-comparable."""
    return (len(row["selected_pos_features"]) + len(row["selected_common_features"]),
            len(row["selected_neg_features"]) + len(row["selected_neither_features"])) 


def image_path(product):
    p = Path(product["image_path"])
    return p if p.is_absolute() else REPO_ROOT / p


def has_images(row):
    return all(image_path(row[k]).exists() for k in ("positive_product", "hard_neg_product"))


def usable(row):
    return has_images(row) and all(n <= MAX_ATTRS for n in attr_counts(row))


def load_pool(path, needed, seed):
    """Usable rows, deduped by product so no garment is shown twice across the workbook."""
    with open(path) as fh:
        raw = [json.loads(line) for line in fh]
    rng = random.Random(seed)
    rng.shuffle(raw)
    picked, seen_products = [], set()
    for row in raw:
        if len(picked) == needed:
            break
        pid, nid = row["positive_product_id"], row["hard_negative_product_id"]
        if pid in seen_products or nid in seen_products or not usable(row):
            continue
        seen_products.update((pid, nid))
        picked.append(row)
    if len(picked) < needed:
        raise SystemExit(f"only {len(picked)} usable rows found, need {needed}")
    return picked


def load_quota_pool(path, seed, quota, exclude_products):
    """Sample `quota[(n_1, n_2)]` rows at each stated-attribute cell.

    Same product dedup as load_pool, plus the garments of the first study: one shown twice
    across the two studies would appear twice in the combined evaluation set.
    """
    with open(path) as fh:
        raw = [json.loads(line) for line in fh]
    rng = random.Random(seed)
    rng.shuffle(raw)
    picked, seen_products, taken = [], set(exclude_products), Counter()
    for row in raw:
        cell = stated_counts(row)
        if taken[cell] >= quota[cell]:
            continue
        pid, nid = row["positive_product_id"], row["hard_negative_product_id"]
        if pid in seen_products or nid in seen_products or not has_images(row):
            continue
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
    left unfinished rather than emptying one corner of the distribution. This replaces the
    category cap: the cap balanced garment types within a sheet, and cell order does not track
    category."""
    ordered = sorted(rows, key=stated_counts)
    rng = random.Random(seed)
    dealt = [ordered[i::sheets] for i in range(sheets)]
    for sheet in dealt:
        rng.shuffle(sheet)
    return dealt


def deal_sheets(pool, sheets, per_sheet, cat_cap):
    """Deal the pool into sheets, capping how many pairs of one garment category a single
    annotator sees. DeepFashion In-shop has only 16 categories and 28% of it is tees."""
    remaining = list(pool)
    dealt = []
    for _ in range(sheets):
        sheet, counts, skipped = [], {}, []
        while remaining and len(sheet) < per_sheet:
            row = remaining.pop(0)
            cat = row["item"]
            if counts.get(cat, 0) >= cat_cap:
                skipped.append(row)
                continue
            counts[cat] = counts.get(cat, 0) + 1
            sheet.append(row)
        while len(sheet) < per_sheet and skipped:      # cap is a preference, not a hard limit
            sheet.append(skipped.pop(0))
        remaining = skipped + remaining
        dealt.append(sheet)
    return dealt


def describe(product, shopping_for=None):
    """Seller copy for the garment: categories, color, then the `||`-separated bullets."""
    header = f"SHOPPING FOR: {shopping_for}\n\n" if shopping_for else ""
    cats = " / ".join(
        str(product.get(k, "")).title() for k in ("category1", "category2", "category3")
        if product.get(k)
    )
    lines = [cats]
    if product.get("color"):
        lines.append(f"Color: {product['color']}")
    body = [part.strip() for part in (product.get("text") or "").split("||")]
    lines.append("")
    lines.extend(part for part in body if part)
    lines.append("")
    lines.append(f"view: {product['image_id'].split('_')[-1]}   id: {product['product_id']}")
    return header + "\n".join(lines)


def as_png(product, tmpdir, cache):
    """openpyxl cannot embed .webp, so convert once per image and reuse."""
    src = image_path(product)
    if src in cache:
        return cache[src]
    dst = Path(tmpdir) / f"{product['image_id']}.png"
    with PILImage.open(src) as im:
        im.convert("RGB").save(dst, "PNG")
    cache[src] = dst
    return dst


def write_sheet(ws, rows, tmpdir, cache, counts):
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
            describe(row["positive_product"], shopping_for=row["item"]),
            "",    # Product 1 Image
            describe(row["hard_neg_product"]),
            "",    # Product 2 Image
            n_1,   # attributes Product 1 has that Product 2 lacks
            "",    # Q1
            n_2,   # attributes Product 2 has that Product 1 lacks
            "",    # Q2
            "",    # Q3
            "",    # Q4
            "",    # Problem?
        ])
        r = ws.max_row
        for col, key in ((2, "positive_product"), (4, "hard_neg_product")):
            img = XLImage(as_png(row[key], tmpdir, cache))
            img.width = img.height = IMG_PX
            img.anchor = f"{get_column_letter(col)}{r}"
            ws.add_image(img)

    for r in range(2, ws.max_row + 1):
        ws.row_dimensions[r].height = ROW_HEIGHT_PT
        for c in range(1, len(COLUMNS) + 1):
            ws.cell(row=r, column=c).alignment = Alignment(wrap_text=True, vertical="top")
    ws.freeze_panes = "A2"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(RAW_DEFAULT), help="raw image-triplet jsonl")
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--sheets", type=int, default=10)
    ap.add_argument("--per-sheet", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--category-cap", type=int, default=6,
                    help="max pairs of one garment category per sheet")
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
        already, products = attr_quota.labeled(args.labeled, "positive_product_id")
        quota, keep = attr_quota.build(synthetic, already, args.target_rows)
        print(attr_quota.summarize(synthetic, already, keep, quota))
        handout = Counter({k: math.ceil(v * args.oversample) for k, v in quota.items()})
        pool = load_quota_pool(args.raw, args.seed, handout, products)
        sheets = deal_round_robin(pool, math.ceil(len(pool) / args.per_sheet), args.seed)
    else:
        counts = attr_counts
        pool = load_pool(args.raw, args.sheets * args.per_sheet, args.seed)
        sheets = deal_sheets(pool, args.sheets, args.per_sheet, args.category_cap)

    wb = Workbook()
    wb.remove(wb.active)
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = {}
        for s, rows in enumerate(sheets, start=1):
            write_sheet(wb.create_sheet(f"Annotator {s}"), rows, tmpdir, cache, counts)
        wb.save(out)
    print(f"wrote {out}: {len(sheets)} sheets, {len(pool)} pairs")


if __name__ == "__main__":
    main()
