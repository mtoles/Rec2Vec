"""Build the Excel workbook for the image half of the Columbia Advanced Search Study.

Same layout as human_data_text.py, but the products are DeepFashion In-shop garments, so each
product gets a description cell (categories, color, seller copy) followed by its photo.

n_1 and n_2 are not random: they are the number of positive / negative constraints in the
synthetic query generated for the same product pair, capped at MAX_ATTRS.

Usage (paths resolve relative to the repo root, so it runs from anywhere):
    python human_study/human_data_image.py                 # 10 sheets x 25 -> human_study/human_study_image.xlsx
    python human_study/human_data_image.py --sheets 4 --per-sheet 10 --out pilot.xlsx
"""

import argparse
import json
import random
import tempfile
from pathlib import Path

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XLImage
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter
from PIL import Image as PILImage

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DEFAULT = REPO_ROOT / "dataset/deepfashion-inshop-image-triplets_hf_20000.jsonl"
OUT_DEFAULT = Path(__file__).resolve().parent / "human_study_image.xlsx"

COLUMNS = ["Product 1", "Product 1 Image", "Product 2", "Product 2 Image",
           "n_1", "Q1", "n_2", "Q2", "Q3", "Q4", "Problem?"]
COL_WIDTHS = [46, 34, 46, 34, 6, 30, 6, 30, 36, 36, 14]

# n_1 / n_2 are the number of attributes the annotator is asked for, in 0-MAX_ATTRS.
MAX_ATTRS = 3
IMG_PX = 224          # DeepFashion In-shop crops are 224x224
ROW_HEIGHT_PT = 210   # ~280 px: fits the 224 px image and the seller copy beside it
COL_WIDTH_IMG = 34    # ~238 px at the default font


def attr_counts(row):
    """(n_1, n_2): positive and negative constraint counts of this pair's synthetic query."""
    return len(row["selected_pos_features"]), len(row["selected_neg_features"])


def image_path(product):
    p = Path(product["image_path"])
    return p if p.is_absolute() else REPO_ROOT / p


def usable(row):
    if not all(0 <= n <= MAX_ATTRS for n in attr_counts(row)):
        return False
    return all(image_path(row[k]).exists() for k in ("positive_product", "hard_neg_product"))


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


def write_sheet(ws, rows, tmpdir, cache):
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
    args = ap.parse_args()

    needed = args.sheets * args.per_sheet
    pool = load_pool(args.raw, needed, args.seed)
    sheets = deal_sheets(pool, args.sheets, args.per_sheet, args.category_cap)

    wb = Workbook()
    wb.remove(wb.active)
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = {}
        for s, rows in enumerate(sheets, start=1):
            ws = wb.create_sheet(f"Annotator {s}")
            write_sheet(ws, rows, tmpdir, cache)
        wb.save(args.out)
    print(f"wrote {args.out}: {args.sheets} sheets x {args.per_sheet} examples = {needed} pairs")


if __name__ == "__main__":
    main()
