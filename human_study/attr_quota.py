"""Attribute-count quotas that make the human study match the synthetic distribution.

The first study asked for 0-3 *differentiating* attributes per side, so annotators wrote about
3 attributes per query against about 10 stated in the synthetic queries. Closing that gap means
reproducing the synthetic joint distribution over (n_1, n_2) -- every attribute the query
requires and every attribute it excludes -- rather than only its mean. Those stated counts are
the only attribute count the two sets share: the human set carries no hard negative, so a
differentiating count is not defined on it.

For a target size M the quota per (n_1, n_2) cell is M * p_synthetic(cell), apportioned by
largest remainder so the cells sum to M. Rows from the first study fill min(already labeled,
target) of each cell and the rest is what has to be labeled now. Rows in cells the synthetic
distribution never produces are left out of the matched set: every synthetic query excludes at
least one attribute, so n_2 = 0 is unreachable, and about a third of the first study sits there.

Fixing a window of attribute counts instead -- say 4-10 per side -- matches the mean at the same
labeling cost but leaves the combined set bimodal, because the new rows land entirely above the
old ones. The mean is set by the new rows' average alone, so their shape is free; spending it on
the synthetic histogram costs nothing extra.
"""

import re
from collections import Counter
from pathlib import Path

from datasets import load_from_disk

ATTR_SPLIT = re.compile(r"[\n,;]+")


def written_counts(row):
    """(n_1, n_2) an annotator actually wrote, from the comma-separated Q1 / Q2 cells."""
    return tuple(len([p for p in ATTR_SPLIT.split(str(row[key])) if p.strip()])
                 for key in ("human_pos_attributes", "human_neg_attributes"))


def labeled(path, id_column):
    """(histogram over (n_1, n_2), product ids) of the rows labeled in the first study.

    Returns empty containers when the dataset is absent, so the first backfill run of a new
    modality builds its quota from the synthetic distribution alone.
    """
    if not Path(path).exists():
        return Counter(), set()
    rows = load_from_disk(str(path))
    return Counter(written_counts(r) for r in rows), set(rows[id_column])


def apportion(weights, total):
    """Largest-remainder apportionment of `total` units across `weights`."""
    scaled = {k: w * total for k, w in weights.items()}
    whole = {k: int(v) for k, v in scaled.items()}
    order = sorted(scaled, key=lambda k: scaled[k] - whole[k], reverse=True)
    for k in order[:total - sum(whole.values())]:
        whole[k] += 1
    return Counter({k: v for k, v in whole.items() if v})


def build(synthetic, already, target_rows):
    """(quota, keep): rows to label at each cell, and rows the first study contributes there."""
    total = sum(synthetic.values())
    target = apportion({k: v / total for k, v in synthetic.items()}, target_rows)
    keep = Counter({k: min(already[k], target[k]) for k in target})
    return Counter({k: target[k] - keep[k] for k in target}), keep


def summarize(synthetic, already, keep, quota):
    """Lines describing the shift from the first study's distribution to the matched one."""
    def stats(hist):
        n = sum(hist.values())
        if not n:
            return 0, 0.0, "-"
        totals = [a + b for a, b in hist.elements()]
        return n, sum(totals) / n, f"{min(totals)}-{max(totals)}"

    matched = keep + quota
    rows = [("synthetic", synthetic), ("first study", already),
            ("first study kept", keep), ("to label", quota), ("matched set", matched)]
    lines = [f"  {'set':<18} {'rows':>6} {'mean attrs':>11} {'range':>8}"]
    for name, hist in rows:
        n, mean, span = stats(hist)
        lines.append(f"  {name:<18} {n:>6} {mean:>11.2f} {span:>8}")
    total = sum(synthetic.values())

    def distance(hist):
        n = sum(hist.values())
        return 0.5 * sum(abs(hist[k] / n - synthetic[k] / total)
                         for k in set(hist) | set(synthetic))

    lines.append(f"  total variation from synthetic: first study {distance(already):.3f} "
                 f"-> matched set {distance(matched):.3f}")
    lines.append(f"  {sum(already.values()) - sum(keep.values())} of {sum(already.values())} "
                 f"first-study rows fall outside the matched set and are not reused")
    return "\n".join(lines)
