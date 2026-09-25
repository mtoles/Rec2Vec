"""Offline loading of paper.sh's test-set predictions for the paper's plots.

paper.sh writes <run_dir>/preds/{queries,corpus,triplets}.jsonl for every condition so
that all analysis runs on CPU, without re-encoding anything. Every test-row model is also
scored on the human-written queries into <run_dir>/preds_human/ (same files; pass
preds_subdir="preds_human" to the loaders below). This module turns those files into one
tidy per-query DataFrame:

    modality  style  query_kind  V  run_dir  query_id  recall@1  recall@5  recall@50
    query_distance  num_attributes

The expected condition list is parsed out of paper.sh's own CONDITIONS table rather than
duplicated here, so adding a row there is enough to make it show up in the analysis.
"""

import json
import os
import re

import pandas as pd
from pathlib import Path
from utils.training_profile import ROOT, analysis_profiles, condition_source, training_profile
from utils.training_plan import RETIRED_INFONCE_STYLES

DEFAULT_KS = (1, 5, 50)

# discover_runs column that says whether a preds subdir has been written.
HAS_PREDS_COLUMN = {"preds": "has_preds", "preds_human": "has_human_preds"}

# ---------------------------------------------------------------------------
# Condition table
# ---------------------------------------------------------------------------

_CONDITIONS_BLOCK = re.compile(r'^CONDITIONS="\s*$(.*?)^"\s*$', re.M | re.S)


def parse_conditions(paper_sh="paper.sh"):
    """The condition table paper.sh drives itself from: modality/style/query_kind/V/extra."""
    if Path(paper_sh).resolve() == ROOT / 'paper.sh' and 'PAPER_CONDITIONS_FILE' not in os.environ:
        profiles = analysis_profiles()
        if profiles:
            frames = [filter_profile_frame(_parse_condition_file(Path(p['conditions'])), p)
                      for p in profiles]
            return pd.concat(frames, ignore_index=True).drop_duplicates()
    return _parse_condition_file(condition_source(paper_sh)).drop_duplicates()


def filter_profile_frame(frame, profile):
    if 'modalities' in profile:
        frame = frame[frame['modality'].isin(profile['modalities'])]
    if 'include_negs' in profile:
        frame = frame[frame['negs'].isin(profile['include_negs'])]
    if 'include_styles' in profile:
        frame = frame[frame['style'].isin(profile['include_styles'])]
    if 'exclude_styles' in profile:
        frame = frame[~frame['style'].isin(profile['exclude_styles'])]
    return frame.copy()


def _parse_condition_file(source):
    text = source.read_text()
    match = _CONDITIONS_BLOCK.search(text)
    block = match.group(1) if match else text

    rows = []
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) != 5:
            raise ValueError(f"Malformed condition line (want 5 fields): {line!r}")
        modality, style, query_kind, v, extra = fields
        if style in RETIRED_INFONCE_STYLES:
            raise ValueError(f"Retired InfoNCE style {style}; use infonce-ours-v3")
        tokens = [] if extra == "-" else extra.split(",")
        negs = [t[len("negs="):] for t in tokens if t.startswith("negs=")]
        mining = [t[len("mining="):] for t in tokens if t.startswith("mining=")]
        seed = [int(t[len("seed="):]) for t in tokens if t.startswith("seed=")]
        rephrase = [t[len("rephrase="):] for t in tokens if t.startswith("rephrase=")]
        order = [t[len("order="):] for t in tokens if t.startswith("order=")]
        rows.append({
            "modality": modality,
            "style": style,
            "query_kind": query_kind,
            "V": None if v == "-" else int(v),
            "extra": None if extra == "-" else extra,
            # Which hard negatives the row trains on: the dataset's labeled ones, or the
            # retrieval-mined sibling built by mine_hard_negs.py (paper.sh negs=mined).
            "negs": negs[0] if negs else "labeled",
            # mine_hard_negs.py --variant suffix (mining sweep); "" is the default config.
            "mining": mining[0] if mining else "",
            # Training seed of the trial; 42 is the trainer default and carries no token.
            "seed": seed[0] if seed else 42,
            # rephrase_dataset.py --in-context sibling ("in-context"); "" is the plain rephrasing.
            "rephrase": rephrase[0] if rephrase else "",
            # train.py --train-order ("mined-first"); "" is one shuffle over all rows.
            "order": order[0] if order else "",
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Run directories
# ---------------------------------------------------------------------------

def active_dataset_bases(paper_sh="paper.sh"):
    """Default dataset identities used by the current paper, excluding old experiments."""
    profile = training_profile() if Path(paper_sh).resolve() == ROOT / "paper.sh" else {}
    if profile:
        return {mod: Path(base).name for mod, base in profile["bases"].items()}
    source = open(paper_sh, encoding="utf-8").read()
    bases = {}
    for modality, variable in (("text", "TEXT_DATASET"), ("multimodal", "IMG_DATASET")):
        match = re.search(variable + r"=\$\{" + variable + r":-([^}]+)\}", source)
        if match is None:
            raise ValueError(f"{paper_sh} does not declare {variable}")
        bases[modality] = os.path.basename(match[1].rstrip("/"))
    return bases


def parse_run_name(name):
    """Inverse of paper.sh's run_name_for / utils.run_naming.build_run_name.

    'text__all-mpnet-base-v2__ours-mse__<dataset>__synthetic__V-40__note-paper'
    Dataset tags never contain '__', so splitting on it is unambiguous.
    """
    parts = name.split("__")
    if len(parts) < 6 or not parts[-1].startswith("note-"):
        return None
    out = {
        "modality": parts[0],
        "model_short": parts[1],
        "style": parts[2],
        "dataset_tag": parts[3],
        "query_kind": parts[4],
        "note": parts[-1][len("note-"):],
        "V": None,
        "easy": None,
        "transform": None,
        "seed": 42,
        "rephrase": "",
        "order": "",
    }
    for token in parts[5:-1]:
        key, _, value = token.partition("-")
        if key == "V":
            out["V"] = int(value)
        elif key in ("easy", "transform", "order"):
            out[key] = value
        elif key == "seed":
            out["seed"] = int(value)
    # The mined sibling dataset carries a `_mined-<kind>` suffix; nothing else in the name
    # says which negatives the run trained on.
    tag = out["dataset_tag"]
    # <base>_rephrased-<variant>...: the rephrasing prompt variant (rephrase= in paper.sh).
    if "_rephrased-" in tag:
        out["rephrase"] = tag.split("_rephrased-", 1)[1].split("_", 1)[0]
    # <dataset>_mixed-<kind>[_<variant>]: the mix_hard_negs.py sibling, half labeled and half
    # mined train negatives; the variant is the mining config of the mined half.
    if "_mixed-" in tag:
        out["negs"] = "mixed"
        out["mining"] = tag.split("_mixed-", 1)[1].partition("_")[2]
        return out
    # New metadata-matched Baseline has a distinct tag from the legacy random control.
    if "_baseline-bm25-" in tag:
        out["negs"] = "baseline-bm25"
        out["mining"] = ""
        return out
    if "_baseline-v3-" in tag:
        out["negs"] = "baseline-v3"
        out["mining"] = ""
        return out
    if "_baseline-" in tag:
        out["negs"] = "baseline"
        out["mining"] = ""
        return out
    # <dataset>_random-<kind>: the random_hard_negs.py sibling, hard negatives replaced at random.
    if "_random-" in tag:
        out["negs"] = "random"
        out["mining"] = ""
        return out
    out["negs"] = ("mined-graded" if tag.endswith("_graded") else "mined") if "_mined-" in tag else "labeled"
    # <dataset>_mined-<kind>[_<variant>][_graded]: the variant is the mining-sweep config.
    out["mining"] = ""
    if "_mined-" in tag:
        rest = tag.split("_mined-", 1)[1].removesuffix("_graded")
        out["mining"] = rest.partition("_")[2]
    return out


def discover_runs(models_root="models", note="paper"):
    """Every run directory of this NOTE, with its parsed config and preds availability.

    models/old/ (runs retired by hand after a change invalidated them) does not parse as a
    run name and is skipped.
    """
    rows = []
    for name in sorted(os.listdir(models_root)):
        run_dir = os.path.join(models_root, name)
        if not os.path.isdir(run_dir):
            continue
        parsed = parse_run_name(name)
        if parsed is None or parsed["note"] != note:
            continue
        parsed["run_dir"] = run_dir
        for subdir, column in HAS_PREDS_COLUMN.items():
            parsed[column] = os.path.isfile(os.path.join(run_dir, subdir, "queries.jsonl"))
        rows.append(parsed)
    return pd.DataFrame(rows)


def discover_profile_runs():
    """Discover each configured run with its own model root and experiment tag."""
    profiles = analysis_profiles()
    if not profiles:
        raise ValueError('No resolved analysis profiles')
    frames = []
    for profile in profiles:
        frame = filter_profile_frame(discover_runs(profile['models_root'], profile['note']), profile)
        frame['reference'] = 'reference' in profile and profile['reference']
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def match_conditions(conditions, runs):
    """Left-join the expected conditions onto the discovered runs.

    `run_dir` is NaN for a condition paper.sh has not produced yet; `has_preds` is False
    for one that trained but whose inference failed or was skipped.
    """
    keys = ["modality", "style", "query_kind", "V", "negs"]
    merged = conditions.merge(runs, on=keys, how="left", suffixes=("", "_run"))
    for column in HAS_PREDS_COLUMN.values():
        merged[column] = merged[column].fillna(False).astype(bool)
    return merged


# ---------------------------------------------------------------------------
# Query attribute count
# ---------------------------------------------------------------------------

# Synthetic/rephrased queries render as
#   'I am looking for: "<item>" that has: A, B; and does not have: C, D'
# with the verbs varying by rephrasing ('that features:', '; and without:', ...).
_NEG_CLAUSE = re.compile(r";\s*and (?:does not|doesn't|do not|don't|without)[^:]*:", re.I)
_POS_CLAUSE = re.compile(r"\b(?:that|which|featuring|includes|including|with)\b[^:]*:", re.I)


def count_query_attributes(query):
    """Attributes explicitly listed in a query, or None if it is not an attribute query.

    Validated against the image dataset's exact selected_* feature lists: agrees on
    2995/3000 sampled rows. Text datasets carry no selected_* columns, so parsing the
    rendered query is the only definition available on both modalities.
    """
    if not query or not query.strip():
        return None
    parts = _NEG_CLAUSE.split(query, maxsplit=1)
    head, tail = parts[0], (parts[1] if len(parts) > 1 else "")
    match = _POS_CLAUSE.search(head)
    if match is None and not tail:
        return None  # keyword query ('fv relay') -- no attributes to count
    positive = head[match.end():] if match else ""
    total = 0
    for clause in (positive, tail):
        total += len([s for s in (x.strip(" .;\t\n") for x in clause.split(",")) if s])
    return total or None


# ---------------------------------------------------------------------------
# Preds -> per-query metrics
# ---------------------------------------------------------------------------

def _read_jsonl(path):
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_run_preds(run_dir, ks=DEFAULT_KS, preds_subdir="preds"):
    """Per-query recall@k plus the two query covariates, straight from preds/.

    query_distance is per triplet row, not per query: every query has one row against a
    hard/substitute negative carrying the real distance and one against a random negative
    carrying a placeholder. Only the hard rows are averaged into the per-query value.

    The placeholder differs by pipeline and must not be filtered on its value: preprocess_text
    writes -1 for random negatives, preprocess_images writes max_distance*2 (= 20), which is a
    positive number larger than any real distance. Filtering on negative_example_source is the
    only rule that is correct for both.
    """
    preds = os.path.join(run_dir, preds_subdir)
    queries = _read_jsonl(os.path.join(preds, "queries.jsonl"))
    # The human eval sets carry no hard negative (download_human_labels.py drops the columns:
    # nothing establishes that the pair's synthetic negative fails the query a person wrote),
    # so test.py writes no triplets.jsonl for them and query_distance is undefined.
    triplets_path = os.path.join(preds, "triplets.jsonl")
    triplets = _read_jsonl(triplets_path) if os.path.exists(triplets_path) else []

    max_k = max(ks)
    rows = []
    for q in queries:
        positives = set(q["positive_corpus_ids"])
        ranked = [int(c) for c, _ in q["top_k"][:max_k]]
        row = {
            "query_id": q["query_id"],
            "query": q["query"],
            "n_positives": len(positives),
            "num_attributes": count_query_attributes(q["query"]),
        }
        for k in ks:
            hits = positives.intersection(ranked[:k])
            row[f"recall@{k}"] = len(hits) / len(positives) if positives else float("nan")
        rows.append(row)
    frame = pd.DataFrame(rows)
    if not triplets:
        frame["query_distance"] = float("nan")
        return frame

    distance = pd.DataFrame([
        {"query_id": t["query_id"],
         "query_distance": t.get("query_distance"),
         "negative_example_source": t.get("negative_example_source")}
        for t in triplets
    ]).dropna(subset=["query_distance"])
    if distance["negative_example_source"].notna().any():
        distance = distance[distance["negative_example_source"] != "random"]
    else:  # older preds without the passthrough column: fall back to the text sentinel
        distance = distance[distance["query_distance"] >= 0]
    per_query = distance.groupby("query_id")["query_distance"].mean()
    return frame.merge(per_query.rename("query_distance"), on="query_id", how="left")


def _row_order_keys(run_dir):
    """(positive_id, negative_id) per triplet row, in the source dataset's row order."""
    triplets = _read_jsonl(os.path.join(run_dir, "preds", "triplets.jsonl"))
    return triplets


def borrow_attributes(target_run_dir, reference_run_dir):
    """query_id -> num_attributes for a run whose own queries carry no attributes.

    Text `original` queries are bare keyword strings ('fv relay'), so the attribute count
    has to come from the same rows' synthetic rendering. test_text.py writes triplets.jsonl
    in source-dataset row order and the text test split does not depend on query_kind, so
    row i of the two runs is the same underlying example -- asserted here on the product ids
    rather than assumed.
    """
    target, reference = _row_order_keys(target_run_dir), _row_order_keys(reference_run_dir)
    if len(target) != len(reference):
        raise ValueError(
            f"Row counts differ ({len(target)} vs {len(reference)}); cannot align "
            f"{target_run_dir} with {reference_run_dir}")
    mismatched = sum(
        1 for a, b in zip(target, reference)
        if (a.get("positive_id"), a.get("negative_id")) != (b.get("positive_id"), b.get("negative_id"))
    )
    if mismatched:
        raise ValueError(f"{mismatched}/{len(target)} rows misaligned between "
                         f"{target_run_dir} and {reference_run_dir}")

    reference_attrs = load_run_preds(reference_run_dir, ks=(1,)).set_index("query_id")["num_attributes"]
    # A bare keyword query ('fv relay') is shared by many underlying examples with different
    # attribute counts, so the per-query value is their mean -- the same averaging the
    # per-query query_distance already gets.
    collected = {}
    for a, b in zip(target, reference):
        value = reference_attrs.get(b["query_id"])
        if pd.notna(value):
            collected.setdefault(a["query_id"], []).append(float(value))
    return {qid: sum(v) / len(v) for qid, v in collected.items()}


def load_all(matched, ks=DEFAULT_KS, attribute_reference=None):
    """Concatenate every condition that has preds into one tidy per-query frame.

    attribute_reference: {run_dir: reference_run_dir} for runs whose queries carry no
    attribute list (text `original`), resolved via borrow_attributes.
    """
    attribute_reference = attribute_reference or {}
    frames = []
    for row in matched.itertuples():
        if not row.has_preds:
            continue
        frame = load_run_preds(row.run_dir, ks=ks)
        reference = attribute_reference.get(row.run_dir)
        if reference is not None:
            mapping = borrow_attributes(row.run_dir, reference)
            frame["num_attributes"] = frame["query_id"].map(mapping)
        frame["modality"] = row.modality
        frame["style"] = row.style
        frame["query_kind"] = row.query_kind
        frame["V"] = row.V
        frame["run_dir"] = row.run_dir
        frames.append(frame)
    if not frames:
        raise ValueError("No condition has preds yet -- run paper.sh first.")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

def staleness(run_dir, preds_subdir="preds"):
    """Why this run's preds must not be read as current, or "" if they may.

    Synthetic predictions retain the model-marker check. Human predictions also
    require a matched-data signature covering the human set, retrieval corpus,
    model payload, and inference code; spreadsheet edits therefore invalidate them.
    """
    if preds_subdir == "preds_human":
        from utils.human_freshness import prediction_problem
        problem = prediction_problem(run_dir)
        if problem:
            return problem
    model = os.path.join(run_dir, "final", "modules.json")
    if not os.path.isfile(model):
        return "no model"
    meta = json.load(open(os.path.join(run_dir, preds_subdir, "meta.json")))
    if abs(meta["model_mtime"] - os.path.getmtime(model)) > 1.0:
        return "stale: preds were made from an earlier model"
    return ""


def health_check(matched, min_queries=100, preds_subdir="preds"):
    """One row per condition with preds, flagging stale or degenerate inference output.

    preds_subdir="preds_human" checks the human-query preds of the same conditions; the
    human set is far smaller than the test split, so pass a lower min_queries with it.

    A condition whose queries are blank collapses to a single empty anchor and scores ~0 --
    which is what paper.sh's Phase 2 produced for every `rephrased` condition while it
    passed the non-rephrased dataset (whose rephrased_query column is empty). Such a run
    must be excluded from the plots rather than drawn as a real result. Stale preds
    (see `staleness`) are excluded the same way.
    """
    rows = []
    has_column = HAS_PREDS_COLUMN[preds_subdir]
    for row in matched.itertuples():
        if not getattr(row, has_column):
            rows.append({"modality": row.modality, "style": row.style,
                         "query_kind": row.query_kind, "V": row.V, "run_dir": row.run_dir,
                         "n_queries": 0, "n_blank": 0, "healthy": False,
                         "problem": f"no {preds_subdir} (train or test failed)"})
            continue
        problem = staleness(row.run_dir, preds_subdir=preds_subdir)
        frame = load_run_preds(row.run_dir, ks=(1,), preds_subdir=preds_subdir)
        blank = int((frame["query"].fillna("").str.strip() == "").sum())
        if problem:
            pass
        elif blank:
            problem = f"{blank} blank queries -- wrong dataset for this query_kind"
        elif len(frame) < min_queries:
            problem = f"only {len(frame)} queries"
        rows.append({"modality": row.modality, "style": row.style,
                     "query_kind": row.query_kind, "V": row.V, "run_dir": row.run_dir,
                     "n_queries": len(frame), "n_blank": blank,
                     "healthy": not problem, "problem": problem})
    return pd.DataFrame(rows)


def human_conditions():
    """Resolved main conditions for a new run; historical declaration otherwise."""
    if training_profile():
        frame = parse_conditions(ROOT / 'paper.sh')
        return frame[~frame.extra.fillna('').str.contains('split=val')].drop_duplicates().copy()
    return pd.read_csv(ROOT / 'analysis/migrations/in_context_20260920/human_conditions.csv',
                       keep_default_na=False)
