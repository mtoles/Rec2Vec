"""Compare human and synthetic queries with their positive text documents.

Run: .venv/bin/python -B analysis/query_overlap.py
Uses current local human-matched and rephrased-in-context datasets.
Does not download annotations or run retrieval models.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datasets import load_from_disk
from rouge_score import rouge_scorer
from sacrebleu.metrics import BLEU
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "dataset/processed/feature-distance-dataset_gemini-2.5-flash_1000000_nolek"
KINDS = {"human": "Human", "rephrased": "Synthetic (rephrased)",
         "template": "Synthetic (template)"}
METRICS = ["query_tokens", "document_tokens", "bleu1", "bleu2", "bleu3", "bleu4",
           "unigram_precision", "bigram_precision", "rouge1_f1",
           "rougeL_precision", "rougeL_f1", "jaccard",
           "content_precision", "content_jaccard"]
DISPLAY_METRICS = ["query_tokens", "bleu3", "bleu4", "unigram_precision", "bigram_precision",
                   "rougeL_f1", "jaccard", "content_precision"]


class WordTokenizer:
    """Shared lowercase, punctuation-free tokenization; no stemming."""

    def tokenize(self, text):
        return re.findall(r"[a-z0-9]+", text.lower())


def precision(query_tokens, document_tokens, order=1):
    query = Counter(zip(*(query_tokens[i:] for i in range(order))))
    document = Counter(zip(*(document_tokens[i:] for i in range(order))))
    return 100 * sum((query & document).values()) / sum(query.values()) if query else 0.0


def jaccard(query_tokens, document_tokens):
    query, document = set(query_tokens), set(document_tokens)
    return 100 * len(query & document) / len(query | document) if query or document else 0.0


class OverlapScorer:
    def __init__(self):
        self.tokenizer = WordTokenizer()
        self.rouge = rouge_scorer.RougeScorer(
            ["rouge1", "rougeL"], use_stemmer=False, tokenizer=self.tokenizer)
        self.bleu = {n: BLEU(tokenize="none", smooth_method="exp",
                             effective_order=True, max_ngram_order=n) for n in (1, 2, 3, 4)}

    def score(self, query, document):
        q, d = self.tokenizer.tokenize(query), self.tokenizer.tokenize(document)
        if not q or not d:
            raise ValueError("Queries and positive documents must contain word tokens")
        content_q = [w for w in q if w not in ENGLISH_STOP_WORDS]
        content_d = [w for w in d if w not in ENGLISH_STOP_WORDS]
        rouge = self.rouge.score(document, query)
        return {
            "query_tokens": len(q), "document_tokens": len(d),
            **{f"bleu{n}": bleu.sentence_score(" ".join(q), [" ".join(d)]).score
               for n, bleu in self.bleu.items()},
            "unigram_precision": precision(q, d),
            "bigram_precision": precision(q, d, order=2),
            "rouge1_f1": 100 * rouge["rouge1"].fmeasure,
            "rougeL_precision": 100 * rouge["rougeL"].precision,
            "rougeL_f1": 100 * rouge["rougeL"].fmeasure,
            "jaccard": jaccard(q, d),
            "content_precision": precision(content_q, content_d),
            "content_jaccard": jaccard(content_q, content_d),
        }


def load_comparisons(human_path, synthetic_path):
    human_ds, synthetic_ds = (load_from_disk(str(p)) for p in (human_path, synthetic_path))
    human = human_ds.to_pandas()
    columns = ["positive_id", "positive_example", "nl_query", "rephrased_query", "split",
               "negative_example"]
    synthetic = synthetic_ds.select_columns(columns).to_pandas()
    valid = synthetic["rephrased_query"].str.strip().ne("") & synthetic["positive_example"].ne(
        synthetic["negative_example"])
    excluded_rows = int((~valid).sum())
    synthetic = synthetic.loc[valid].drop(columns="negative_example").drop_duplicates()
    keys = ["positive_id", "nl_query"]
    if synthetic.duplicated(keys).any():
        raise ValueError("Ambiguous synthetic source query after removing negative-row copies")
    if human["positive_id"].duplicated().any() or human["human_query"].duplicated().any():
        raise ValueError("Paired document bootstrap requires one human query per positive document")
    paired = human.drop(columns=["rephrased_query", "split"]).merge(
        synthetic.rename(columns={"positive_example": "synthetic_document", "split": "source_split"}),
        on=keys, how="left", validate="one_to_one", indicator=True)
    if not paired["_merge"].eq("both").all():
        raise ValueError("Some human queries have no exact (positive_id, nl_query) synthetic match")
    if not paired["positive_example"].eq(paired["synthetic_document"]).all():
        raise ValueError("Human and synthetic positives have different document text")
    if not human["split"].eq("test").all():
        raise ValueError("Expected evaluation-only human annotations")
    paired = paired.drop(columns=["_merge", "synthetic_document"])
    test = synthetic[synthetic["split"].eq("test")].copy()
    if test.empty or paired.empty:
        raise ValueError("Both human and synthetic test sets must be nonempty")
    if test["rephrased_query"].duplicated().any():
        raise ValueError("Synthetic test queries have multiple source pairs; define query weighting first")
    provenance = {
        "human_dataset": str(human_path), "synthetic_dataset": str(synthetic_path),
        "human_fingerprint": human_ds._fingerprint, "synthetic_fingerprint": synthetic_ds._fingerprint,
        "human_pairs": len(paired), "synthetic_test_pairs": len(test),
        "excluded_synthetic_rows_before_deduplication": excluded_rows,
        "paired_synthetic_source_splits": paired["source_split"].value_counts().to_dict(),
        "human_source_studies": paired["source_study"].value_counts().to_dict(),
    }
    return paired, test, provenance


def score_rows(frame, query_column, kind, cohort, scorer):
    records, queries, documents = [], [], []
    for row in frame.to_dict("records"):
        query, document = row[query_column], row["positive_example"]
        records.append({"cohort": cohort, "kind": kind, "positive_id": row["positive_id"],
                        "query": query, "source_study": row["source_study"] if cohort == "paired" else "",
                        "annotator": row["annotator"] if cohort == "paired" else "",
                        **scorer.score(query, document)})
        queries.append(" ".join(scorer.tokenizer.tokenize(query)))
        documents.append(" ".join(scorer.tokenizer.tokenize(document)))
    corpus = {"cohort": cohort, "kind": kind, "n": len(records)}
    for n, bleu in scorer.bleu.items():
        score = bleu.corpus_score(queries, [documents])
        corpus[f"bleu{n}"] = score.score
        corpus[f"bleu{n}_signature"] = str(bleu.get_signature())
        corpus["brevity_penalty"] = score.bp
        corpus["query_document_length_ratio"] = score.sys_len / score.ref_len
    print(f"Scored {cohort}/{kind}: {len(records):,} query–document pairs", flush=True)
    return pd.DataFrame(records), corpus


def paired_differences(scores, samples=10000, seed=42):
    rng, records = np.random.default_rng(seed), []
    paired = scores[scores["cohort"].eq("paired")]
    groups = [("all", paired)] + list(paired.groupby("source_study", sort=True))
    for study, group in groups:
        human = group[group["kind"].eq("human")].set_index("positive_id")
        indices = rng.integers(len(human), size=(samples, len(human)))
        for kind in ("rephrased", "template"):
            synthetic = group[group["kind"].eq(kind)].set_index("positive_id").loc[human.index]
            for metric in METRICS:
                delta = human[metric].to_numpy() - synthetic[metric].to_numpy()
                lo, hi = np.quantile(delta[indices].mean(axis=1), [0.025, 0.975])
                records.append({"source_study": study, "synthetic_kind": kind, "metric": metric,
                                "n": len(human), "human_mean": human[metric].mean(),
                                "synthetic_mean": synthetic[metric].mean(), "difference": delta.mean(),
                                "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(records)


def plot_differences(differences, out):
    selected = ["unigram_precision", "content_precision", "bigram_precision",
                "rougeL_precision", "rougeL_f1", "jaccard"]
    labels = ["Unigram precision", "Content-word precision", "Bigram precision",
              "ROUGE-L precision", "ROUGE-L F1", "Token Jaccard"]
    data = differences[(differences["source_study"] == "all") &
                       (differences["synthetic_kind"] == "rephrased")].set_index("metric").loc[selected]
    fig, ax = plt.subplots(figsize=(8, 4))
    y, delta = np.arange(len(data)), data["difference"].to_numpy()
    ax.errorbar(delta, y, xerr=np.vstack([delta - data["ci_low"], data["ci_high"] - delta]),
                fmt="o", color="#28788e", capsize=4)
    ax.axvline(0, color="0.5", linewidth=1, linestyle="--")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("Human − synthetic rephrased (percentage points; paired 95% bootstrap CI)")
    ax.set_title(f"Query overlap with the same positive documents (n = {int(data['n'].iloc[0])})")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out / f"paired_differences.{ext}", dpi=180)
    plt.close(fig)


def write_report(out, summary, corpus, differences, provenance):
    paired = summary[summary["cohort"].eq("paired")].set_index("kind").loc[list(KINDS)]
    paired.index = [KINDS[k] for k in paired.index]
    primary = differences[(differences["source_study"] == "all") &
                          (differences["synthetic_kind"] == "rephrased")].set_index("metric")
    studies = differences[(differences["source_study"] != "all") &
                          (differences["synthetic_kind"] == "rephrased") &
                          differences["metric"].isin(["content_precision", "bigram_precision", "jaccard"])]
    text = f"""# Query overlap with positive text documents

{len(paired)} query variants on each of {provenance['human_pairs']} identical positive documents.
Human Q3 (`human_query`) is paired to the synthetic row by exact `(positive_id, nl_query)`;
document text must also agree. Synthetic means the in-context **rephrased** query used by
the paper; the unrephrased template is an additional control. Hard/easy-negative copies
are removed. Blank rephrasings and positive-equals-negative rows are excluded, as in evaluation.
Fixed in-context example products are already excluded from the human input.

## Paired means

All overlap scores are on a 0–100 scale; token counts are unscaled. BLEU here is the
**mean sentence BLEU**, not corpus BLEU.

{paired[['n'] + DISPLAY_METRICS].to_markdown(floatfmt='.3f')}

## Human minus synthetic rephrased

{primary.loc[DISPLAY_METRICS, ['difference', 'ci_low', 'ci_high']].to_markdown(floatfmt='.3f')}

95% percentile intervals resample paired documents {provenance['bootstrap_samples']:,} times
with seed {provenance['seed']}. They describe document sampling variation, conditional on
the collected annotators; they do not account for dependence within annotator or multiple
metric comparisons. Different queries can ask for different attributes even on the same document.

## Corpus BLEU (separate aggregation)

{corpus[['cohort', 'kind', 'n', 'bleu1', 'bleu2', 'bleu3', 'bleu4', 'brevity_penalty', 'query_document_length_ratio']].to_markdown(index=False, floatfmt='.6f')}

## Full synthetic test set (unpaired descriptive comparison)

{summary[['cohort', 'kind', 'n'] + DISPLAY_METRICS].to_markdown(index=False, floatfmt='.3f')}

The full-test comparison uses different documents and is therefore confounded by document
selection. The paired synthetic source splits are {provenance['paired_synthetic_source_splits']}.
The paired analysis measures wording only; it is not a held-out retrieval evaluation.

## Sensitivity by human-study wave

{studies[['source_study', 'metric', 'n', 'difference', 'ci_low', 'ci_high']].to_markdown(index=False, floatfmt='.3f')}

## Definitions and interpretation

- Candidate = query; single reference = the entire `positive_example` document supplied to retrieval.
- Shared tokenization: lowercase ASCII alphanumeric words (`[a-z0-9]+`), no stemming.
  BLEU receives these tokens with `tokenize=none`; its signature alone does not describe this preprocessing.
- [SacreBLEU](https://github.com/mjpost/sacrebleu) BLEU-1/2/3/4 use exponential smoothing,
  equal weights through the specified order, effective order for short queries, and the standard
  brevity penalty. Corpus scores aggregate n-gram counts; sentence scores are macro-averaged.
  Full-document references are much longer than queries, so BLEU is strongly length-penalized.
- Unigram/bigram precision = clipped matching query n-grams divided by all query n-grams,
  without a brevity penalty. Queries shorter than two tokens receive bigram precision zero.
- [ROUGE](https://github.com/google-research/google-research/tree/master/rouge) reports
  unigram F1 and longest-common-subsequence (ROUGE-L) precision/F1 against the full document.
- Jaccard = intersection / union of unique word sets. Content precision and content Jaccard
  remove scikit-learn's English stop words before scoring; empty content queries score zero.
- Greater precision indicates more document wording reused within the query; F1/Jaccard
  also depend on document coverage and query length. Query constraints, negation, spelling,
  and attribute selection can change overlap without reflecting paraphrasing effort.
  These are lexical measures, not evidence of annotator motivation or a causal explanation of recall.
- Images are excluded: their positive examples are image paths, not reference text.

`per_query.csv` contains all row scores; `paired_examples.csv` contains every paired query
and its actual document, and `by_annotator.csv` permits inspection of annotator variation.
`provenance.json` records input fingerprints, package versions, and BLEU signatures.
Rerun with `.venv/bin/python -B analysis/query_overlap.py`; existing outputs are archived in `old/`.
"""
    (out / "report.md").write_text(text)


def run(human_path, synthetic_path, out, samples=10000, seed=42):
    if samples < 1:
        raise ValueError("bootstrap_samples must be positive")
    paired, test, provenance = load_comparisons(human_path, synthetic_path)
    scorer, frames, corpora = OverlapScorer(), [], []
    for kind, column in [("human", "human_query"), ("rephrased", "rephrased_query"), ("template", "nl_query")]:
        frame, corpus = score_rows(paired, column, kind, "paired", scorer)
        frames.append(frame)
        corpora.append(corpus)
    frame, corpus = score_rows(test, "rephrased_query", "rephrased", "synthetic_test", scorer)
    frames.append(frame)
    corpora.append(corpus)
    scores, corpus = pd.concat(frames, ignore_index=True), pd.DataFrame(corpora)
    if not np.isfinite(scores[METRICS].to_numpy()).all():
        raise ValueError("Non-finite overlap metrics")
    summary = scores.groupby(["cohort", "kind"], sort=False)[METRICS].mean().reset_index()
    summary["n"] = scores.groupby(["cohort", "kind"], sort=False).size().to_numpy()
    differences = paired_differences(scores, samples, seed)
    provenance.update({"created_utc": datetime.now(timezone.utc).isoformat(),
                       "bootstrap_samples": samples, "seed": seed,
                       "versions": {name: version(name) for name in
                                    ["sacrebleu", "rouge-score", "scikit-learn", "datasets", "numpy"]},
                       "tokenization": "lowercase ASCII [a-z0-9]+, no stemming",
                       "bleu_signatures": {str(n): str(b.get_signature()) for n, b in scorer.bleu.items()}})
    if out.exists():
        archive = out.parent / "old" / (out.name + "_" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f"))
        archive.parent.mkdir(parents=True, exist_ok=True)
        out.rename(archive)
    out.mkdir(parents=True)
    scores.to_csv(out / "per_query.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    corpus.to_csv(out / "corpus_bleu.csv", index=False)
    differences.to_csv(out / "paired_differences.csv", index=False)
    scores[scores["cohort"].eq("paired")].groupby(
        ["source_study", "annotator", "kind"])[METRICS].agg(["count", "mean"]).to_csv(out / "by_annotator.csv")
    paired.drop(columns="human_query_alt").to_csv(out / "paired_examples.csv", index=False)
    (out / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    plot_differences(differences, out)
    write_report(out, summary, corpus, differences, provenance)
    print(summary[["cohort", "kind", "n"] + DISPLAY_METRICS].round(3).to_string(index=False))
    print(f"Results: {out / 'report.md'}", flush=True)
    return summary, differences


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--human-dataset", type=Path, default=Path(str(BASE) + "_human-matched-in-context"))
    parser.add_argument("--synthetic-dataset", type=Path, default=Path(str(BASE) + "_rephrased-in-context"))
    parser.add_argument("--out", type=Path, default=ROOT / "analysis/figs/query_overlap")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args.human_dataset, args.synthetic_dataset, args.out, args.bootstrap_samples, args.seed)


if __name__ == "__main__":
    main()
