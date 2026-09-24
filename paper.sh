#!/usr/bin/env bash
# paper.sh — in-context rephrased training/evaluation for every active paper experiment.
# EVAL_HUMAN=1 opts into the separate human study; off by default.
# Legacy condition rows are retained as comments; their checkpoints live in models/old/.
#
# Phase 1 trains every missing model, in parallel across GPUs (one GPU per job;
# DDP/NCCL is broken on this node). Phase 2 runs test-set inference for every
# condition via test.py, writing preds jsonl files into each model's run dir so all
# analysis can happen offline. Every test-split condition is also scored on the
# human-written queries (<dataset>_human, built by human_study/download_human_labels.py)
# into <run_dir>/preds_human/, ranked against the same test-split corpus as the row's own
# preds/ plus the human pairs' products; the human set is evaluation-only, so it rides
# along with every eval rather than being a condition of its own. This script does NOT
# preprocess data.
#
# Reuse rules:
#   - A model is reused whenever <run_dir>/final/modules.json exists.
#   - Preds are reused whenever their meta.json exists and is not older than the model
#     they were made from. Every preds dir carries meta.json with git SHA + mtimes.
#   - There is deliberately no mtime check against datasets or scripts (2026-09-10): an
#     edit to test.py that changes nothing about the outputs used to mark every preds dir
#     stale. Staleness is a decision, not a timestamp: when a change to a dataset, a loss
#     or a script invalidates runs, move those run dirs to $MODELS_ROOT/old/ (analysis
#     ignores that dir) or rerun with FORCE_TRAIN=1 / FORCE_TEST=1 and ONLY=<regex>.
#
# Knobs (env):
#   GPUS="0 1 2"   GPUs to schedule on (default: 0-7)
#   NOTE=paper     experiment tag; part of every run dir name
#   SMOKE=1        tiny data, models/_smoke root, NOTE=smoke, wandb offline
#   DRY_RUN=1      print the plan and exit
#   FORCE_TRAIN=1  retrain even if present    FORCE_TEST=1  re-infer even if present
#   ONLY=<regex>   plan only the rows whose run name matches (e.g. ONLY=_mined- to run the
#                  retrieval-mined baseline beside another instance without re-queuing its rows)
#   OMP_NUM_THREADS=12  CPU threads per training process (default 12; see the export below)

set -euo pipefail

# Bash reads a script incrementally from disk, so editing this file while a run is in
# flight corrupts the running instance (it resumes at a byte offset that has moved).
# Re-exec from an immutable snapshot: edits during a run are then always safe.
# The snapshot lives in /tmp, so the repo root must be passed through explicitly —
# dirname "$0" inside the child would resolve to /tmp.
if [[ -z ${PAPER_SNAPSHOT:-} ]]; then
  cd "$(dirname "$0")"
  export PAPER_ROOT=$PWD
  snapshot=$(mktemp "${TMPDIR:-/tmp}/paper.sh.XXXXXX")
  cat "$0" >"$snapshot"
  export PAPER_SNAPSHOT=$snapshot
  trap 'rm -f "$snapshot"' EXIT
  bash "$snapshot" "$@"
  exit $?
fi
cd "${PAPER_ROOT:?PAPER_ROOT unset — re-exec did not pass the repo root}"

PY=${PY:-./.venv/bin/python}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
SMOKE=${SMOKE:-0}
DRY_RUN=${DRY_RUN:-0}
FORCE_TRAIN=${FORCE_TRAIN:-0}
FORCE_TEST=${FORCE_TEST:-0}
# Human-study inference is opt-in and outside the synthetic paper refresh.
EVAL_HUMAN=${EVAL_HUMAN:-0}
ONLY=${ONLY:-}
# CPU threads per training process. Unset, torch spawns one per core (208 on emu) in every
# process; 8 such processes thrash the node (load 550, GPUs starved) and an image epoch takes
# 2h10m. Capped at 12 the same epoch takes 22m and text steps run ~2x faster (2026-09-11).
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-12} MKL_NUM_THREADS=${MKL_NUM_THREADS:-12}

# Styles named here are scheduled ahead of everything else, in this order. The condition table
# below is grouped by experiment, not by urgency, so this reorders the queue without moving any
# rows out of the section they belong to. Order follows the protocol: each family's graded
# member then its ungraded (mined) control, families in paper priority -- infonce, mse, siglip.
# ours-cosent is cosent with a third rank (positive > hard > random); CoSENT reads label order
# only, so it has no V/easy and no search -- main-grid rows only.
PRIORITY_STYLES=(infonce-ours-v3 ours-infonce-margin infonce-mined ours-mse-batched mse-mined ours-cosent cosent siglip-v3 ours-siglip siglip-mined)

TEXT_MODEL=sentence-transformers/all-mpnet-base-v2
IMG_MODEL=sentence-transformers/clip-ViT-B-32

if [[ $SMOKE == 1 ]]; then
  NOTE=${NOTE:-smoke}
  MODELS_ROOT=${MODELS_ROOT:-models/_smoke}
  TEXT_DATASET=${TEXT_DATASET:-dataset/processed/feature-distance-dataset_gemini-2.5-flash_1000000_nolek}
  IMG_DATASET=${IMG_DATASET:-dataset/processed/deepfashion-inshop-image-triplets_hf_20000_disjoint}
  IMG_TRAIN_EXTRA="--train-fraction 0.1"
  TOP_K=20
  export WANDB_MODE=offline
  REPORT_TO="--report-to none"
  WANDB_ARGS=""
else
  NOTE=${NOTE:-paper}
  MODELS_ROOT=${MODELS_ROOT:-models}
  TEXT_DATASET=${TEXT_DATASET:-dataset/processed/feature-distance-dataset_gemini-2.5-flash_1000000_nolek}
  IMG_DATASET=${IMG_DATASET:-dataset/processed/deepfashion-inshop-image-triplets_hf_20000_disjoint}
  IMG_TRAIN_EXTRA=""
  TOP_K=100
  REPORT_TO=""
  # Every run of a given NOTE lands in one wandb group so the conditions compare side by side.
  WANDB_ARGS="--wandb-group $NOTE"
fi

# Disk is tight: keep only final weights, no optimizer checkpoints.
TRAIN_COMMON="--save-strategy no --save-total-limit 1"

LOG_DIR=logs/paper
mkdir -p "$LOG_DIR" "$MODELS_ROOT"

# ---------------------------------------------------------------------------
# Condition table — the living list of everything the paper needs.
# Columns: modality  style  query_kind  V  extra
#   modality: text | multimodal
#   style:    untrained | baseline-triplet | infonce | infonce-mined | siglip-mined | mse-mined | cosent | ours-cosent
#             | mse (pairwise MSE of cos(q, x) onto 1/0; the ungraded control of classic-mse)
#             | classic-mse | ours-mse | ours-mse-batched | ours-infonce | ours-siglip
#             | ours-infonce-margin (one-hot infonce-mined + distance-scheduled logit margins)
#             | infonce-ours-v3 (infonce-mined with soft target mass e^{-s d/V} on the hard negative)
#             | siglip-v3 (siglip-mined with the own hard negative's target e^{-s d/V}; ours-siglip's 1 - d/V
#               target replaced the way infonce-ours-v3 replaces ours-infonce's)
#             infonce is the standard 2-column objective (in-batch negatives only);
#             infonce-mined adds our labeled hard negative as a third column.
#             | ours-mse-batched (ours over the full in-batch candidate pool)
#             (| ours-mse-reversed)
#   V:        distance normalizer, '-' = not applicable (untrained, triplet, infonce/-mined, siglip-mined,
#             and cosent -- all baselines that never see the measured distance)
#   extra:    '-' or comma-separated key=value; supported: easy=<int>, transform=<name>,
#             split=val, negs=mined (train on the mine_hard_negs.py sibling dataset),
#             negs=mined-graded (its label_mined_negs.py sibling, query_distance measured),
#             negs=random (the random_hard_negs.py sibling: hard negatives replaced by random products),
#             negs=mixed (the mix_hard_negs.py sibling: a seeded half of the train-split hard
#             negatives are the mined ones, the rest ours; mining= names the mined source),
#             mining=<variant> (a mine_hard_negs.py --variant sibling; mining sweep),
#             seed=<n> (a repeated trial with training seed n; the unsuffixed row is seed 42),
#             order=mined-first (train.py --train-order: every batch of the mined-negative
#             queries before any batch of the labeled ones, each epoch; negs=mixed rows),
#             rephrase=in-context (rephrased rows only: the _rephrased-in-context dataset, and the
#             _human-in-context eval set with the style examples held out)
# The image dataset only has synthetic (nl_query) queries, so multimodal rows are
# synthetic-only. The V ablation runs on synthetic queries for both modalities.

# Always hard code these. Do not rewrite. In general, comment out things we don't need to rerun; don't delete them.

# ---------------------------------------------------------------------------
CONDITIONS="

# 2026-09-09: synthetic test-split rows are commented out; the paper reports original and
# rephrased only. Their models and preds stay on disk; the synthetic val rows below remain
# as the record of each style's hparam selection.
# Main grid: {text, multimodal} x {original, synthetic, rephrased} x {baselines, ours}.
# Graded rows carry the hparams chosen on the validation easy x V sweep (analysis.ipynb
# section 2; selection metric Recall@5 for text since 2026-09-03 -- Recall@10 had
# saturated -- and Recall@20 for image), so there is no single global V: easy=10 throughout, V=80 for
# ours-infonce-margin and for every multimodal MSE row, V=40 for the text MSE rows --
# text and image disagree on V for the MSE styles. ours-infonce keeps the old V=40 and
# no easy override: it is deprecated, and easy=10 collides with its largest measured
# label, which train.py refuses for that loss alone.
#
# Rows commented out are up to date from the 2026-08-29 run and unchanged by anything
# since (new styles + batch-sampler cleanup only; see git log). Re-enable a row to have
# paper.sh check its freshness again.
#
# text        untrained         original   -   -
# text        untrained         synthetic  -   -
# text        untrained         rephrased  -   -
# text        baseline-triplet  original   -   -
# text        baseline-triplet  synthetic  -   -
# text        baseline-triplet  rephrased  -   -
# archived/retired: text        infonce           original   -   -
# archived/retired: text        infonce-mined     original   -   -
# archived/retired: text        siglip-mined      original   -   -
# text        infonce           synthetic  -   -
# text        infonce-mined     synthetic  -   -
# text        siglip-mined      synthetic  -   -
# archived/retired: text        infonce           rephrased  -   -
# archived/retired: text        infonce-mined     rephrased  -   -
# archived/retired: text        siglip-mined      rephrased  -   -
# archived/retired: text        cosent            original   -   -
# text        cosent            synthetic  -   -
# archived/retired: text        cosent            rephrased  -   -
# archived/retired: text        ours-cosent       original   -   -
# text        ours-cosent       synthetic  -   -
# archived/retired: text        ours-cosent       rephrased  -   -
# text        classic-mse       original   40  -
# text        classic-mse       synthetic  40  -
# text        classic-mse       rephrased  40  -
# archived/retired: text        ours-mse          original   40  easy=10
# archived/retired: text        ours-infonce      original   40  -
# archived/retired: text        ours-siglip       original   20  easy=10
# archived/retired: text        ours-infonce-margin original  80  easy=10
# text        ours-mse          synthetic  40  easy=10
# text        ours-infonce      synthetic  40  -
# text        ours-siglip       synthetic  20  easy=10
# text        ours-infonce-margin synthetic 80  easy=10
# archived/retired: text        ours-mse          rephrased  40  easy=10
# archived/retired: text        ours-infonce      rephrased  40  -
# archived/retired: text        ours-siglip       rephrased  20  easy=10
# archived/retired: text        ours-infonce-margin rephrased 80  easy=10
# archived/retired: text        ours-mse-batched  original   40  easy=10
# text        ours-mse-batched  synthetic  40  easy=10
# archived/retired: text        ours-mse-batched  rephrased  40  easy=10
# archived/retired: text        mse-mined         original   40  easy=10
# text        mse-mined         synthetic  40  easy=10
# archived/retired: text        mse-mined         rephrased  40  easy=10
# The image dataset has no real search queries (original_query is empty for all 12,957 rows),
# so multimodal runs synthetic and rephrased only.
# multimodal  untrained         synthetic  -   -
# multimodal  untrained         rephrased  -   -
# multimodal  baseline-triplet  synthetic  -   -
# multimodal  baseline-triplet  rephrased  -   -
# multimodal  infonce           synthetic  -   -
# multimodal  infonce-mined     synthetic  -   -
# multimodal  siglip-mined      synthetic  -   -
# archived/retired: multimodal  infonce           rephrased  -   -
# archived/retired: multimodal  infonce-mined     rephrased  -   -
# archived/retired: multimodal  siglip-mined      rephrased  -   -
# multimodal  cosent            synthetic  -   -
# archived/retired: multimodal  cosent            rephrased  -   -
# multimodal  ours-cosent       synthetic  -   -
# archived/retired: multimodal  ours-cosent       rephrased  -   -
# multimodal  classic-mse       synthetic  40  -
# multimodal  classic-mse       rephrased  40  -
# multimodal  ours-mse          synthetic  80  easy=10
# multimodal  ours-infonce      synthetic  40  -
# multimodal  ours-siglip       synthetic  20  easy=10
# multimodal  ours-infonce-margin synthetic 80  easy=10
# archived/retired: multimodal  ours-mse          rephrased  80  easy=10
# archived/retired: multimodal  ours-infonce      rephrased  40  -
# multimodal  ours-siglip       rephrased  20  easy=10   # synthetic-val pick; per-query-kind val (2026-09-05) selects V=40
# archived/retired: multimodal  ours-siglip       rephrased  40  easy=10
# archived/retired: multimodal  ours-infonce-margin rephrased 80  easy=10
# multimodal  ours-mse-batched  synthetic  80  easy=10
# multimodal  ours-mse-batched  rephrased  80  easy=10   # synthetic-val pick; per-query-kind val (2026-09-05) selects V=20
# archived/retired: multimodal  ours-mse-batched  rephrased  20  easy=10
# multimodal  mse-mined         synthetic  80  easy=10
# archived/retired: multimodal  mse-mined         rephrased  80  easy=10

# ---------------------------------------------------------------------------
# V ablation: 20/40/60 on synthetic queries for every graded style. The V=40 point
# is the main-grid row above, so it is not repeated here.
# ---------------------------------------------------------------------------
# text        ours-mse          synthetic  20  -
# text        ours-mse          synthetic  60  -
# text        ours-mse-batched  synthetic  20  -
# text        ours-mse-batched  synthetic  60  -
# archived/retired: text        ours-infonce      synthetic  20  split=val
# archived/retired: text        ours-infonce      synthetic  60  split=val
# text        ours-siglip       synthetic  20  split=val
# text        ours-siglip       synthetic  60  split=val
# archived/retired: text        ours-infonce-margin synthetic 10  split=val
# archived/retired: text        ours-infonce-margin synthetic 20  split=val
# archived/retired: text        ours-infonce-margin synthetic 60  split=val
# multimodal  ours-mse          synthetic  20  -
# multimodal  ours-mse          synthetic  60  -
# multimodal  ours-mse-batched  synthetic  20  -
# multimodal  ours-mse-batched  synthetic  60  -
# archived/retired: multimodal  ours-infonce      synthetic  20  split=val
# archived/retired: multimodal  ours-infonce      synthetic  60  split=val
# multimodal  ours-siglip       synthetic  20  split=val
# multimodal  ours-siglip       synthetic  60  split=val
# archived/retired: multimodal  ours-infonce-margin synthetic 10  split=val
# archived/retired: multimodal  ours-infonce-margin synthetic 20  split=val
# archived/retired: multimodal  ours-infonce-margin synthetic 60  split=val

# ---------------------------------------------------------------------------
# Easy-value ablation: where does the random-negative penalty sit on the scale?
# easy=20 (label .5, the default) is the main-grid/V-ablation row for each style;
# easy=30 (label .75) and easy=40 (label 1.0 = the scale maximum) decouple the easy
# placement from V, which the V ablation alone conflates (easy label = 20/V there).
# ours-infonce is exempt: its easy rows are one-hot regardless of the easy value.
# ---------------------------------------------------------------------------
# archived/retired: text        classic-mse       synthetic  40  easy=30,split=val
# archived/retired: text        classic-mse       synthetic  40  easy=40,split=val
# archived/retired: text        ours-mse          synthetic  40  easy=30,split=val
# archived/retired: text        ours-mse          synthetic  40  easy=40,split=val
# archived/retired: text        ours-mse-batched  synthetic  40  easy=30,split=val
# archived/retired: text        ours-mse-batched  synthetic  40  easy=40,split=val
# text        ours-siglip       synthetic  40  easy=30,split=val
# text        ours-siglip       synthetic  40  easy=40,split=val
# archived/retired: multimodal  classic-mse       synthetic  40  easy=30,split=val
# archived/retired: multimodal  classic-mse       synthetic  40  easy=40,split=val
# archived/retired: multimodal  ours-mse          synthetic  40  easy=30,split=val
# archived/retired: multimodal  ours-mse          synthetic  40  easy=40,split=val
# archived/retired: multimodal  ours-mse-batched  synthetic  40  easy=30,split=val
# archived/retired: multimodal  ours-mse-batched  synthetic  40  easy=40,split=val
# multimodal  ours-siglip       synthetic  40  easy=30,split=val
# multimodal  ours-siglip       synthetic  40  easy=40,split=val

# -------------------------------------------------------------------------
# easy x V grid (2026-09-01). Selection sweep for the ours-* family: every combination
# of the easy-negative distance and the distance normalizer, synthetic queries, scored on
# VALIDATION (split=val -> preds_val/). Selecting hyperparameters on the test split would
# tune on the reported numbers, so every row here reads val; the main grid above stays test.
#
# easy=10 sits at the top of the measured scale (hard distances run 1..10), so its label
# equals that of the most distant hard negatives. The three batch-wide losses only FILL
# cross-row cells with easy_label and are unaffected; ours-mse regresses the scalar label
# and never compares. GradedInfoNCELoss (ours-infonce) DOES compare, and train.py refuses
# that combination rather than silently retargeting those rows.
#
# ours-infonce is deprecated in favour of ours-infonce-margin and is deliberately absent:
# no further ours-infonce runs unless something turns out to be badly wrong with margin.
# -------------------------------------------------------------------------
# archived/retired: text        ours-mse          synthetic  20  easy=10,split=val
# archived/retired: text        ours-mse          synthetic  40  easy=10,split=val
# archived/retired: text        ours-mse          synthetic  80  easy=10,split=val
# archived/retired: text        ours-mse          synthetic  20  easy=20,split=val
# archived/retired: text        ours-mse          synthetic  40  easy=20,split=val
# archived/retired: text        ours-mse          synthetic  80  easy=20,split=val
# archived/retired: text        ours-mse          synthetic  20  easy=40,split=val
# archived/retired: text        ours-mse          synthetic  40  easy=40,split=val
# archived/retired: text        ours-mse          synthetic  80  easy=40,split=val
# archived/retired: text        ours-mse-batched  synthetic  20  easy=10,split=val
# archived/retired: text        ours-mse-batched  synthetic  40  easy=10,split=val
# archived/retired: text        ours-mse-batched  synthetic  80  easy=10,split=val
# archived/retired: text        ours-mse-batched  synthetic  20  easy=20,split=val
# archived/retired: text        ours-mse-batched  synthetic  40  easy=20,split=val
# archived/retired: text        ours-mse-batched  synthetic  80  easy=20,split=val
# archived/retired: text        ours-mse-batched  synthetic  20  easy=40,split=val
# archived/retired: text        ours-mse-batched  synthetic  40  easy=40,split=val
# archived/retired: text        ours-mse-batched  synthetic  80  easy=40,split=val
# archived/retired: text        ours-siglip       synthetic  20  easy=10,split=val
# archived/retired: text        ours-siglip       synthetic  40  easy=10,split=val
# archived/retired: text        ours-siglip       synthetic  80  easy=10,split=val
# archived/retired: text        ours-siglip       synthetic  20  easy=20,split=val
# archived/retired: text        ours-siglip       synthetic  40  easy=20,split=val
# archived/retired: text        ours-siglip       synthetic  80  easy=20,split=val
# archived/retired: text        ours-siglip       synthetic  20  easy=40,split=val
# archived/retired: text        ours-siglip       synthetic  40  easy=40,split=val
# archived/retired: text        ours-siglip       synthetic  80  easy=40,split=val
# text        ours-siglip       synthetic  20  easy=10,split=val
# text        ours-siglip       synthetic  40  easy=10,split=val
# text        ours-siglip       synthetic  80  easy=10,split=val
# text        ours-siglip       synthetic  20  easy=20,split=val
# text        ours-siglip       synthetic  40  easy=20,split=val
# text        ours-siglip       synthetic  80  easy=20,split=val
# text        ours-siglip       synthetic  20  easy=40,split=val
# text        ours-siglip       synthetic  40  easy=40,split=val
# text        ours-siglip       synthetic  80  easy=40,split=val
# archived/retired: text        ours-infonce-margin synthetic  20  easy=10,split=val
# archived/retired: text        ours-infonce-margin synthetic  40  easy=10,split=val
# archived/retired: text        ours-infonce-margin synthetic  80  easy=10,split=val
# archived/retired: text        ours-infonce-margin synthetic  20  easy=20,split=val
# archived/retired: text        ours-infonce-margin synthetic  40  easy=20,split=val
# archived/retired: text        ours-infonce-margin synthetic  80  easy=20,split=val
# archived/retired: text        ours-infonce-margin synthetic  20  easy=40,split=val
# archived/retired: text        ours-infonce-margin synthetic  40  easy=40,split=val
# archived/retired: text        ours-infonce-margin synthetic  80  easy=40,split=val
# archived/retired: multimodal  ours-mse          synthetic  20  easy=10,split=val
# archived/retired: multimodal  ours-mse          synthetic  40  easy=10,split=val
# archived/retired: multimodal  ours-mse          synthetic  80  easy=10,split=val
# archived/retired: multimodal  ours-mse          synthetic  20  easy=20,split=val
# archived/retired: multimodal  ours-mse          synthetic  40  easy=20,split=val
# archived/retired: multimodal  ours-mse          synthetic  80  easy=20,split=val
# archived/retired: multimodal  ours-mse          synthetic  20  easy=40,split=val
# archived/retired: multimodal  ours-mse          synthetic  40  easy=40,split=val
# archived/retired: multimodal  ours-mse          synthetic  80  easy=40,split=val
# archived/retired: multimodal  ours-mse-batched  synthetic  20  easy=10,split=val
# archived/retired: multimodal  ours-mse-batched  synthetic  40  easy=10,split=val
# archived/retired: multimodal  ours-mse-batched  synthetic  80  easy=10,split=val
# archived/retired: multimodal  ours-mse-batched  synthetic  20  easy=20,split=val
# archived/retired: multimodal  ours-mse-batched  synthetic  40  easy=20,split=val
# archived/retired: multimodal  ours-mse-batched  synthetic  80  easy=20,split=val
# archived/retired: multimodal  ours-mse-batched  synthetic  20  easy=40,split=val
# archived/retired: multimodal  ours-mse-batched  synthetic  40  easy=40,split=val
# archived/retired: multimodal  ours-mse-batched  synthetic  80  easy=40,split=val
# archived/retired: multimodal  ours-siglip       synthetic  20  easy=10,split=val
# archived/retired: multimodal  ours-siglip       synthetic  40  easy=10,split=val
# archived/retired: multimodal  ours-siglip       synthetic  80  easy=10,split=val
# archived/retired: multimodal  ours-siglip       synthetic  20  easy=20,split=val
# archived/retired: multimodal  ours-siglip       synthetic  40  easy=20,split=val
# archived/retired: multimodal  ours-siglip       synthetic  80  easy=20,split=val
# archived/retired: multimodal  ours-siglip       synthetic  20  easy=40,split=val
# archived/retired: multimodal  ours-siglip       synthetic  40  easy=40,split=val
# archived/retired: multimodal  ours-siglip       synthetic  80  easy=40,split=val
# multimodal  ours-siglip       synthetic  20  easy=10,split=val
# multimodal  ours-siglip       synthetic  40  easy=10,split=val
# multimodal  ours-siglip       synthetic  80  easy=10,split=val
# multimodal  ours-siglip       synthetic  20  easy=20,split=val
# multimodal  ours-siglip       synthetic  40  easy=20,split=val
# multimodal  ours-siglip       synthetic  80  easy=20,split=val
# multimodal  ours-siglip       synthetic  20  easy=40,split=val
# multimodal  ours-siglip       synthetic  40  easy=40,split=val
# multimodal  ours-siglip       synthetic  80  easy=40,split=val
# archived/retired: multimodal  ours-infonce-margin synthetic  20  easy=10,split=val
# archived/retired: multimodal  ours-infonce-margin synthetic  40  easy=10,split=val
# archived/retired: multimodal  ours-infonce-margin synthetic  80  easy=10,split=val
# archived/retired: multimodal  ours-infonce-margin synthetic  20  easy=20,split=val
# archived/retired: multimodal  ours-infonce-margin synthetic  40  easy=20,split=val
# archived/retired: multimodal  ours-infonce-margin synthetic  80  easy=20,split=val
# archived/retired: multimodal  ours-infonce-margin synthetic  20  easy=40,split=val
# archived/retired: multimodal  ours-infonce-margin synthetic  40  easy=40,split=val
# archived/retired: multimodal  ours-infonce-margin synthetic  80  easy=40,split=val

# -------------------------------------------------------------------------
# infonce-ours-v3 (2026-09-03): ours-infonce with the hard negative's target mass put through
# the exponential, e^{-s d/V} in place of 1 - d/V, so the gap at the optimum is d/V itself
# rather than d/(sV) (tmp/infonce_bounds.tex, Point 4). Two-sided: a negative pushed past its
# gap holds less probability than its target and is pushed back up.
#
# The sweep is one-dimensional. easy plays no part in this loss -- random products hold target
# mass 0 whatever easy is; the label only identifies them -- so there is no easy axis. It stays
# at the default (20): easy=10 collides with d=10 and this loss, like ours-infonce, identifies
# random rows by label equality, which train.py refuses. (At V=10 the d=10 label clips to 1.0
# and meets the easy label anyway; those rows then get mass 0 instead of e^{-20}, no difference.)
#
# V sets the target odds ratio per violated constraint, e^{-s/V}: 0.14 at V=10, 0.37 at 20,
# 0.61 at 40, 0.78 at 80. The note's Section 6 measured the margin loss's natural gaps at about
# d/V for V ~ 20, so the grid brackets that. Scored on VALIDATION. The two infonce-mined rows
# put the ungraded control on the same split (models exist; inference only); ours-infonce-margin
# at its selected V=80/easy=10 is already scored there by the easy x V grid above.
# -------------------------------------------------------------------------
# archived/retired: text        infonce-ours-v3   synthetic  10  split=val
# archived/retired: text        infonce-ours-v3   synthetic  20  split=val
# archived/retired: text        infonce-ours-v3   synthetic  40  split=val
# archived/retired: text        infonce-ours-v3   synthetic  80  split=val
# archived/retired: multimodal  infonce-ours-v3   synthetic  10  split=val
# archived/retired: multimodal  infonce-ours-v3   synthetic  20  split=val
# archived/retired: multimodal  infonce-ours-v3   synthetic  40  split=val
# archived/retired: multimodal  infonce-ours-v3   synthetic  80  split=val
# archived/retired: text        infonce-mined     synthetic  -   split=val
# archived/retired: multimodal  infonce-mined     synthetic  -   split=val
# Main-grid rows at the V selected on the sweep (analysis.ipynb section 2): text V=20 (Recall@5;
# V=10 and V=20 tie to 2e-4, and the V=10 rows were trained once before the metric changed),
# image V=10 (Recall@20). Fill V in, then
# uncomment; the synthetic rows share their model with the sweep and are inference only.
# archived/retired: text        infonce-ours-v3   original   10  -
# text        infonce-ours-v3   synthetic  10  -         # recall@10 pick; recall@5 val (2026-09-05) selects V=20
# text        infonce-ours-v3   synthetic  20  -
# text        infonce-ours-v3   rephrased  10  -         # synthetic-val pick; per-query-kind val (2026-09-05) selects V=20
# archived/retired: text        infonce-ours-v3   rephrased  20  -
# multimodal  infonce-ours-v3   synthetic  10  -
# archived/retired: multimodal  infonce-ours-v3   rephrased  10  -
# -------------------------------------------------------------------------
# Per-query-kind hparam search (2026-09-04). The grids above select on synthetic val and
# transfer that choice to the original/rephrased core rows. These sweep the same grid on
# every training distribution a core row uses, so each core row is selected on its own
# val split. Text is scored at recall@5, image at recall@20 (analysis.ipynb K_TEXT/K_IMAGE).
# Cells that coincide with a current core row share its model and are inference only.
# -------------------------------------------------------------------------
# archived/retired: text        ours-siglip       original   20  easy=10,split=val
# archived/retired: text        ours-siglip       original   40  easy=10,split=val
# archived/retired: text        ours-siglip       original   80  easy=10,split=val
# archived/retired: text        ours-siglip       original   20  easy=20,split=val
# archived/retired: text        ours-siglip       original   40  easy=20,split=val
# archived/retired: text        ours-siglip       original   80  easy=20,split=val
# archived/retired: text        ours-siglip       original   20  easy=40,split=val
# archived/retired: text        ours-siglip       original   40  easy=40,split=val
# archived/retired: text        ours-siglip       original   80  easy=40,split=val
# archived/retired: text        ours-siglip       rephrased  20  easy=10,split=val
# archived/retired: text        ours-siglip       rephrased  40  easy=10,split=val
# archived/retired: text        ours-siglip       rephrased  80  easy=10,split=val
# archived/retired: text        ours-siglip       rephrased  20  easy=20,split=val
# archived/retired: text        ours-siglip       rephrased  40  easy=20,split=val
# archived/retired: text        ours-siglip       rephrased  80  easy=20,split=val
# archived/retired: text        ours-siglip       rephrased  20  easy=40,split=val
# archived/retired: text        ours-siglip       rephrased  40  easy=40,split=val
# archived/retired: text        ours-siglip       rephrased  80  easy=40,split=val
# archived/retired: multimodal  ours-siglip       rephrased  20  easy=10,split=val
# archived/retired: multimodal  ours-siglip       rephrased  40  easy=10,split=val
# archived/retired: multimodal  ours-siglip       rephrased  80  easy=10,split=val
# archived/retired: multimodal  ours-siglip       rephrased  20  easy=20,split=val
# archived/retired: multimodal  ours-siglip       rephrased  40  easy=20,split=val
# archived/retired: multimodal  ours-siglip       rephrased  80  easy=20,split=val
# archived/retired: multimodal  ours-siglip       rephrased  20  easy=40,split=val
# archived/retired: multimodal  ours-siglip       rephrased  40  easy=40,split=val
# archived/retired: multimodal  ours-siglip       rephrased  80  easy=40,split=val
# archived/retired: text        ours-mse-batched  original   20  easy=10,split=val
# archived/retired: text        ours-mse-batched  original   40  easy=10,split=val
# archived/retired: text        ours-mse-batched  original   80  easy=10,split=val
# archived/retired: text        ours-mse-batched  original   20  easy=20,split=val
# archived/retired: text        ours-mse-batched  original   40  easy=20,split=val
# archived/retired: text        ours-mse-batched  original   80  easy=20,split=val
# archived/retired: text        ours-mse-batched  original   20  easy=40,split=val
# archived/retired: text        ours-mse-batched  original   40  easy=40,split=val
# archived/retired: text        ours-mse-batched  original   80  easy=40,split=val
# archived/retired: text        ours-mse-batched  rephrased  20  easy=10,split=val
# archived/retired: text        ours-mse-batched  rephrased  40  easy=10,split=val
# archived/retired: text        ours-mse-batched  rephrased  80  easy=10,split=val
# archived/retired: text        ours-mse-batched  rephrased  20  easy=20,split=val
# archived/retired: text        ours-mse-batched  rephrased  40  easy=20,split=val
# archived/retired: text        ours-mse-batched  rephrased  80  easy=20,split=val
# archived/retired: text        ours-mse-batched  rephrased  20  easy=40,split=val
# archived/retired: text        ours-mse-batched  rephrased  40  easy=40,split=val
# archived/retired: text        ours-mse-batched  rephrased  80  easy=40,split=val
# archived/retired: multimodal  ours-mse-batched  rephrased  20  easy=10,split=val
# archived/retired: multimodal  ours-mse-batched  rephrased  40  easy=10,split=val
# archived/retired: multimodal  ours-mse-batched  rephrased  80  easy=10,split=val
# archived/retired: multimodal  ours-mse-batched  rephrased  20  easy=20,split=val
# archived/retired: multimodal  ours-mse-batched  rephrased  40  easy=20,split=val
# archived/retired: multimodal  ours-mse-batched  rephrased  80  easy=20,split=val
# archived/retired: multimodal  ours-mse-batched  rephrased  20  easy=40,split=val
# archived/retired: multimodal  ours-mse-batched  rephrased  40  easy=40,split=val
# archived/retired: multimodal  ours-mse-batched  rephrased  80  easy=40,split=val
# archived/retired: text        infonce-ours-v3   original   10  split=val
# archived/retired: text        infonce-ours-v3   original   20  split=val
# archived/retired: text        infonce-ours-v3   original   40  split=val
# archived/retired: text        infonce-ours-v3   original   80  split=val
# archived/retired: text        infonce-ours-v3   rephrased  10  split=val
# archived/retired: text        infonce-ours-v3   rephrased  20  split=val
# archived/retired: text        infonce-ours-v3   rephrased  40  split=val
# archived/retired: text        infonce-ours-v3   rephrased  80  split=val
# archived/retired: multimodal  infonce-ours-v3   rephrased  10  split=val
# archived/retired: multimodal  infonce-ours-v3   rephrased  20  split=val
# archived/retired: multimodal  infonce-ours-v3   rephrased  40  split=val
# archived/retired: multimodal  infonce-ours-v3   rephrased  80  split=val

# -------------------------------------------------------------------------
# Retrieval-mined negatives (2026-09-04): the standard hard-negative baseline. Datasets are
# the mine_hard_negs.py siblings (<dataset>_mined-<kind>): same rows and split, but every
# train-split hard negative is the product a frozen teacher (e5-mistral-7b-instruct for text,
# clip-ViT-L-14 for image) retrieves under NV-Retriever's positive-aware rule -- top-100,
# drop candidates above 95% of the positive's score, keep the best survivor (Moreira et al.,
# CIKM 2025). Val and test rows are untouched, so these score on the same corpus as every
# other row. infonce-mined is the strongest ungraded loss in the protocol figure, so it alone
# carries the best-miner-plus-strongest-standard-loss control; the graded losses cannot run on
# these rows until a labeling pass measures each mined negative's distance.
# Uncomment once the five mined datasets exist; a missing dataset aborts the plan.
# -------------------------------------------------------------------------
# archived/retired: text        infonce-mined     original   -   negs=mined
# text        infonce-mined     synthetic  -   negs=mined
# text        infonce-mined     rephrased  -   negs=mined   # default mining; the sweep (below) selects m0.025_s10 on val
# archived/retired: text        infonce-mined     rephrased  -   negs=mined,mining=m0.025_s10
# multimodal  infonce-mined     synthetic  -   negs=mined
# multimodal  infonce-mined     rephrased  -   negs=mined   # default mining; the sweep (below) selects m0.025_s10 on val
# archived/retired: multimodal  infonce-mined     rephrased  -   negs=mined,mining=m0.025_s10
# -------------------------------------------------------------------------
# Graded losses on retrieval-mined negatives (2026-09-08): the same rows as the five
# negs=mined rows above, after label_mined_negs.py measured each mined negative's
# query_distance. Best miner + graded loss against best miner + standard loss, on
# identical negatives; hparams are each style's per-query-kind selection on labeled rows.
# Uncomment once the five _graded datasets exist.
# -------------------------------------------------------------------------
# archived/retired: text        infonce-ours-v3   original   10  negs=mined-graded
# text        infonce-ours-v3   synthetic  20  negs=mined-graded
# archived/retired: text        infonce-ours-v3   rephrased  20  negs=mined-graded
# multimodal  infonce-ours-v3   synthetic  10  negs=mined-graded
# archived/retired: multimodal  infonce-ours-v3   rephrased  10  negs=mined-graded
# archived/retired: text        ours-infonce-margin original  80  easy=10,negs=mined-graded
# text        ours-infonce-margin synthetic 80  easy=10,negs=mined-graded
# archived/retired: text        ours-infonce-margin rephrased 80  easy=10,negs=mined-graded
# multimodal  ours-infonce-margin synthetic 80  easy=10,negs=mined-graded
# archived/retired: multimodal  ours-infonce-margin rephrased 80  easy=10,negs=mined-graded
# -------------------------------------------------------------------------
# NV-Retriever mining sweep for the infonce-mined baseline (2026-09-09), rephrased only.
# relative margin {0.025, 0.05, 0.1, 0.2} x survivor {first (s0), skip top 10 (s10)},
# scored on VALIDATION at recall@5 text / recall@20 image, like every ours-* sweep. The
# (0.05, s0) cell is the default dataset and reuses the core row's model. Datasets come
# from logs/mine/run_nv_sweep.sh; uncomment once they exist.
# -------------------------------------------------------------------------
# archived/retired: text        infonce-mined     rephrased  -   negs=mined,mining=m0.025_s0,split=val
# archived/retired: multimodal  infonce-mined     rephrased  -   negs=mined,mining=m0.025_s0,split=val
# archived/retired: text        infonce-mined     rephrased  -   negs=mined,mining=m0.025_s10,split=val
# archived/retired: multimodal  infonce-mined     rephrased  -   negs=mined,mining=m0.025_s10,split=val
# archived/retired: text        infonce-mined     rephrased  -   negs=mined,split=val
# archived/retired: multimodal  infonce-mined     rephrased  -   negs=mined,split=val
# archived/retired: text        infonce-mined     rephrased  -   negs=mined,mining=m0.05_s10,split=val
# archived/retired: multimodal  infonce-mined     rephrased  -   negs=mined,mining=m0.05_s10,split=val
# archived/retired: text        infonce-mined     rephrased  -   negs=mined,mining=m0.1_s0,split=val
# archived/retired: multimodal  infonce-mined     rephrased  -   negs=mined,mining=m0.1_s0,split=val
# archived/retired: text        infonce-mined     rephrased  -   negs=mined,mining=m0.1_s10,split=val
# archived/retired: multimodal  infonce-mined     rephrased  -   negs=mined,mining=m0.1_s10,split=val
# archived/retired: text        infonce-mined     rephrased  -   negs=mined,mining=m0.2_s0,split=val
# archived/retired: multimodal  infonce-mined     rephrased  -   negs=mined,mining=m0.2_s0,split=val
# archived/retired: text        infonce-mined     rephrased  -   negs=mined,mining=m0.2_s10,split=val
# archived/retired: multimodal  infonce-mined     rephrased  -   negs=mined,mining=m0.2_s10,split=val
# -------------------------------------------------------------------------
# Repeated trials (2026-09-10): the headline comparison, rephrased, both modalities, at
# each row's selected hparams -- infonce-ours-v3 (ours, graded), infonce-mined on labeled
# negatives (ours, ungraded), infonce-mined on the val-selected retrieval-mined negatives
# (baseline). Seeds 43-44; the unsuffixed core row above is seed 42, so every condition
# has 3 trials (cut from 5 on 2026-09-11; the seed 45/46 rows stay commented). analysis.ipynb
# reports mean and a 95% CI over the healthy trials.
# -------------------------------------------------------------------------
# archived/retired: text        infonce-ours-v3   rephrased  20  seed=43
# archived/retired: multimodal  infonce-ours-v3   rephrased  10  seed=43
# archived/retired: text        infonce-mined     rephrased  -   seed=43
# archived/retired: multimodal  infonce-mined     rephrased  -   seed=43
# archived/retired: text        infonce-mined     rephrased  -   negs=mined,mining=m0.025_s10,seed=43
# archived/retired: multimodal  infonce-mined     rephrased  -   negs=mined,mining=m0.025_s10,seed=43
# archived/retired: text        infonce-ours-v3   rephrased  20  seed=44
# archived/retired: multimodal  infonce-ours-v3   rephrased  10  seed=44
# archived/retired: text        infonce-mined     rephrased  -   seed=44
# archived/retired: multimodal  infonce-mined     rephrased  -   seed=44
# archived/retired: text        infonce-mined     rephrased  -   negs=mined,mining=m0.025_s10,seed=44
# archived/retired: multimodal  infonce-mined     rephrased  -   negs=mined,mining=m0.025_s10,seed=44
# text        infonce-ours-v3   rephrased  20  seed=45
# multimodal  infonce-ours-v3   rephrased  10  seed=45
# text        infonce-mined     rephrased  -   seed=45
# multimodal  infonce-mined     rephrased  -   seed=45
# text        infonce-mined     rephrased  -   negs=mined,mining=m0.025_s10,seed=45
# multimodal  infonce-mined     rephrased  -   negs=mined,mining=m0.025_s10,seed=45
# text        infonce-ours-v3   rephrased  20  seed=46
# multimodal  infonce-ours-v3   rephrased  10  seed=46
# text        infonce-mined     rephrased  -   seed=46
# multimodal  infonce-mined     rephrased  -   seed=46
# text        infonce-mined     rephrased  -   negs=mined,mining=m0.025_s10,seed=46
# multimodal  infonce-mined     rephrased  -   negs=mined,mining=m0.025_s10,seed=46
# -------------------------------------------------------------------------
# In-context rephrasings (2026-09-11): every rephrased main-grid row retrained on the
# rephrase_dataset.py --in-context sibling (_rephrased-in-context), whose prompt carried
# human-written style examples, and scored on its test split and on _human-in-context.
# Hparams are each style's per-query-kind selection on the plain rephrasing; not re-swept.
# -------------------------------------------------------------------------
text        infonce           rephrased  -   rephrase=in-context
text        infonce-mined     rephrased  -   rephrase=in-context
text        siglip-mined      rephrased  -   rephrase=in-context
text        cosent            rephrased  -   rephrase=in-context
text        ours-cosent       rephrased  -   rephrase=in-context
text        mse               rephrased  -   rephrase=in-context
# archived/retired: text        ours-mse          rephrased  40  easy=10,rephrase=in-context
# archived/retired: text        ours-infonce      rephrased  40  rephrase=in-context
# archived/retired: text        ours-siglip       rephrased  20  easy=10,rephrase=in-context
# archived/retired: text        ours-infonce-margin rephrased  80  easy=10,rephrase=in-context
text        ours-mse-batched  rephrased  40  easy=10,rephrase=in-context
text        mse-mined         rephrased  40  easy=10,rephrase=in-context
multimodal  infonce           rephrased  -   rephrase=in-context
multimodal  infonce-mined     rephrased  -   rephrase=in-context
multimodal  siglip-mined      rephrased  -   rephrase=in-context
multimodal  cosent            rephrased  -   rephrase=in-context
multimodal  ours-cosent       rephrased  -   rephrase=in-context
multimodal  mse               rephrased  -   rephrase=in-context
# archived/retired: multimodal  ours-mse          rephrased  80  easy=10,rephrase=in-context
# archived/retired: multimodal  ours-infonce      rephrased  40  rephrase=in-context
# archived/retired: multimodal  ours-siglip       rephrased  40  easy=10,rephrase=in-context
# archived/retired: multimodal  ours-infonce-margin rephrased  80  easy=10,rephrase=in-context
multimodal ours-mse-batched rephrased 20 easy=10,rephrase=in-context
multimodal  mse-mined         rephrased  80  easy=10,rephrase=in-context
text        infonce-ours-v3   rephrased  20  rephrase=in-context
multimodal infonce-ours-v3 rephrased 40 rephrase=in-context
# Repeated trials of the in-context headline pair, so its -ic bars carry n=3 like the plain ones.
# Baseline loss selection (analysis.ipynb section 4, 2026-09-12): the ungraded losses scored on
# the in-context VALIDATION split, so the choice of baseline is not made on the test set. Same
# weights as the in-context rows (the seed=43/44 trials are below); inference only. The table
# marks any slot with fewer than 3 seeds in red, so every loss here has all three.
text        infonce-mined     rephrased  -   split=val,rephrase=in-context
text        infonce-mined     rephrased  -   split=val,seed=43,rephrase=in-context
text        infonce-mined     rephrased  -   split=val,seed=44,rephrase=in-context
text        siglip-mined      rephrased  -   split=val,rephrase=in-context
text        siglip-mined      rephrased  -   split=val,seed=43,rephrase=in-context
text        siglip-mined      rephrased  -   split=val,seed=44,rephrase=in-context
text        cosent            rephrased  -   split=val,rephrase=in-context
text        cosent            rephrased  -   split=val,seed=43,rephrase=in-context
text        cosent            rephrased  -   split=val,seed=44,rephrase=in-context
text        mse               rephrased  -   split=val,rephrase=in-context
text        mse               rephrased  -   split=val,seed=43,rephrase=in-context
text        mse               rephrased  -   split=val,seed=44,rephrase=in-context
multimodal  infonce-mined     rephrased  -   split=val,rephrase=in-context
multimodal  infonce-mined     rephrased  -   split=val,seed=43,rephrase=in-context
multimodal  infonce-mined     rephrased  -   split=val,seed=44,rephrase=in-context
multimodal  siglip-mined      rephrased  -   split=val,rephrase=in-context
multimodal  siglip-mined      rephrased  -   split=val,seed=43,rephrase=in-context
multimodal  siglip-mined      rephrased  -   split=val,seed=44,rephrase=in-context
multimodal  cosent            rephrased  -   split=val,rephrase=in-context
multimodal  cosent            rephrased  -   split=val,seed=43,rephrase=in-context
multimodal  cosent            rephrased  -   split=val,seed=44,rephrase=in-context
multimodal  mse               rephrased  -   split=val,rephrase=in-context
multimodal  mse               rephrased  -   split=val,seed=43,rephrase=in-context
multimodal  mse               rephrased  -   split=val,seed=44,rephrase=in-context
# InfoNCE + NV: infonce-mined on the nv-mined in-context datasets, same 3 seeds, validation split
text        infonce-mined     rephrased  -   split=val,negs=mined,mining=m0.025_s10,rephrase=in-context
# text        infonce-mined     rephrased  -   split=val,negs=mined,mining=m0.025_s10,seed=43,rephrase=in-context
# text        infonce-mined     rephrased  -   split=val,negs=mined,mining=m0.025_s10,seed=44,rephrase=in-context
multimodal  infonce-mined     rephrased  -   split=val,negs=mined,mining=m0.025_s10,rephrase=in-context
# multimodal  infonce-mined     rephrased  -   split=val,negs=mined,mining=m0.025_s10,seed=43,rephrase=in-context
# multimodal  infonce-mined     rephrased  -   split=val,negs=mined,mining=m0.025_s10,seed=44,rephrase=in-context
# seeds 43/44 of the in-context re-selected variant (2026-09-12)
text        infonce-mined     rephrased  -   split=val,negs=mined,mining=m0.1_s10,seed=43,rephrase=in-context
text        infonce-mined     rephrased  -   split=val,negs=mined,mining=m0.1_s10,seed=44,rephrase=in-context
multimodal infonce-mined rephrased - split=val,negs=mined,mining=m0.2_s10,seed=43,rephrase=in-context
multimodal infonce-mined rephrased - split=val,negs=mined,mining=m0.2_s10,seed=44,rephrase=in-context
# In-context re-run of the NV-Retriever mining sweep (logs/mine/run_nv_sweep_ic_all.sh,
# 2026-09-12): every grid cell carries an explicit tag, the (0.025, s10) cell is the seed-42
# val row above. Selection lives in analysis.ipynb NV_SELECTED[..., "in-context"].
text        infonce-mined     rephrased  -   split=val,negs=mined,mining=m0.025_s0,rephrase=in-context
text        infonce-mined     rephrased  -   split=val,negs=mined,mining=m0.05_s0,rephrase=in-context
text        infonce-mined     rephrased  -   split=val,negs=mined,mining=m0.05_s10,rephrase=in-context
text        infonce-mined     rephrased  -   split=val,negs=mined,mining=m0.1_s0,rephrase=in-context
text        infonce-mined     rephrased  -   split=val,negs=mined,mining=m0.1_s10,rephrase=in-context
text        infonce-mined     rephrased  -   split=val,negs=mined,mining=m0.2_s0,rephrase=in-context
text        infonce-mined     rephrased  -   split=val,negs=mined,mining=m0.2_s10,rephrase=in-context
multimodal  infonce-mined     rephrased  -   split=val,negs=mined,mining=m0.025_s0,rephrase=in-context
multimodal  infonce-mined     rephrased  -   split=val,negs=mined,mining=m0.05_s0,rephrase=in-context
multimodal  infonce-mined     rephrased  -   split=val,negs=mined,mining=m0.05_s10,rephrase=in-context
multimodal  infonce-mined     rephrased  -   split=val,negs=mined,mining=m0.1_s0,rephrase=in-context
multimodal  infonce-mined     rephrased  -   split=val,negs=mined,mining=m0.1_s10,rephrase=in-context
multimodal  infonce-mined     rephrased  -   split=val,negs=mined,mining=m0.2_s0,rephrase=in-context
multimodal  infonce-mined     rephrased  -   split=val,negs=mined,mining=m0.2_s10,rephrase=in-context
text        infonce-ours-v3   rephrased  20  seed=43,rephrase=in-context
text        infonce-ours-v3   rephrased  20  seed=44,rephrase=in-context
multimodal infonce-ours-v3 rephrased 40 seed=43,rephrase=in-context
multimodal infonce-ours-v3 rephrased 40 seed=44,rephrase=in-context
text        infonce-mined     rephrased  -   seed=43,rephrase=in-context
text        infonce-mined     rephrased  -   seed=44,rephrase=in-context
multimodal  infonce-mined     rephrased  -   seed=43,rephrase=in-context
multimodal  infonce-mined     rephrased  -   seed=44,rephrase=in-context
# seeds 43/44 of the other ungraded losses (baseline loss selection, 2026-09-12)
text        siglip-mined      rephrased  -   seed=43,rephrase=in-context
text        siglip-mined      rephrased  -   seed=44,rephrase=in-context
text        cosent            rephrased  -   seed=43,rephrase=in-context
text        cosent            rephrased  -   seed=44,rephrase=in-context
text        mse               rephrased  -   seed=43,rephrase=in-context
text        mse               rephrased  -   seed=44,rephrase=in-context
multimodal  siglip-mined      rephrased  -   seed=43,rephrase=in-context
multimodal  siglip-mined      rephrased  -   seed=44,rephrase=in-context
multimodal  cosent            rephrased  -   seed=43,rephrase=in-context
multimodal  cosent            rephrased  -   seed=44,rephrase=in-context
multimodal  mse               rephrased  -   seed=43,rephrase=in-context
multimodal  mse               rephrased  -   seed=44,rephrase=in-context
# nv-mined on in-context: the _rephrased-in-context datasets mined with the selected variant
# (logs/mine/run_incontext_m0.025_s10.sh), infonce-mined x 3 seeds, like the plain nv-mined rows.
# text        infonce-mined     rephrased  -   negs=mined,mining=m0.025_s10,rephrase=in-context
# text        infonce-mined     rephrased  -   negs=mined,mining=m0.025_s10,seed=43,rephrase=in-context
# text        infonce-mined     rephrased  -   negs=mined,mining=m0.025_s10,seed=44,rephrase=in-context
# multimodal  infonce-mined     rephrased  -   negs=mined,mining=m0.025_s10,rephrase=in-context
# multimodal  infonce-mined     rephrased  -   negs=mined,mining=m0.025_s10,seed=43,rephrase=in-context
# multimodal  infonce-mined     rephrased  -   negs=mined,mining=m0.025_s10,seed=44,rephrase=in-context
# 2026-09-12: re-selected on the in-context sweep (below): text m0.1_s10, image m0.05_s0.
# The m0.025_s10 rows above carried the plain-rephrased selection and are kept for reference.
text        infonce-mined     rephrased  -   negs=mined,mining=m0.1_s10,rephrase=in-context
text        infonce-mined     rephrased  -   negs=mined,mining=m0.1_s10,seed=43,rephrase=in-context
text        infonce-mined     rephrased  -   negs=mined,mining=m0.1_s10,seed=44,rephrase=in-context
multimodal infonce-mined rephrased - negs=mined,mining=m0.2_s10,rephrase=in-context
multimodal infonce-mined rephrased - negs=mined,mining=m0.2_s10,seed=43,rephrase=in-context
multimodal infonce-mined rephrased - negs=mined,mining=m0.2_s10,seed=44,rephrase=in-context
# -------------------------------------------------------------------------
# 50/50 mixed negatives (2026-09-11), in-context only: the mix_hard_negs.py sibling
# (_mixed-rephrased_m0.025_s10) keeps our labeled negative on a seeded half of the train-split
# hard rows and takes the nv-mined (m0.025_s10) one on the other half. infonce-ours-v3 grades
# the labeled half; the mined half has no measured distance and is labeled at the easy
# distance (--unmeasured-negatives easy), so those rows train one-hot, as under infonce-mined.
# V is re-swept on validation (recall@5 text / recall@20 image) since the training
# distribution changed; analysis.ipynb section 2 draws the sweep and section 5 gates the
# test rows on MIXED_SELECTED. Fill V in from the argmax and uncomment the 3-seed rows.
# Two groups: ours-nv-mixed (one shuffle over the mix) and ours-nv-ordered (order=mined-first:
# each epoch trains every batch of the nv-mined queries before any batch of ours).
# -------------------------------------------------------------------------
text        infonce-ours-v3   rephrased  10  negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
text        infonce-ours-v3   rephrased  20  negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
text        infonce-ours-v3   rephrased  40  negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
text        infonce-ours-v3   rephrased  80  negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
multimodal  infonce-ours-v3   rephrased  10  negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
multimodal  infonce-ours-v3   rephrased  20  negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
multimodal  infonce-ours-v3   rephrased  40  negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
multimodal  infonce-ours-v3   rephrased  80  negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
text        infonce-ours-v3   rephrased  10  negs=mixed,mining=m0.025_s10,rephrase=in-context,order=mined-first,split=val
text        infonce-ours-v3   rephrased  20  negs=mixed,mining=m0.025_s10,rephrase=in-context,order=mined-first,split=val
text        infonce-ours-v3   rephrased  40  negs=mixed,mining=m0.025_s10,rephrase=in-context,order=mined-first,split=val
text        infonce-ours-v3   rephrased  80  negs=mixed,mining=m0.025_s10,rephrase=in-context,order=mined-first,split=val
multimodal  infonce-ours-v3   rephrased  10  negs=mixed,mining=m0.025_s10,rephrase=in-context,order=mined-first,split=val
multimodal  infonce-ours-v3   rephrased  20  negs=mixed,mining=m0.025_s10,rephrase=in-context,order=mined-first,split=val
multimodal  infonce-ours-v3   rephrased  40  negs=mixed,mining=m0.025_s10,rephrase=in-context,order=mined-first,split=val
multimodal  infonce-ours-v3   rephrased  80  negs=mixed,mining=m0.025_s10,rephrase=in-context,order=mined-first,split=val
# Selected V (2026-09-12, val argmax: text 20, image 80, both groups): the seed-42 row shares its model with the sweep cell and is inference only.
text        infonce-ours-v3   rephrased  20  negs=mixed,mining=m0.025_s10,rephrase=in-context
text        infonce-ours-v3   rephrased  20  negs=mixed,mining=m0.025_s10,seed=43,rephrase=in-context
text        infonce-ours-v3   rephrased  20  negs=mixed,mining=m0.025_s10,seed=44,rephrase=in-context
multimodal infonce-ours-v3 rephrased 80 negs=mixed,mining=m0.025_s10,rephrase=in-context
multimodal infonce-ours-v3 rephrased 80 negs=mixed,mining=m0.025_s10,seed=43,rephrase=in-context
multimodal infonce-ours-v3 rephrased 80 negs=mixed,mining=m0.025_s10,seed=44,rephrase=in-context
text        infonce-ours-v3   rephrased  20  negs=mixed,mining=m0.025_s10,rephrase=in-context,order=mined-first
text        infonce-ours-v3   rephrased  20  negs=mixed,mining=m0.025_s10,seed=43,rephrase=in-context,order=mined-first
text        infonce-ours-v3   rephrased  20  negs=mixed,mining=m0.025_s10,seed=44,rephrase=in-context,order=mined-first
multimodal infonce-ours-v3 rephrased 80 negs=mixed,mining=m0.025_s10,rephrase=in-context,order=mined-first
multimodal infonce-ours-v3 rephrased 80 negs=mixed,mining=m0.025_s10,seed=43,rephrase=in-context,order=mined-first
multimodal infonce-ours-v3 rephrased 80 negs=mixed,mining=m0.025_s10,seed=44,rephrase=in-context,order=mined-first
# -------------------------------------------------------------------------
# Full in-context result set for the mse / cosent / siglip families (2026-09-13), matching
# infonce's: 3 seeds on the graded and ungraded styles; the ungraded loss on the NV-mined
# in-context negatives at the variant selected on the infonce sweep (text m0.1_s10, image
# m0.05_s0; not re-selected per loss); the graded loss on the 50/50 mix, V re-swept on the
# mixed validation split (easy fixed at the style's selection), then 3 seeds at the argmax.
# -------------------------------------------------------------------------
# seeds 43/44 on the existing in-context rows
text        ours-mse-batched  rephrased  40  easy=10,seed=43,rephrase=in-context
text        ours-mse-batched  rephrased  40  easy=10,seed=44,rephrase=in-context
multimodal ours-mse-batched rephrased 20 easy=10,seed=43,rephrase=in-context
multimodal ours-mse-batched rephrased 20 easy=10,seed=44,rephrase=in-context
# archived/retired: text        ours-siglip       rephrased  20  easy=10,seed=43,rephrase=in-context
# archived/retired: text        ours-siglip       rephrased  20  easy=10,seed=44,rephrase=in-context
# archived/retired: multimodal  ours-siglip       rephrased  40  easy=10,seed=43,rephrase=in-context
# archived/retired: multimodal  ours-siglip       rephrased  40  easy=10,seed=44,rephrase=in-context
text        ours-cosent       rephrased  -   seed=43,rephrase=in-context
text        ours-cosent       rephrased  -   seed=44,rephrase=in-context
multimodal  ours-cosent       rephrased  -   seed=43,rephrase=in-context
multimodal  ours-cosent       rephrased  -   seed=44,rephrase=in-context
text        mse-mined         rephrased  40  easy=10,seed=43,rephrase=in-context
text        mse-mined         rephrased  40  easy=10,seed=44,rephrase=in-context
multimodal  mse-mined         rephrased  80  easy=10,seed=43,rephrase=in-context
multimodal  mse-mined         rephrased  80  easy=10,seed=44,rephrase=in-context
text        cosent            rephrased  -   seed=43,rephrase=in-context
text        cosent            rephrased  -   seed=44,rephrase=in-context
multimodal  cosent            rephrased  -   seed=43,rephrase=in-context
multimodal  cosent            rephrased  -   seed=44,rephrase=in-context
text        siglip-mined      rephrased  -   seed=43,rephrase=in-context
text        siglip-mined      rephrased  -   seed=44,rephrase=in-context
multimodal  siglip-mined      rephrased  -   seed=43,rephrase=in-context
multimodal  siglip-mined      rephrased  -   seed=44,rephrase=in-context
# nv-mined group: the ungraded loss on the NV-mined in-context negatives, 3 seeds
text        mse-mined         rephrased  40  easy=10,negs=mined,mining=m0.1_s10,rephrase=in-context
text        mse-mined         rephrased  40  easy=10,negs=mined,mining=m0.1_s10,seed=43,rephrase=in-context
text        mse-mined         rephrased  40  easy=10,negs=mined,mining=m0.1_s10,seed=44,rephrase=in-context
multimodal mse-mined rephrased 80 easy=10,negs=mined,mining=m0.2_s10,rephrase=in-context
multimodal mse-mined rephrased 80 easy=10,negs=mined,mining=m0.2_s10,seed=43,rephrase=in-context
multimodal mse-mined rephrased 80 easy=10,negs=mined,mining=m0.2_s10,seed=44,rephrase=in-context
text        cosent            rephrased  -   negs=mined,mining=m0.1_s10,rephrase=in-context
text        cosent            rephrased  -   negs=mined,mining=m0.1_s10,seed=43,rephrase=in-context
text        cosent            rephrased  -   negs=mined,mining=m0.1_s10,seed=44,rephrase=in-context
multimodal cosent rephrased - negs=mined,mining=m0.2_s10,rephrase=in-context
multimodal cosent rephrased - negs=mined,mining=m0.2_s10,seed=43,rephrase=in-context
multimodal cosent rephrased - negs=mined,mining=m0.2_s10,seed=44,rephrase=in-context
text        siglip-mined      rephrased  -   negs=mined,mining=m0.1_s10,rephrase=in-context
text        siglip-mined      rephrased  -   negs=mined,mining=m0.1_s10,seed=43,rephrase=in-context
text        siglip-mined      rephrased  -   negs=mined,mining=m0.1_s10,seed=44,rephrase=in-context
multimodal siglip-mined rephrased - negs=mined,mining=m0.2_s10,rephrase=in-context
multimodal siglip-mined rephrased - negs=mined,mining=m0.2_s10,seed=43,rephrase=in-context
multimodal siglip-mined rephrased - negs=mined,mining=m0.2_s10,seed=44,rephrase=in-context
# mixed group: V sweep on the mixed in-context validation split (ours-cosent has no V).
# The graded style per family is the newest one: ours-mse-batched, siglip-v3, ours-cosent.
text        ours-mse-batched  rephrased  10  easy=10,negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
text        ours-mse-batched  rephrased  20  easy=10,negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
text        ours-mse-batched  rephrased  40  easy=10,negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
text        ours-mse-batched  rephrased  80  easy=10,negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
multimodal  ours-mse-batched  rephrased  10  easy=10,negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
multimodal  ours-mse-batched  rephrased  20  easy=10,negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
multimodal  ours-mse-batched  rephrased  40  easy=10,negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
multimodal  ours-mse-batched  rephrased  80  easy=10,negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
# 2026-09-15: the siglip family's mixed group moved to siglip-v3, so its mixed bar uses the
# family's newest graded style the way infonce's uses infonce-ours-v3. The ours-siglip sweep
# below ran 2026-09-13 and is superseded; its models stay in models/ unused.
# text        ours-siglip       rephrased  10  easy=10,negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
# text        ours-siglip       rephrased  20  easy=10,negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
# text        ours-siglip       rephrased  40  easy=10,negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
# text        ours-siglip       rephrased  80  easy=10,negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
# multimodal  ours-siglip       rephrased  10  easy=10,negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
# multimodal  ours-siglip       rephrased  20  easy=10,negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
# multimodal  ours-siglip       rephrased  40  easy=10,negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
# multimodal  ours-siglip       rephrased  80  easy=10,negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
# siglip-v3 has no easy axis (random and cross-row cells target 0), so its mixed sweep is V alone.
text        siglip-v3         rephrased  10  negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
text        siglip-v3         rephrased  20  negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
text        siglip-v3         rephrased  40  negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
text        siglip-v3         rephrased  80  negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
multimodal  siglip-v3         rephrased  10  negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
multimodal  siglip-v3         rephrased  20  negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
multimodal  siglip-v3         rephrased  40  negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
multimodal  siglip-v3         rephrased  80  negs=mixed,mining=m0.025_s10,rephrase=in-context,split=val
# mixed group: 3 seeds, V selected 2026-09-15 on the mixed val sweep above (ours-cosent has no V):
# ours-mse-batched text 20 / image 80, siglip-v3 text 20 / image 80.
text        ours-cosent       rephrased  -   negs=mixed,mining=m0.025_s10,rephrase=in-context
text        ours-cosent       rephrased  -   negs=mixed,mining=m0.025_s10,seed=43,rephrase=in-context
text        ours-cosent       rephrased  -   negs=mixed,mining=m0.025_s10,seed=44,rephrase=in-context
multimodal  ours-cosent       rephrased  -   negs=mixed,mining=m0.025_s10,rephrase=in-context
multimodal  ours-cosent       rephrased  -   negs=mixed,mining=m0.025_s10,seed=43,rephrase=in-context
multimodal  ours-cosent       rephrased  -   negs=mixed,mining=m0.025_s10,seed=44,rephrase=in-context
text        ours-mse-batched  rephrased  20  easy=10,negs=mixed,mining=m0.025_s10,rephrase=in-context
text        ours-mse-batched  rephrased  20  easy=10,negs=mixed,mining=m0.025_s10,seed=43,rephrase=in-context
text        ours-mse-batched  rephrased  20  easy=10,negs=mixed,mining=m0.025_s10,seed=44,rephrase=in-context
multimodal ours-mse-batched rephrased 80 easy=10,negs=mixed,mining=m0.025_s10,rephrase=in-context
multimodal ours-mse-batched rephrased 80 easy=10,negs=mixed,mining=m0.025_s10,seed=43,rephrase=in-context
multimodal ours-mse-batched rephrased 80 easy=10,negs=mixed,mining=m0.025_s10,seed=44,rephrase=in-context
text        siglip-v3         rephrased  20  negs=mixed,mining=m0.025_s10,rephrase=in-context
text        siglip-v3         rephrased  20  negs=mixed,mining=m0.025_s10,seed=43,rephrase=in-context
text        siglip-v3         rephrased  20  negs=mixed,mining=m0.025_s10,seed=44,rephrase=in-context
multimodal siglip-v3 rephrased 40 negs=mixed,mining=m0.025_s10,rephrase=in-context
multimodal siglip-v3 rephrased 40 negs=mixed,mining=m0.025_s10,seed=43,rephrase=in-context
multimodal siglip-v3 rephrased 40 negs=mixed,mining=m0.025_s10,seed=44,rephrase=in-context
# -------------------------------------------------------------------------
# Random-negative control (2026-09-13), in-context only: each family's ungraded style on the
# random_hard_negs.py sibling, where every train-split hard negative is a uniform random
# product. Shows what any hard negative (ours or NV-mined) adds over none. 3 seeds.
# -------------------------------------------------------------------------
text        infonce-mined     rephrased  -   negs=random,rephrase=in-context
text        infonce-mined     rephrased  -   negs=random,seed=43,rephrase=in-context
text        infonce-mined     rephrased  -   negs=random,seed=44,rephrase=in-context
multimodal  infonce-mined     rephrased  -   negs=random,rephrase=in-context
multimodal  infonce-mined     rephrased  -   negs=random,seed=43,rephrase=in-context
multimodal  infonce-mined     rephrased  -   negs=random,seed=44,rephrase=in-context
text        mse-mined         rephrased  40  easy=10,negs=random,rephrase=in-context
text        mse-mined         rephrased  40  easy=10,negs=random,seed=43,rephrase=in-context
text        mse-mined         rephrased  40  easy=10,negs=random,seed=44,rephrase=in-context
multimodal  mse-mined         rephrased  80  easy=10,negs=random,rephrase=in-context
multimodal  mse-mined         rephrased  80  easy=10,negs=random,seed=43,rephrase=in-context
multimodal  mse-mined         rephrased  80  easy=10,negs=random,seed=44,rephrase=in-context
text        cosent            rephrased  -   negs=random,rephrase=in-context
text        cosent            rephrased  -   negs=random,seed=43,rephrase=in-context
text        cosent            rephrased  -   negs=random,seed=44,rephrase=in-context
multimodal  cosent            rephrased  -   negs=random,rephrase=in-context
multimodal  cosent            rephrased  -   negs=random,seed=43,rephrase=in-context
multimodal  cosent            rephrased  -   negs=random,seed=44,rephrase=in-context
text        siglip-mined      rephrased  -   negs=random,rephrase=in-context
text        siglip-mined      rephrased  -   negs=random,seed=43,rephrase=in-context
text        siglip-mined      rephrased  -   negs=random,seed=44,rephrase=in-context
multimodal  siglip-mined      rephrased  -   negs=random,rephrase=in-context
multimodal  siglip-mined      rephrased  -   negs=random,seed=43,rephrase=in-context
multimodal  siglip-mined      rephrased  -   negs=random,seed=44,rephrase=in-context
# Baseline loss selection on random negatives (2026-09-14, analysis.ipynb section 4): Table 1
# compares the ungraded losses trained on the random_hard_negs.py sibling, not on our labeled
# hard negative, so that the baseline choice does not depend on our data. Validation split.
# infonce-mined / cosent / siglip-mined reuse the random-control weights above (inference only);
# mse (pairwise, the table's MSE row) has no random-control test row, so its rows train here.
text        infonce-mined     rephrased  -   split=val,negs=random,rephrase=in-context
text        infonce-mined     rephrased  -   split=val,negs=random,seed=43,rephrase=in-context
text        infonce-mined     rephrased  -   split=val,negs=random,seed=44,rephrase=in-context
multimodal  infonce-mined     rephrased  -   split=val,negs=random,rephrase=in-context
multimodal  infonce-mined     rephrased  -   split=val,negs=random,seed=43,rephrase=in-context
multimodal  infonce-mined     rephrased  -   split=val,negs=random,seed=44,rephrase=in-context
text        cosent            rephrased  -   split=val,negs=random,rephrase=in-context
text        cosent            rephrased  -   split=val,negs=random,seed=43,rephrase=in-context
text        cosent            rephrased  -   split=val,negs=random,seed=44,rephrase=in-context
multimodal  cosent            rephrased  -   split=val,negs=random,rephrase=in-context
multimodal  cosent            rephrased  -   split=val,negs=random,seed=43,rephrase=in-context
multimodal  cosent            rephrased  -   split=val,negs=random,seed=44,rephrase=in-context
text        siglip-mined      rephrased  -   split=val,negs=random,rephrase=in-context
text        siglip-mined      rephrased  -   split=val,negs=random,seed=43,rephrase=in-context
text        siglip-mined      rephrased  -   split=val,negs=random,seed=44,rephrase=in-context
multimodal  siglip-mined      rephrased  -   split=val,negs=random,rephrase=in-context
multimodal  siglip-mined      rephrased  -   split=val,negs=random,seed=43,rephrase=in-context
multimodal  siglip-mined      rephrased  -   split=val,negs=random,seed=44,rephrase=in-context
text        mse               rephrased  -   split=val,negs=random,rephrase=in-context
text        mse               rephrased  -   split=val,negs=random,seed=43,rephrase=in-context
text        mse               rephrased  -   split=val,negs=random,seed=44,rephrase=in-context
multimodal  mse               rephrased  -   split=val,negs=random,rephrase=in-context
multimodal  mse               rephrased  -   split=val,negs=random,seed=43,rephrase=in-context
multimodal  mse               rephrased  -   split=val,negs=random,seed=44,rephrase=in-context
# -------------------------------------------------------------------------
# siglip-v3 (2026-09-14): ours-siglip with the exponential target (utils/graded_losses.py,
# exponential=True). Like infonce-ours-v3, easy plays no part (random and cross-row cells
# target 0), so the sweep is V alone, on the in-context rephrased validation split -- the
# paper's rephrased kind -- at recall@5 text / recall@20 image. Fill V in from the argmax and
# uncomment the 3-seed rows; the seed-42 row shares its model with the sweep cell.
# Selected 2026-09-14: V=80 both modalities (val argmax; monotone in V, grid edge).
# -------------------------------------------------------------------------
text        siglip-v3         rephrased  10  split=val,rephrase=in-context
text        siglip-v3         rephrased  20  split=val,rephrase=in-context
text        siglip-v3         rephrased  40  split=val,rephrase=in-context
text        siglip-v3         rephrased  80  split=val,rephrase=in-context
multimodal  siglip-v3         rephrased  10  split=val,rephrase=in-context
multimodal  siglip-v3         rephrased  20  split=val,rephrase=in-context
multimodal  siglip-v3         rephrased  40  split=val,rephrase=in-context
multimodal  siglip-v3         rephrased  80  split=val,rephrase=in-context
text        siglip-v3         rephrased  80  rephrase=in-context
text        siglip-v3         rephrased  80  seed=43,rephrase=in-context
text        siglip-v3         rephrased  80  seed=44,rephrase=in-context
multimodal siglip-v3 rephrased 80 rephrase=in-context
multimodal siglip-v3 rephrased 80 seed=43,rephrase=in-context
multimodal siglip-v3 rephrased 80 seed=44,rephrase=in-context

# In-context labelled validation sweeps (seed 42); selection precedes main trials.
text        infonce-ours-v3 rephrased 10 split=val,rephrase=in-context
text        infonce-ours-v3 rephrased 20 split=val,rephrase=in-context
text        infonce-ours-v3 rephrased 40 split=val,rephrase=in-context
text        infonce-ours-v3 rephrased 80 split=val,rephrase=in-context
text        ours-mse-batched rephrased 20 easy=10,split=val,rephrase=in-context
text        ours-mse-batched rephrased 40 easy=10,split=val,rephrase=in-context
text        ours-mse-batched rephrased 80 easy=10,split=val,rephrase=in-context
text        ours-mse-batched rephrased 20 easy=20,split=val,rephrase=in-context
text        ours-mse-batched rephrased 40 easy=20,split=val,rephrase=in-context
text        ours-mse-batched rephrased 80 easy=20,split=val,rephrase=in-context
text        ours-mse-batched rephrased 20 easy=40,split=val,rephrase=in-context
text        ours-mse-batched rephrased 40 easy=40,split=val,rephrase=in-context
text        ours-mse-batched rephrased 80 easy=40,split=val,rephrase=in-context
text        mse-mined          rephrased 40 split=val,rephrase=in-context,easy=10
text        mse-mined          rephrased 40 split=val,rephrase=in-context,easy=10,seed=43
text        mse-mined          rephrased 40 split=val,rephrase=in-context,easy=10,seed=44
text        ours-cosent        rephrased - split=val,rephrase=in-context
text        ours-cosent        rephrased - split=val,rephrase=in-context,seed=43
text        ours-cosent        rephrased - split=val,rephrase=in-context,seed=44
multimodal  infonce-ours-v3 rephrased 10 split=val,rephrase=in-context
multimodal  infonce-ours-v3 rephrased 20 split=val,rephrase=in-context
multimodal  infonce-ours-v3 rephrased 40 split=val,rephrase=in-context
multimodal  infonce-ours-v3 rephrased 80 split=val,rephrase=in-context
multimodal  ours-mse-batched rephrased 20 easy=10,split=val,rephrase=in-context
multimodal  ours-mse-batched rephrased 40 easy=10,split=val,rephrase=in-context
multimodal  ours-mse-batched rephrased 80 easy=10,split=val,rephrase=in-context
multimodal  ours-mse-batched rephrased 20 easy=20,split=val,rephrase=in-context
multimodal  ours-mse-batched rephrased 40 easy=20,split=val,rephrase=in-context
multimodal  ours-mse-batched rephrased 80 easy=20,split=val,rephrase=in-context
multimodal  ours-mse-batched rephrased 20 easy=40,split=val,rephrase=in-context
multimodal  ours-mse-batched rephrased 40 easy=40,split=val,rephrase=in-context
multimodal  ours-mse-batched rephrased 80 easy=40,split=val,rephrase=in-context
multimodal  mse-mined          rephrased 80 split=val,rephrase=in-context,easy=10
multimodal  mse-mined          rephrased 80 split=val,rephrase=in-context,easy=10,seed=43
multimodal  mse-mined          rephrased 80 split=val,rephrase=in-context,easy=10,seed=44
multimodal  ours-cosent        rephrased - split=val,rephrase=in-context
multimodal  ours-cosent        rephrased - split=val,rephrase=in-context,seed=43
multimodal  ours-cosent        rephrased - split=val,rephrase=in-context,seed=44
"

# ---------------------------------------------------------------------------
# GPU pool scheduler: one background job per GPU slot.
# ---------------------------------------------------------------------------
# Initialise rather than only declare: under `set -u` a declared-but-never-assigned
# array is still unbound, so ${#PID_GPU[@]} in drain() aborts the run whenever a
# phase launches no jobs at all — exactly what happens on a re-run where everything
# is already trained.
declare -A PID_GPU=() PID_DESC=() PID_KEY=() PID_LOG=() PID_PHASE=()
declare -A TRAIN_STATUS=() TEST_STATUS=() HUMAN_STATUS=() RUN_DIRS=()
MISSING_DATASETS=()

# Background jobs started with & in a non-interactive shell have SIGINT set to ignore
# (POSIX), so Ctrl-C kills the scheduler but every training job survives as an orphan.
# Forward the interrupt to the tracked jobs explicitly, then exit.
interrupt() {
  trap - INT TERM
  local pids=("${!PID_GPU[@]}")
  if ((${#pids[@]})); then
    echo >&2
    echo "Interrupted -- stopping ${#pids[@]} running job(s)" >&2
    kill -TERM "${pids[@]}" 2>/dev/null || true
    sleep 3
    kill -KILL "${pids[@]}" 2>/dev/null || true
  fi
  exit 130
}
trap interrupt INT TERM
FREE_GPUS=($GPUS)

reap_one() {
  local pid="" st=0
  wait -n -p pid || st=$?
  if [[ -z "$pid" ]]; then
    PID_GPU=()
    return 0
  fi
  FREE_GPUS+=("${PID_GPU[$pid]}")
  local key=${PID_KEY[$pid]} phase=${PID_PHASE[$pid]}
  local outcome
  if [[ $st -eq 0 ]]; then
    outcome=written
    [[ $phase == train ]] && outcome=trained
    echo "[done] ${PID_DESC[$pid]}"
  else
    outcome="FAILED($st)"
    echo "!! FAILED (exit $st): ${PID_DESC[$pid]} — see ${PID_LOG[$pid]}"
  fi
  case $phase in
    train) TRAIN_STATUS[$key]=$outcome ;;
    test)  TEST_STATUS[$key]=$outcome ;;
    human) HUMAN_STATUS[$key]=$outcome ;;
  esac
  unset "PID_GPU[$pid]" "PID_DESC[$pid]" "PID_KEY[$pid]" "PID_LOG[$pid]" "PID_PHASE[$pid]"
  return 0
}

launch() { # phase key desc logfile cmd...
  local phase=$1 key=$2 desc=$3 logf=$4
  shift 4
  while [[ ${#FREE_GPUS[@]} -eq 0 ]]; do reap_one; done
  local gpu=${FREE_GPUS[0]}
  FREE_GPUS=("${FREE_GPUS[@]:1}")
  echo "[gpu $gpu] $desc"
  echo "  log: $logf"
  CUDA_VISIBLE_DEVICES=$gpu "$@" >"$logf" 2>&1 &
  local pid=$!
  PID_GPU[$pid]=$gpu PID_DESC[$pid]=$desc PID_KEY[$pid]=$key PID_LOG[$pid]=$logf PID_PHASE[$pid]=$phase
}

drain() {
  while [[ ${#PID_GPU[@]} -gt 0 ]]; do reap_one; done
}

# ---------------------------------------------------------------------------
# Per-condition derivations
# ---------------------------------------------------------------------------
model_for()   { [[ $1 == text ]] && echo "$TEXT_MODEL" || echo "$IMG_MODEL"; }
# The human-query eval set: a row subset of the base dataset with human_query filled in and
# every row split=test. One per modality, shared by every condition whatever it trained on.
human_dataset_for() { # modality [rephrase] -> dataset dir
  # rephrase=in-context rows score on the human set minus the style examples their rephraser
  # saw (download_human_labels.py writes both).
  local base; [[ $1 == text ]] && base="${TEXT_DATASET}_human" || base="${IMG_DATASET}_human"
  echo "${base}${2:+-$2}"
}
dataset_for() { # modality [query_kind] [negs] [mining] [rephrase] -> dataset dir
  local base
  [[ $1 == text ]] && base=$TEXT_DATASET || base=$IMG_DATASET
  # The rephrased queries live in a sibling dataset built by rephrase_dataset.py; it carries the
  # same rows, split column and labels, with rephrased_query filled in.
  [[ ${2:-} == rephrased ]] && base="${base}_rephrased"
  # rephrase=in-context: the rephrase_dataset.py --in-context sibling, whose prompt carried
  # human-written style examples (human_study/in_context_examples_<modality>.json).
  [[ ${2:-} == rephrased && -n ${5:-} ]] && base="${base}-${5}"
  # negs=mined: the sibling built by mine_hard_negs.py for this query kind. Same rows and
  # split; only the train split's hard negatives differ (retrieval-mined, unmeasured distance).
  [[ ${3:-labeled} == mined ]] && base="${base}_mined-${2}"
  # mining=<variant>: a mine_hard_negs.py --variant sibling (mining sweep); no suffix is the
  # default config (relative margin 0.05, first survivor, fallback weakest).
  [[ ${3:-labeled} == mined && -n ${4:-} ]] && base="${base}_${4}"
  # negs=mined-graded: the mined sibling after label_mined_negs.py measured every mined
  # negative's query_distance, so the graded losses can train on it.
  [[ ${3:-labeled} == mined-graded ]] && base="${base}_mined-${2}_graded"
  # negs=mixed: the mix_hard_negs.py sibling, half labeled and half mined train negatives;
  # mining= names the mined sibling it drew from, as for negs=mined.
  [[ ${3:-labeled} == mixed ]] && base="${base}_mixed-${2}"
  [[ ${3:-labeled} == mixed && -n ${4:-} ]] && base="${base}_${4}"
  # negs=random: the random_hard_negs.py sibling, every train-split hard negative replaced by a
  # uniform random product -- the control for hard-negative mining of any kind.
  [[ ${3:-labeled} == random ]] && base="${base}_random-${2}"
  echo "$base"
}

run_name_for() { # modality style query_kind V extra
  local modality=$1 style=$2 qk=$3 v=$4 extra=$5
  local model_short easy="" transform="" split=test negs=labeled mining="" seed="" rephrase="" order=""
  model_short=$(basename "$(model_for "$modality")")
  # split is parsed but deliberately NOT part of the name: a val row and its test twin
  # share one model dir, and only their preds subdir differs. negs is not a token either:
  # it selects the dataset, whose tag already carries the _mined-<kind> suffix.
  parse_extra "$extra" easy transform split negs mining seed rephrase order
  local name="${modality}__${model_short}__${style}__$(basename "$(dataset_for "$modality" "$qk" "$negs" "$mining" "$rephrase")")__${qk}"
  # Token order must match build_run_name extras order: easy, V, transform, order, seed, note.
  if [[ -n $easy ]]; then name+="__easy-${easy}"; fi
  if [[ $v != - ]]; then name+="__V-${v}"; fi
  if [[ -n $transform ]]; then name+="__transform-${transform}"; fi
  if [[ -n $order ]]; then name+="__order-${order}"; fi
  # seed=<n> names a repeated trial; the trainer default 42 carries no token (train.py name_extras).
  if [[ -n $seed && $seed != 42 ]]; then name+="__seed-${seed}"; fi
  name+="__note-${NOTE}"
  echo "$name"
}

parse_extra() { # extra_string easy_var transform_var split_var negs_var [mining_var] [seed_var] [rephrase_var] [order_var]
  local extra=$1 token
  local -n _easy=$2 _transform=$3 _split=$4 _negs=$5
  local _mining_unused
  local -n _mining=${6:-_mining_unused}
  local _seed_unused
  local -n _seed=${7:-_seed_unused}
  local _rephrase_unused
  local -n _rephrase=${8:-_rephrase_unused}
  local _order_unused
  local -n _order=${9:-_order_unused}
  _easy="" _transform="" _split=test _negs=labeled _mining="" _seed="" _rephrase="" _order=""
  if [[ $extra == - ]]; then return 0; fi
  IFS=, read -ra tokens <<<"$extra"
  for token in "${tokens[@]}"; do
    case $token in
      easy=*) _easy=${token#easy=} ;;
      transform=*) _transform=${token#transform=} ;;
      split=*) _split=${token#split=} ;;
      negs=*) _negs=${token#negs=} ;;
      mining=*) _mining=${token#mining=} ;;
      seed=*) _seed=${token#seed=} ;;
      rephrase=*) _rephrase=${token#rephrase=} ;;
      order=*) _order=${token#order=} ;;
      *) echo "Unsupported extra '$token' (supported: easy=, transform=, split=, negs=, mining=, seed=, rephrase=, order=)" >&2; exit 1 ;;
    esac
  done
  case $_split in
    test|val) ;;
    *) echo "Unsupported split '$_split' (supported: test, val)" >&2; exit 1 ;;
  esac
  case $_negs in
    labeled|mined|mined-graded|mixed|random) ;;
    *) echo "Unsupported negs '$_negs' (supported: labeled, mined, mined-graded, mixed, random)" >&2; exit 1 ;;
  esac
}

train_cmd_for() { # modality style query_kind V extra run_dir -> echoes full command
  local modality=$1 style=$2 qk=$3 v=$4 extra=$5 run_dir=$6
  local easy="" transform="" split=test negs=labeled mining="" seed="" rephrase="" order=""
  parse_extra "$extra" easy transform split negs mining seed rephrase order
  local cmd="$PY -u train.py --modality $modality --training-style $style --dataset $(dataset_for "$modality" "$qk" "$negs" "$mining" "$rephrase") --output-dir $run_dir --note $NOTE --query-kind $qk $TRAIN_COMMON $REPORT_TO $WANDB_ARGS"
  if [[ $v != - ]]; then cmd+=" --V $v"; fi
  if [[ -n $easy ]]; then cmd+=" --easy-negative-value $easy"; fi
  if [[ -n $transform ]]; then cmd+=" --distance-transform $transform"; fi
  if [[ -n $seed ]]; then cmd+=" --seed $seed"; fi
  # negs=mixed: the mined half has no measured distance; a graded loss labels it at the
  # easy-negative distance (target mass 0, one-hot like infonce-mined on those rows).
  if [[ $negs == mixed ]]; then cmd+=" --unmeasured-negatives easy"; fi
  if [[ -n $order ]]; then cmd+=" --train-order $order"; fi
  if [[ $modality == multimodal && -n $IMG_TRAIN_EXTRA ]]; then cmd+=" $IMG_TRAIN_EXTRA"; fi
  echo "$cmd"
}

# ---------------------------------------------------------------------------
# Build the plan
# ---------------------------------------------------------------------------
KEYS=()
declare -A K_MODALITY=() K_STYLE=() K_QK=() K_V=() K_EXTRA=() K_TRAIN_ACTION=() K_SPLIT=() K_NEGS=() K_MINING=() K_REPHRASE=() SEEN_RUN_DIR=()

while read -r modality style qk v extra; do
  [[ -z $modality || $modality == \#* ]] && continue
  run_name=$(run_name_for "$modality" "$style" "$qk" "$v" "$extra")
  [[ -n $ONLY && ! $run_name =~ $ONLY ]] && continue
  # The split is an evaluation choice, not a training one: a val row and its test twin are
  # the same weights scored on a different split. The key carries the split so both can sit
  # in the plan, while run_dir does not, so the second one reuses the first one's model.
  row_split=""; row_easy=""; row_transform=""; row_negs=""; row_mining=""; row_rephrase=""
  parse_extra "$extra" row_easy row_transform row_split row_negs row_mining row_seed_unused row_rephrase
  if [[ $qk != rephrased || $row_rephrase != in-context ]]; then
    echo "ERROR: active paper conditions must use rephrased-in-context: $run_name" >&2
    exit 1
  fi
  key=$run_name
  [[ $row_split == val ]] && key="$run_name@val"
  run_dir=$MODELS_ROOT/$run_name
  KEYS+=("$key")
  RUN_DIRS[$key]=$run_dir
  K_SPLIT[$key]=$row_split
  K_NEGS[$key]=$row_negs
  K_MINING[$key]=$row_mining
  K_REPHRASE[$key]=$row_rephrase
  K_MODALITY[$key]=$modality K_STYLE[$key]=$style K_QK[$key]=$qk K_V[$key]=$v K_EXTRA[$key]=$extra

  if [[ $style == untrained ]]; then
    K_TRAIN_ACTION[$key]=none
    TRAIN_STATUS[$key]="n/a"
    continue
  fi

  # A val row whose model another key already trains must not queue that training twice.
  if [[ -n ${SEEN_RUN_DIR[$run_name]:-} ]]; then
    K_TRAIN_ACTION[$key]=none
    TRAIN_STATUS[$key]="shared with ${SEEN_RUN_DIR[$run_name]}"
    continue
  fi
  SEEN_RUN_DIR[$run_name]=$key

  marker=$run_dir/final/modules.json
  dataset=$(dataset_for "$modality" "$qk" "$row_negs" "$row_mining" "$row_rephrase")
  # Collect missing datasets and abort after the plan prints, so every one shows at once.
  if [[ ! -d $dataset ]]; then
    MISSING_DATASETS+=("$key -> $dataset")
  fi
  if [[ $FORCE_TRAIN == 1 ]]; then
    K_TRAIN_ACTION[$key]=train; TRAIN_STATUS[$key]="queued (forced)"
  elif [[ ! -f $marker ]]; then
    K_TRAIN_ACTION[$key]=train; TRAIN_STATUS[$key]="queued (missing)"
  else
    K_TRAIN_ACTION[$key]=skip; TRAIN_STATUS[$key]="reused"
  fi
done <<<"$CONDITIONS"

# Phase 1 and 2 both launch in KEYS order, so priority is applied once, here. The sort is
# stable on the original index, so conditions of equal priority keep their table order.
style_priority() {
  local i
  for i in "${!PRIORITY_STYLES[@]}"; do
    [[ ${PRIORITY_STYLES[i]} == "$1" ]] && { echo "$i"; return; }
  done
  echo "${#PRIORITY_STYLES[@]}"
}

mapfile -t KEYS < <(
  for i in "${!KEYS[@]}"; do
    printf '%s\t%06d\t%s\n' "$(style_priority "${K_STYLE[${KEYS[i]}]}")" "$i" "${KEYS[i]}"
  done | sort -k1,1n -k2,2n | cut -f3-
)

echo
echo "== Plan (NOTE=$NOTE, root=$MODELS_ROOT, GPUS=$GPUS) =="
for key in "${KEYS[@]}"; do
  printf '  %-9s %s\n' "[${K_TRAIN_ACTION[$key]}]" "$key"
done
echo

# Every human eval set a test row of the plan will score on: one per (modality, rephrase).
declare -A HUMAN_NEEDED=()
for key in "${KEYS[@]}"; do
  [[ $EVAL_HUMAN != 1 || ${K_SPLIT[$key]} == val ]] && continue
  HUMAN_NEEDED["${K_MODALITY[$key]}|${K_REPHRASE[$key]}"]=1
done
for need in "${!HUMAN_NEEDED[@]}"; do
  human_dataset=$(human_dataset_for "${need%%|*}" "${need#*|}")
  [[ -d $human_dataset ]] || MISSING_DATASETS+=("human eval (${need%%|*}) -> $human_dataset")
done
if ((${#MISSING_DATASETS[@]})); then
  echo
  echo "ERROR: ${#MISSING_DATASETS[@]} condition(s) point at a dataset that does not exist:" >&2
  printf '  %s\n' "${MISSING_DATASETS[@]}" >&2
  echo "Build them before running (rephrased datasets come from rephrase_dataset.sh, the" >&2
  echo "human sets from human_study/download_human_labels.py)." >&2
  exit 1
fi

# Refuse to start a run the disk cannot hold: ~0.5G per text model, ~0.6G per image
# model, plus 2G for preds and slack. Override with SKIP_DISK_CHECK=1.
n_text_train=0 n_img_train=0
for key in "${KEYS[@]}"; do
  [[ ${K_TRAIN_ACTION[$key]} == train ]] || continue
  [[ ${K_MODALITY[$key]} == text ]] && ((n_text_train += 1)) || ((n_img_train += 1))
done
need_gb=$(( (n_text_train * 5 + n_img_train * 6 + 20 + 9) / 10 ))
free_gb=$(( $(df -Pk "$MODELS_ROOT" | awk 'NR==2{print $4}') / 1024 / 1024 ))
echo "Disk: ${free_gb}G free, ~${need_gb}G needed for $((n_text_train + n_img_train)) trainings ($n_text_train text + $n_img_train image)"
if [[ ${SKIP_DISK_CHECK:-0} != 1 && $free_gb -lt $need_gb ]]; then
  echo "ERROR: not enough disk for this plan. Free space or SKIP_DISK_CHECK=1." >&2
  [[ $DRY_RUN == 1 ]] || exit 1
fi

if [[ $DRY_RUN == 1 ]]; then
  echo "DRY_RUN=1 — exiting without running anything."
  exit 0
fi

# ---------------------------------------------------------------------------
# Phase 1: train
# ---------------------------------------------------------------------------
echo "== Phase 1: training =="
for key in "${KEYS[@]}"; do
  [[ ${K_TRAIN_ACTION[$key]} == train ]] || continue
  cmd=$(train_cmd_for "${K_MODALITY[$key]}" "${K_STYLE[$key]}" "${K_QK[$key]}" "${K_V[$key]}" "${K_EXTRA[$key]}" "${RUN_DIRS[$key]}")
  launch train "$key" "train $key" "$LOG_DIR/$key.train.log" $cmd
done
drain
echo "== Phase 1 done =="
echo

# ---------------------------------------------------------------------------
# Phase 2: test-set inference, plus the human-query eval for every test row
# ---------------------------------------------------------------------------
preds_reusable() { # meta model_marker -> 0 if the preds at meta were made from the model on disk
  local meta=$1 model_marker=$2
  [[ $FORCE_TEST != 1 && -f $meta ]] && [[ -z $model_marker || ! $model_marker -nt $meta ]]
}

echo "== Phase 2: inference =="
for key in "${KEYS[@]}"; do
  modality=${K_MODALITY[$key]} style=${K_STYLE[$key]} run_dir=${RUN_DIRS[$key]}
  # Must be ${K_QK[$key]}, not $qk: $qk is a leftover global from the Phase-1 `while read`
  # loop and holds the LAST condition line's query_kind for every iteration here. Using it
  # pointed every rephrased condition at the non-rephrased dataset, whose rephrased_query
  # column is empty -- multimodal then died on "Need at least 3 unique queries" and text
  # silently wrote preds for one empty query.
  dataset=$(dataset_for "$modality" "${K_QK[$key]}" "${K_NEGS[$key]}" "${K_MINING[$key]}" "${K_REPHRASE[$key]}")
  test_script=test.py

  if [[ $style == untrained ]]; then
    model_path=$(model_for "$modality")
    model_marker=""   # nothing local to compare against
  else
    if [[ ${TRAIN_STATUS[$key]} == FAILED* ]]; then
      TEST_STATUS[$key]="skipped (training failed)"
      continue
    fi
    model_path=$run_dir/final
    model_marker=$run_dir/final/modules.json
    if [[ ! -f $model_marker ]]; then
      TEST_STATUS[$key]="skipped (no model)"
      continue
    fi
  fi

  if [[ ${K_SPLIT[$key]} == val ]]; then
    preds_subdir=preds_val; split_arg="--split validation"
  else
    preds_subdir=preds; split_arg="--split test"
  fi
  if preds_reusable "$run_dir/$preds_subdir/meta.json" "$model_marker"; then
    TEST_STATUS[$key]="reused"
  else
    launch test "$key" "test  $key" "$LOG_DIR/$key.test.log" \
      $PY -u "$test_script" --modality "$modality" --model-path "$model_path" --dataset "$dataset" \
      --query-kind "${K_QK[$key]}" --run-dir "$run_dir" --top-k "$TOP_K" $split_arg
  fi

  # Human queries are a test set, so val rows (hparam sweeps) do not score them: selecting
  # on a test set is what the val split exists to prevent.
  if [[ $EVAL_HUMAN != 1 || ${K_SPLIT[$key]} == val ]]; then
    HUMAN_STATUS[$key]="not requested"
    continue
  fi
  human_dataset=$(human_dataset_for "$modality" "${K_REPHRASE[$key]}")
  if preds_reusable "$run_dir/preds_human/meta.json" "$model_marker"; then
    HUMAN_STATUS[$key]="reused"
  else
    launch human "$key" "human $key" "$LOG_DIR/$key.human.log" \
      $PY -u "$test_script" --modality "$modality" --model-path "$model_path" --dataset "$human_dataset" \
      --query-kind human --run-dir "$run_dir" --top-k "$TOP_K" --split test \
      --distractor-dataset "$dataset" --distractor-query-kind "${K_QK[$key]}"
  fi
done
drain
echo "== Phase 2 done =="
echo

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
fail=0
echo "== Summary (NOTE=$NOTE) =="
printf '%-11s %-17s %-10s %-4s %-24s %-24s %-16s\n' modality style query V train preds human
for key in "${KEYS[@]}"; do
  t=${TRAIN_STATUS[$key]:-"?"} p=${TEST_STATUS[$key]:-"?"} h=${HUMAN_STATUS[$key]:-"?"}
  printf '%-11s %-17s %-10s %-4s %-24s %-24s %-16s\n' \
    "${K_MODALITY[$key]}" "${K_STYLE[$key]}" "${K_QK[$key]}" "${K_V[$key]}" "$t" "$p" "$h"
  [[ $t == FAILED* || $p == FAILED* || $p == skipped* || $h == FAILED* ]] && fail=1
done
echo
echo "Preds live in <run_dir>/preds/ (human queries: <run_dir>/preds_human/); logs in $LOG_DIR/"
if [[ $fail == 1 ]]; then
  echo "Some conditions FAILED or were skipped — see above."
  exit 1
fi
echo "All conditions complete and up to date."
