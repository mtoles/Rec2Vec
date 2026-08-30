#!/usr/bin/env bash
# paper.sh — single entry point that produces every result in the paper.
#
# Phase 1 trains every missing model, in parallel across GPUs (one GPU per job;
# DDP/NCCL is broken on this node). Phase 2 runs test-set inference for every
# condition via test_text.py / test_multimodal.py, writing preds jsonl files into
# each model's run dir so all analysis can happen offline. This script does NOT
# preprocess data.
#
# Freshness rules (no stale results masquerading as current):
#   - A model is reused only if <run_dir>/final/modules.json exists AND is newer
#     than its dataset. A dataset regenerated after training marks the model stale
#     and it is retrained (override with ALLOW_STALE=1).
#   - If a train script is newer than a model, a warning is printed (retrain with
#     FORCE_TRAIN=1 if the change affects training).
#   - Preds are regenerated when missing or older than the model, the test script,
#     or the dataset. Every preds/ dir carries meta.json with git SHA + mtimes.
#
# Knobs (env):
#   GPUS="0 1 2"   GPUs to schedule on (default: 0-7)
#   NOTE=paper     experiment tag; part of every run dir name
#   SMOKE=1        tiny data, models/_smoke root, NOTE=smoke, wandb offline
#   DRY_RUN=1      print the plan and exit
#   FORCE_TRAIN=1  retrain even if fresh      FORCE_TEST=1  re-infer even if fresh
#   ALLOW_STALE=1  keep models older than their dataset

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
ALLOW_STALE=${ALLOW_STALE:-0}

TEXT_MODEL=sentence-transformers/all-mpnet-base-v2
IMG_MODEL=sentence-transformers/clip-ViT-B-32

if [[ $SMOKE == 1 ]]; then
  NOTE=${NOTE:-smoke}
  MODELS_ROOT=${MODELS_ROOT:-models/_smoke}
  TEXT_DATASET=${TEXT_DATASET:-/local/data/mt/recall/dataset/processed/feature-distance-dataset_gemini-2.5-flash-lite_10000}
  IMG_DATASET=${IMG_DATASET:-dataset/processed/deepfashion-inshop-image-triplets_hf_20000}
  IMG_TRAIN_EXTRA="--train-fraction 0.1"
  TOP_K=20
  export WANDB_MODE=offline
  REPORT_TO="--report-to none"
  WANDB_ARGS=""
else
  NOTE=${NOTE:-paper}
  MODELS_ROOT=${MODELS_ROOT:-models}
  TEXT_DATASET=${TEXT_DATASET:-dataset/processed/feature-distance-dataset_gemini-2.5-flash_1000000_nolek}
  IMG_DATASET=${IMG_DATASET:-dataset/processed/deepfashion-inshop-image-triplets_hf_20000}
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
#   style:    untrained | baseline-triplet | infonce | infonce-mined | cosent | classic-mse
#             | ours-mse | ours-mse-batched
#             infonce is the standard 2-column objective (in-batch negatives only);
#             infonce-mined adds our labeled hard negative as a third column.
#             | ours-mse-batched (ours over the full in-batch candidate pool)
#             (| ours-mse-reversed)
#   V:        distance normalizer, '-' = not applicable (untrained, triplet, infonce,
#             and cosent -- all baselines that never see the measured distance)
#   extra:    '-' or comma-separated key=value; supported: easy=<int>, transform=<name>
# The image dataset only has synthetic (nl_query) queries, so multimodal rows are
# synthetic-only. The V ablation runs on synthetic queries for both modalities.
# ---------------------------------------------------------------------------
CONDITIONS="
# Main grid: {text, multimodal} x {original, synthetic, rephrased} x {baselines, ours}.
# V=40 for every ours-mse row: it wins the validation sweep on both modalities
# (text ndcg@10 .7630 / acc@1 .6177; multimodal .1522 / .0640). See the ablation below.
text        untrained         original   -   -
text        untrained         synthetic  -   -
text        untrained         rephrased  -   -
text        baseline-triplet  original   -   -
text        baseline-triplet  synthetic  -   -
text        baseline-triplet  rephrased  -   -
text        infonce           original   -   -
text        infonce-mined     original   -   -
text        infonce           synthetic  -   -
text        infonce-mined     synthetic  -   -
text        infonce           rephrased  -   -
text        infonce-mined     rephrased  -   -
text        cosent            original   -   -
text        cosent            synthetic  -   -
text        cosent            rephrased  -   -
text        classic-mse       original   40  -
text        classic-mse       synthetic  40  -
text        classic-mse       rephrased  40  -
text        ours-mse          original   40  -
text        ours-mse          synthetic  40  -
text        ours-mse          rephrased  40  -
text        ours-mse-batched  original   40  -
text        ours-mse-batched  synthetic  40  -
text        ours-mse-batched  rephrased  40  -
# The image dataset has no real search queries (original_query is empty for all 12,957 rows),
# so multimodal runs synthetic and rephrased only.
multimodal  untrained         synthetic  -   -
multimodal  untrained         rephrased  -   -
multimodal  baseline-triplet  synthetic  -   -
multimodal  baseline-triplet  rephrased  -   -
multimodal  infonce           synthetic  -   -
multimodal  infonce-mined     synthetic  -   -
multimodal  infonce           rephrased  -   -
multimodal  infonce-mined     rephrased  -   -
multimodal  cosent            synthetic  -   -
multimodal  cosent            rephrased  -   -
multimodal  classic-mse       synthetic  40  -
multimodal  classic-mse       rephrased  40  -
multimodal  ours-mse          synthetic  40  -
multimodal  ours-mse          rephrased  40  -
multimodal  ours-mse-batched  synthetic  40  -
multimodal  ours-mse-batched  rephrased  40  -

# ---------------------------------------------------------------------------
# V ablation: 20/40/60 on synthetic queries for both ours variants. The V=40 point
# is the main-grid row above, so it is not repeated here.
# ---------------------------------------------------------------------------
text        ours-mse          synthetic  20  -
text        ours-mse          synthetic  60  -
text        ours-mse-batched  synthetic  20  -
text        ours-mse-batched  synthetic  60  -
multimodal  ours-mse          synthetic  20  -
multimodal  ours-mse          synthetic  60  -
multimodal  ours-mse-batched  synthetic  20  -
multimodal  ours-mse-batched  synthetic  60  -
"

# ---------------------------------------------------------------------------
# GPU pool scheduler: one background job per GPU slot.
# ---------------------------------------------------------------------------
# Initialise rather than only declare: under `set -u` a declared-but-never-assigned
# array is still unbound, so ${#PID_GPU[@]} in drain() aborts the run whenever a
# phase launches no jobs at all — exactly what happens on a re-run where everything
# is already trained.
declare -A PID_GPU=() PID_DESC=() PID_KEY=() PID_LOG=() PID_PHASE=()
declare -A TRAIN_STATUS=() TEST_STATUS=() RUN_DIRS=()
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
  if [[ $st -eq 0 ]]; then
    if [[ $phase == train ]]; then TRAIN_STATUS[$key]=trained; else TEST_STATUS[$key]=written; fi
    echo "[done] ${PID_DESC[$pid]}"
  else
    if [[ $phase == train ]]; then TRAIN_STATUS[$key]="FAILED($st)"; else TEST_STATUS[$key]="FAILED($st)"; fi
    echo "!! FAILED (exit $st): ${PID_DESC[$pid]} — see ${PID_LOG[$pid]}"
  fi
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
dataset_for() { # modality [query_kind] -> dataset dir
  local base
  [[ $1 == text ]] && base=$TEXT_DATASET || base=$IMG_DATASET
  # The rephrased queries live in a sibling dataset built by rephrase_dataset.py; it carries the
  # same rows, split column and labels, with rephrased_query filled in.
  [[ ${2:-} == rephrased ]] && echo "${base}_rephrased" || echo "$base"
}

file_mtime() { # path -> epoch seconds, 0 if absent
  [[ -e $1 ]] && stat -c %Y "$1" || echo 0
}

# `datasets` writes cache-*.arrow into the dataset dir on every load, so the dir's
# own mtime tracks the last time something *read* it, not the last time it was
# regenerated. Using it for staleness makes each finished training mark all the
# earlier ones stale. Look at the payload files only.
dataset_mtime() { # dataset_dir -> epoch seconds of newest payload file, 0 if none
  local dir=$1 newest=0 m f
  for f in "$dir"/data-*.arrow "$dir"/dataset_info.json "$dir"/state.json; do
    [[ -f $f ]] || continue
    m=$(stat -c %Y "$f")
    if ((m > newest)); then newest=$m; fi
  done
  echo "$newest"
}

run_name_for() { # modality style query_kind V extra
  local modality=$1 style=$2 qk=$3 v=$4 extra=$5
  local model_short easy="" transform=""
  model_short=$(basename "$(model_for "$modality")")
  parse_extra "$extra" easy transform
  local name="${modality}__${model_short}__${style}__$(basename "$(dataset_for "$modality" "$qk")")__${qk}"
  # Token order must match build_run_name extras order: easy, V, transform, note.
  if [[ -n $easy ]]; then name+="__easy-${easy}"; fi
  if [[ $v != - ]]; then name+="__V-${v}"; fi
  if [[ -n $transform ]]; then name+="__transform-${transform}"; fi
  name+="__note-${NOTE}"
  echo "$name"
}

parse_extra() { # extra_string easy_var transform_var
  local extra=$1 token
  local -n _easy=$2 _transform=$3
  _easy="" _transform=""
  if [[ $extra == - ]]; then return 0; fi
  IFS=, read -ra tokens <<<"$extra"
  for token in "${tokens[@]}"; do
    case $token in
      easy=*) _easy=${token#easy=} ;;
      transform=*) _transform=${token#transform=} ;;
      *) echo "Unsupported extra '$token' (supported: easy=, transform=)" >&2; exit 1 ;;
    esac
  done
}

train_cmd_for() { # modality style query_kind V extra run_dir -> echoes full command
  local modality=$1 style=$2 qk=$3 v=$4 extra=$5 run_dir=$6
  local easy="" transform=""
  parse_extra "$extra" easy transform
  local cmd="$PY -u train.py --modality $modality --training-style $style --dataset $(dataset_for "$modality" "$qk") --output-dir $run_dir --note $NOTE --query-kind $qk $TRAIN_COMMON $REPORT_TO $WANDB_ARGS"
  if [[ $v != - ]]; then cmd+=" --V $v"; fi
  if [[ -n $easy ]]; then cmd+=" --easy-negative-value $easy"; fi
  if [[ -n $transform ]]; then cmd+=" --distance-transform $transform"; fi
  if [[ $modality == multimodal && -n $IMG_TRAIN_EXTRA ]]; then cmd+=" $IMG_TRAIN_EXTRA"; fi
  echo "$cmd"
}

# ---------------------------------------------------------------------------
# Build the plan
# ---------------------------------------------------------------------------
KEYS=()
declare -A K_MODALITY=() K_STYLE=() K_QK=() K_V=() K_EXTRA=() K_TRAIN_ACTION=()

while read -r modality style qk v extra; do
  [[ -z $modality || $modality == \#* ]] && continue
  key=$(run_name_for "$modality" "$style" "$qk" "$v" "$extra")
  run_dir=$MODELS_ROOT/$key
  KEYS+=("$key")
  RUN_DIRS[$key]=$run_dir
  K_MODALITY[$key]=$modality K_STYLE[$key]=$style K_QK[$key]=$qk K_V[$key]=$v K_EXTRA[$key]=$extra

  if [[ $style == untrained ]]; then
    K_TRAIN_ACTION[$key]=none
    TRAIN_STATUS[$key]="n/a"
    continue
  fi

  marker=$run_dir/final/modules.json
  dataset=$(dataset_for "$modality" "$qk")
  train_script=train.py
  # A missing dataset makes dataset_mtime 0, which reads as "older than the model" and
  # silently skips the condition -- the same shape of failure as the Phase-2 $qk bug.
  # Collect them and abort after the plan prints, so every missing dataset shows at once.
  if [[ ! -d $dataset ]]; then
    MISSING_DATASETS+=("$key -> $dataset")
  fi
  if [[ $FORCE_TRAIN == 1 ]]; then
    K_TRAIN_ACTION[$key]=train; TRAIN_STATUS[$key]="queued (forced)"
  elif [[ ! -f $marker ]]; then
    K_TRAIN_ACTION[$key]=train; TRAIN_STATUS[$key]="queued (missing)"
  elif (( $(dataset_mtime "$dataset") > $(file_mtime "$marker") )) && [[ $ALLOW_STALE != 1 ]]; then
    echo "STALE: $key — dataset newer than model; retraining (ALLOW_STALE=1 to keep)"
    K_TRAIN_ACTION[$key]=train; TRAIN_STATUS[$key]="queued (stale dataset)"
  else
    if [[ $train_script -nt $marker ]]; then
      echo "WARNING: $train_script is newer than $key/final — FORCE_TRAIN=1 if the change affects training"
    fi
    K_TRAIN_ACTION[$key]=skip; TRAIN_STATUS[$key]="reused"
  fi
done <<<"$CONDITIONS"

echo
echo "== Plan (NOTE=$NOTE, root=$MODELS_ROOT, GPUS=$GPUS) =="
for key in "${KEYS[@]}"; do
  printf '  %-9s %s\n' "[${K_TRAIN_ACTION[$key]}]" "$key"
done
echo

if ((${#MISSING_DATASETS[@]})); then
  echo
  echo "ERROR: ${#MISSING_DATASETS[@]} condition(s) point at a dataset that does not exist:" >&2
  printf '  %s\n' "${MISSING_DATASETS[@]}" >&2
  echo "Build them before running (rephrased datasets come from rephrase_dataset.sh)." >&2
  exit 1
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
# Phase 2: test-set inference
# ---------------------------------------------------------------------------
echo "== Phase 2: inference =="
for key in "${KEYS[@]}"; do
  modality=${K_MODALITY[$key]} style=${K_STYLE[$key]} run_dir=${RUN_DIRS[$key]}
  # Must be ${K_QK[$key]}, not $qk: $qk is a leftover global from the Phase-1 `while read`
  # loop and holds the LAST condition line's query_kind for every iteration here. Using it
  # pointed every rephrased condition at the non-rephrased dataset, whose rephrased_query
  # column is empty -- multimodal then died on "Need at least 3 unique queries" and text
  # silently wrote preds for one empty query.
  dataset=$(dataset_for "$modality" "${K_QK[$key]}")
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

  meta=$run_dir/preds/meta.json
  if [[ $FORCE_TEST != 1 && -f $meta ]] \
     && [[ -z $model_marker || ! $model_marker -nt $meta ]] \
     && [[ ! $test_script -nt $meta ]] \
     && (( $(dataset_mtime "$dataset") <= $(file_mtime "$meta") )); then
    TEST_STATUS[$key]="reused"
    continue
  fi

  launch test "$key" "test  $key" "$LOG_DIR/$key.test.log" \
    $PY -u "$test_script" --modality "$modality" --model-path "$model_path" --dataset "$dataset" \
    --query-kind "${K_QK[$key]}" --run-dir "$run_dir" --top-k "$TOP_K"
done
drain
echo "== Phase 2 done =="
echo

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
fail=0
echo "== Summary (NOTE=$NOTE) =="
printf '%-11s %-17s %-10s %-4s %-24s %-24s\n' modality style query V train preds
for key in "${KEYS[@]}"; do
  t=${TRAIN_STATUS[$key]:-"?"} p=${TEST_STATUS[$key]:-"?"}
  printf '%-11s %-17s %-10s %-4s %-24s %-24s\n' \
    "${K_MODALITY[$key]}" "${K_STYLE[$key]}" "${K_QK[$key]}" "${K_V[$key]}" "$t" "$p"
  [[ $t == FAILED* || $p == FAILED* || $p == skipped* ]] && fail=1
done
echo
echo "Preds live in <run_dir>/preds/; logs in $LOG_DIR/"
if [[ $fail == 1 ]]; then
  echo "Some conditions FAILED or were skipped — see above."
  exit 1
fi
echo "All conditions complete and up to date."
