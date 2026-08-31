"""Unified trainer for the text and multimodal retrieval models.

One script for both modalities, selected with --modality {text,multimodal}. Everything a
condition needs is shared -- dataset preparation, label construction, loss selection,
trainer setup, final evaluation -- and the two genuinely modality-specific concerns are
isolated below: how documents are encoded (strings vs image paths loaded as PIL images)
and which base model is the default. Splitting is data-driven rather than per modality:
a dataset with a precomputed `split` column (the leakage-free text split) uses it, and
one without (the image dataset) gets the seeded query-level split.

Supersedes train_text.py and train_multimodal.py.
"""

from collections import defaultdict
from enum import Enum
import argparse
import json
import os
from typing import Any, Dict, Iterable, List

from datasets import Dataset, Value, load_from_disk
from PIL import Image
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    losses,
    util,
)
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from sentence_transformers.training_args import BatchSamplers
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch
import wandb
import yaml

from utils.distance_transform import DistanceTransform, transform_normalized_distance
from utils.graded_losses import BatchGradedMarginMSELoss, GradedInfoNCELoss, GradedSigLIPLoss
from utils.distance_labels import (
    DEFAULT_MAX_DISTANCE, default_easy_negative_distance, to_training_labels,
)
from utils.run_naming import build_output_dir, build_run_name

# Which dataset column each query kind trains on.
QUERY_COLUMNS = {
    "original": "original_query",
    "synthetic": "nl_query",
    "rephrased": "rephrased_query",
}

DEFAULT_MODELS = {
    "text": "sentence-transformers/all-mpnet-base-v2",
    "multimodal": "sentence-transformers/clip-ViT-B-32",
}


class TrainingStyle(Enum):
    BASELINE_TRIPLET = "baseline-triplet"
    INFONCE = "infonce"
    INFONCE_MINED = "infonce-mined"
    SIGLIP_MINED = "siglip-mined"
    COSENT = "cosent"
    OURS_MSE = "ours-mse"
    OURS_MSE_BATCHED = "ours-mse-batched"
    OURS_INFONCE = "ours-infonce"
    OURS_SIGLIP = "ours-siglip"
    OURS_MSE_REVERSED = "ours-mse-reversed"
    CLASSIC_MSE = "classic-mse"


TRIPLET_STYLES = (
    TrainingStyle.BASELINE_TRIPLET.value,
    TrainingStyle.INFONCE.value,
    TrainingStyle.INFONCE_MINED.value,
    TrainingStyle.SIGLIP_MINED.value,
    TrainingStyle.COSENT.value,
)
LABELED_STYLES = (
    TrainingStyle.OURS_MSE.value,
    TrainingStyle.OURS_MSE_BATCHED.value,
    TrainingStyle.OURS_INFONCE.value,
    TrainingStyle.OURS_SIGLIP.value,
    TrainingStyle.OURS_MSE_REVERSED.value,
    TrainingStyle.CLASSIC_MSE.value,
)
PAIR_STYLES = (TrainingStyle.CLASSIC_MSE.value, TrainingStyle.COSENT.value)


# ---------------------------------------------------------------------------
# Image handling -- the multimodal side's document encoding
# ---------------------------------------------------------------------------

def load_rgb_image(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB").copy()


def add_image_transform(dataset: Dataset, image_columns: Iterable[str]) -> Dataset:
    image_columns = tuple(image_columns)

    def transform(batch: Dict[str, List[Any]]) -> Dict[str, List[Any]]:
        batch = dict(batch)
        for column in image_columns:
            if column in batch:
                batch[column] = [load_rgb_image(path) for path in batch[column]]
        return batch

    return dataset.with_transform(transform)


class ImageCorpusInformationRetrievalEvaluator(InformationRetrievalEvaluator):
    """InformationRetrievalEvaluator that treats document/corpus values as image paths."""

    def embed_inputs(
        self,
        model: SentenceTransformer,
        sentences,
        encode_fn_name: str | None = None,
        prompt_name: str | None = None,
        prompt: str | None = None,
        **kwargs,
    ):
        if encode_fn_name == "document":
            sentences = [load_rgb_image(path) if isinstance(path, str) else path for path in sentences]
        return super().embed_inputs(
            model,
            sentences,
            encode_fn_name=encode_fn_name,
            prompt_name=prompt_name,
            prompt=prompt,
            **kwargs,
        )


def encode_documents(
    model: SentenceTransformer,
    documents: List[str],
    modality: str,
    batch_size: int,
    show_progress_bar: bool,
):
    """Embed documents: strings directly for text, image paths loaded as PIL for multimodal."""
    if modality == "text":
        return model.encode(
            documents,
            batch_size=batch_size,
            convert_to_tensor=True,
            show_progress_bar=show_progress_bar,
        )

    embeddings = []
    iterator = range(0, len(documents), batch_size)
    if show_progress_bar:
        iterator = tqdm(iterator, desc="Encoding images")
    for start in iterator:
        images = [load_rgb_image(path) for path in documents[start : start + batch_size]]
        embeddings.append(
            model.encode(images, batch_size=batch_size, convert_to_tensor=True, show_progress_bar=False)
        )
        for image in images:
            image.close()
    return torch.cat(embeddings, dim=0) if embeddings else torch.empty(0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def prep_ds_for_ir_eval(dataset: Dataset, query_key: str, pos_key: str, neg_key: str, show_progress: bool = True):
    corpus_items = sorted(set(list(dataset[pos_key]) + list(dataset[neg_key])))
    corpus = {str(i): item for i, item in enumerate(corpus_items)}
    reverse_corpus = {item: cid for cid, item in corpus.items()}

    queries = {}
    relevant_docs = defaultdict(set)
    query_to_id = {}

    iterator = tqdm(range(len(dataset)), desc="Preparing IR evaluation") if show_progress else range(len(dataset))
    for i in iterator:
        ex = dataset[i]
        query = ex[query_key]
        if query not in query_to_id:
            query_id = str(len(query_to_id))
            query_to_id[query] = query_id
            queries[query_id] = query
        relevant_docs[query_to_id[query]].add(reverse_corpus[ex[pos_key]])

    return queries, corpus, {key: set(value) for key, value in relevant_docs.items()}


def evaluate_model(
    model: SentenceTransformer,
    dataset: Dataset,
    modality: str,
    split_name: str = "val",
    batch_size: int = 32,
    log_predictions_table_name: str | None = None,
    output_dir: str | None = None,
    wandb_step: int | None = None,
):
    print(f"Evaluating {modality} retrieval on {split_name}...")
    results = {}

    query_embeddings = model.encode(
        dataset["anchor"], batch_size=batch_size, convert_to_tensor=True, show_progress_bar=True
    )
    positive_embeddings = encode_documents(model, dataset["positive"], modality, batch_size, True)
    negative_embeddings = encode_documents(model, dataset["negative"], modality, batch_size, True)

    pos_cosine_scores = torch.nn.functional.cosine_similarity(query_embeddings, positive_embeddings)
    neg_cosine_scores = torch.nn.functional.cosine_similarity(query_embeddings, negative_embeddings)
    is_triplet_correct = (pos_cosine_scores > neg_cosine_scores).float()

    results["avg_pos_cosine_sim"] = torch.mean(pos_cosine_scores).item()
    results["avg_neg_cosine_sim"] = torch.mean(neg_cosine_scores).item()
    results["manual_triplet_cosine_accuracy"] = torch.mean(is_triplet_correct).item()
    print(f"Average Positive Cosine Similarity: {results['avg_pos_cosine_sim']}")
    print(f"Average Negative Cosine Similarity: {results['avg_neg_cosine_sim']}")
    print(f"Manual Triplet Cosine Accuracy: {results['manual_triplet_cosine_accuracy']}")

    if "negative_example_source" in dataset.column_names:
        easy_mask = torch.tensor(
            [source == "random" for source in dataset["negative_example_source"]],
            device=neg_cosine_scores.device,
        )
        hard_mask = ~easy_mask
        if torch.any(easy_mask):
            results["avg_easy_neg_cosine_sim"] = torch.mean(neg_cosine_scores[easy_mask]).item()
            results["triplet_cosine_accuracy_easy"] = torch.mean(is_triplet_correct[easy_mask]).item()
            print(f"Easy Triplet Cosine Accuracy: {results['triplet_cosine_accuracy_easy']}")
        if torch.any(hard_mask):
            results["avg_hard_neg_cosine_sim"] = torch.mean(neg_cosine_scores[hard_mask]).item()
            results["triplet_cosine_accuracy_hard"] = torch.mean(is_triplet_correct[hard_mask]).item()
            print(f"Hard Triplet Cosine Accuracy: {results['triplet_cosine_accuracy_hard']}")

    queries, corpus, relevant_docs = prep_ds_for_ir_eval(dataset, "anchor", "positive", "negative")
    ks = [1, 5, 10, 50, 100, 1000]
    ks = sorted({min(k, max(1, len(corpus))) for k in ks})

    evaluator_cls = (
        InformationRetrievalEvaluator if modality == "text" else ImageCorpusInformationRetrievalEvaluator
    )
    ir_evaluator = evaluator_cls(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        mrr_at_k=ks,
        map_at_k=ks,
        precision_recall_at_k=ks,
        ndcg_at_k=ks,
        accuracy_at_k=ks,
        batch_size=batch_size,
        show_progress_bar=True,
        write_predictions=bool(log_predictions_table_name),
        name=split_name,
    )
    ir_results = ir_evaluator(model)
    print("Information Retrieval Evaluator:")
    print(ir_results)
    # ST prefixes its keys with the evaluator name and metric family, e.g.
    # 'val_cosine_accuracy@1'; strip both so keys read 'val/accuracy@1' after prefixing.
    results.update({
        key.replace(f"{split_name}_", "", 1).replace("cosine_", ""): value
        for key, value in ir_results.items()
    })

    prefixed_results = {f"{split_name}/{key}": value for key, value in results.items()}

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, f"eval_{split_name}_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(prefixed_results, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Saved {split_name} metrics to {metrics_path}")

    if wandb.run is not None:
        # The final retrieval evaluator runs outside the HF trainer. Update both history
        # and summary so the metrics remain visible even with no later trainer logs.
        wandb.log(prefixed_results, step=wandb_step, commit=True)
        wandb.run.summary.update(prefixed_results)
        if output_dir is not None:
            wandb.save(metrics_path, policy="now")

    return results


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def prepare_dataset_for_trainer(dataset: Dataset, swap_pos_neg: bool = False) -> Dataset:
    columns_to_remove = [
        "query_distance",
        "negative_example_source",
        "distance_source",
        "positive_product_id",
        "negative_product_id",
        "item",
        "positive_id",
        "negative_id",
        "positive_category",
        "negative_category",
        "selected_pos_features",
        "selected_neg_features",
        "selected_common_features",
        "selected_neither_features",
        "full_common_features",
        "full_unique_pos_features",
        "full_unique_neg_features",
        "full_neither_features",
    ]
    removable = [column for column in columns_to_remove if column in dataset.column_names]
    if removable:
        dataset = dataset.remove_columns(removable)
    if swap_pos_neg:
        # MarginMSELoss reads columns positionally (query, positive, negative) and fits
        # sim(q, col2) - sim(q, col3) to the label, so swapping the column order flips the
        # sign of the predicted margin without touching the labels.
        assert {"anchor", "positive", "negative"} <= set(dataset.column_names), (
            f"Cannot swap positive/negative, columns are {dataset.column_names}"
        )
        reordered = ["anchor", "negative", "positive"]
        reordered += [column for column in dataset.column_names if column not in reordered]
        dataset = dataset.select_columns(reordered)
    return dataset


def drop_mined_negative(dataset: Dataset) -> Dataset:
    """(anchor, positive, negative) -> (anchor, positive), for plain infonce.

    MultipleNegativesRankingLoss pools every column after the anchor into one softmax, so
    keeping `negative` is what turns the standard objective into the hard-negative-mined one.
    The baseline must see in-batch negatives only, which means the column has to go before the
    trainer sees it. The evaluator reads raw_eval_dataset and still has its negatives.
    """
    assert {"anchor", "positive"} <= set(dataset.column_names), dataset.column_names
    return dataset.select_columns(["anchor", "positive"])


def split_dataset(dataset: Dataset, seed: int):
    """Precomputed leakage-free `split` column when present, seeded query split otherwise."""
    if "split" in dataset.column_names:
        train_dataset = dataset.filter(lambda x: x["split"] == "train")
        eval_dataset = dataset.filter(lambda x: x["split"] == "validation")
        test_dataset = dataset.filter(lambda x: x["split"] == "test")
        aux_cols = [c for c in ("split",) if c in dataset.column_names]
        return (
            train_dataset.remove_columns(aux_cols),
            eval_dataset.remove_columns(aux_cols),
            test_dataset.remove_columns(aux_cols),
        )

    unique_queries = list(dict.fromkeys(dataset["anchor"]))
    if len(unique_queries) < 3:
        raise ValueError("Need at least 3 unique queries for train/eval/test split")
    train_queries, temp_queries = train_test_split(unique_queries, test_size=0.2, random_state=seed)
    eval_queries, test_queries = train_test_split(temp_queries, test_size=0.5, random_state=seed)
    train_queries, eval_queries, test_queries = set(train_queries), set(eval_queries), set(test_queries)
    return (
        dataset.filter(lambda x: x["anchor"] in train_queries),
        dataset.filter(lambda x: x["anchor"] in eval_queries),
        dataset.filter(lambda x: x["anchor"] in test_queries),
    )


def select_query_fraction(dataset: Dataset, fraction: float, seed: int, split_name: str) -> Dataset:
    if not 0 < fraction <= 1:
        raise ValueError(f"train_fraction must be in (0, 1], got {fraction}")
    if fraction == 1:
        return dataset

    unique_queries = list(dict.fromkeys(dataset["anchor"]))
    selected_count = max(1, int(round(len(unique_queries) * fraction)))
    selected_queries, _ = train_test_split(unique_queries, train_size=selected_count, random_state=seed)
    selected_queries = set(selected_queries)
    selected = dataset.filter(lambda x: x["anchor"] in selected_queries)
    print(
        f"Using {len(selected)}/{len(dataset)} {split_name} examples "
        f"from {selected_count}/{len(unique_queries)} queries (fraction={fraction})"
    )
    return selected


def build_pair_dataset(dataset: Dataset, is_cosent: bool) -> Dataset:
    """(anchor, positive, negative[, label]) rows -> (sentence_A, sentence_B, label) pairs.

    CoSENT is a baseline and never sees the measured distance: its negatives are labelled 0,
    so only the positive/negative split reaches the loss. classic-mse regresses the graded
    label. CoSENT also scores every pair in a batch against every other, so the shared
    positive is emitted from the hard row only; CosineSimilarityLoss scores each pair
    independently and keeps both copies.
    """
    dedup_positive = is_cosent and "negative_example_source" in dataset.column_names

    def to_pairs(batch):
        n = len(batch["anchor"])
        sources = batch["negative_example_source"]
        neg_labels = [0.0] * n if is_cosent else [1.0 - l for l in batch["label"]]
        anchors, others, labels = [], [], []
        for anchor, positive, negative, neg_label, source in zip(
            batch["anchor"], batch["positive"], batch["negative"], neg_labels, sources
        ):
            if not (dedup_positive and source == "random"):
                anchors.append(anchor)
                others.append(positive)
                labels.append(1.0)
            anchors.append(anchor)
            others.append(negative)
            labels.append(neg_label)
        return {"sentence_A": anchors, "sentence_B": others, "label": labels}

    return dataset.map(to_pairs, batched=True, remove_columns=dataset.column_names)


def build_loss(model: SentenceTransformer, training_style: str, easy_label: float | None = None,
               batch_size: int | None = None):
    if training_style == TrainingStyle.BASELINE_TRIPLET.value:
        return losses.TripletLoss(
            model=model, distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=0.2
        )
    if training_style in (TrainingStyle.INFONCE.value, TrainingStyle.INFONCE_MINED.value):
        # Cross-entropy over in-batch candidates. The loss object is the same for both;
        # what differs is how many columns reach it (see drop_mined_negative):
        #   infonce       (anchor, positive)           -> B candidates, all in-batch
        #   infonce-mined (anchor, positive, negative) -> 2B candidates, adding our labeled
        #                                                 hard negative for every row
        # Either way the target is one-hot, so this is ordering only, no graded signal.
        return losses.MultipleNegativesRankingLoss(model=model)
    if training_style == TrainingStyle.COSENT.value:
        # Baseline: binary pair labels, so every positive pair must outscore every
        # negative pair in the batch. No graded supervision.
        return losses.CoSENTLoss(model=model)
    if training_style in (TrainingStyle.OURS_MSE.value, TrainingStyle.OURS_MSE_REVERSED.value):
        return losses.MarginMSELoss(model=model, similarity_fct=util.pairwise_cos_sim)
    if training_style == TrainingStyle.OURS_MSE_BATCHED.value:
        # ours-mse over the whole batch: every in-batch candidate is a comparison, the
        # cross-row ones targeted at the easy-negative label. See utils/graded_losses.py.
        return BatchGradedMarginMSELoss(model=model, easy_label=easy_label)
    if training_style == TrainingStyle.OURS_INFONCE.value:
        # infonce-mined with soft targets: the own hard negative holds target mass
        # 1 - label instead of 0, row-normalized. See utils/graded_losses.py.
        return GradedInfoNCELoss(model=model, easy_label=easy_label)
    if training_style == TrainingStyle.OURS_SIGLIP.value:
        # Per-pair sigmoid BCE: every in-batch cell fit to its own graded similarity
        # target, no softmax competition. See utils/graded_losses.py.
        return GradedSigLIPLoss(model=model, easy_label=easy_label)
    if training_style == TrainingStyle.SIGLIP_MINED.value:
        # ours-siglip's binary ablation: same layout, scale/bias and weighting, but
        # one-hot targets -- the mined negative is just 0, no graded signal.
        return GradedSigLIPLoss(model=model, easy_label=None, binary=True, batch_size=batch_size)
    if training_style == TrainingStyle.CLASSIC_MSE.value:
        return losses.CosineSimilarityLoss(model=model)
    raise ValueError(f"Invalid training style: {training_style}")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def parse_bool(value: str) -> bool:
    return str(value).lower() in ["true", "1", "t", "y", "yes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", choices=["text", "multimodal"], required=True,
                        help="text: string documents; multimodal: image-path documents encoded with a vision tower")
    parser.add_argument("--dataset", type=str, default=None, help="Path to processed dataset directory")
    parser.add_argument("--query-kind", choices=["original", "synthetic", "rephrased"], default=None,
                        help="query column to train on: original = the real search query, synthetic = the generated conjunctive query, rephrased = the same constraints reworded")
    parser.add_argument("--use-synthetic-data", nargs="?", const=True, type=parse_bool, default=False,
                        help="Deprecated shorthand for --query-kind synthetic")
    parser.add_argument("--training-style", type=str, default=None,
                        help=", ".join(style.value for style in TrainingStyle))
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--easy-negative-value", type=int, default=None)
    parser.add_argument("--V", type=int, default=None)
    parser.add_argument("--note", type=str, default=None,
                        help="Free-form experiment tag; becomes part of the run name and output dir")
    parser.add_argument("--distance-transform", type=str, default=None,
                        choices=[transform.value for transform in DistanceTransform])
    parser.add_argument("--distance-transform-alpha", type=float, default=None)
    parser.add_argument("--wandb-project", type=str, default="Rec2Vec")
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--truncate-dim", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--num-train-epochs", type=int, default=None)
    parser.add_argument("--global-batch-size", type=int, default=None)
    parser.add_argument("--per-device-max-batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--warmup-ratio", type=float, default=None)
    parser.add_argument("--lr-scheduler-type", type=str, default=None)
    parser.add_argument("--bf16", type=parse_bool, default=None)
    parser.add_argument("--eval-strategy", type=str, default=None)
    parser.add_argument("--save-strategy", type=str, default=None)
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--report-to", type=str, default=None)
    parser.add_argument("--log-predictions-table", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true",
                        help="Evaluate the model on the requested split without training")
    parser.add_argument("--eval-split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--train-fraction", type=float, default=1.0,
                        help="Fraction of training-split queries to keep")
    parser.add_argument("--split-seed", type=int, default=42)
    return parser.parse_args()


def load_config(args: argparse.Namespace, query_kind: str) -> Dict[str, Any]:
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["model_name"] = args.model_name or DEFAULT_MODELS[args.modality]
    config.setdefault("training_style", TrainingStyle.BASELINE_TRIPLET.value)
    config.setdefault("training_args", {})

    for key in ["dataset", "training_style", "wandb_project", "wandb_group",
                "easy_negative_value", "V", "note", "distance_transform", "distance_transform_alpha"]:
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    training_arg_keys = [
        "output_dir", "num_train_epochs", "global_batch_size", "per_device_max_batch_size",
        "learning_rate", "warmup_ratio", "lr_scheduler_type", "bf16",
        "eval_strategy", "save_strategy", "save_total_limit", "logging_steps", "report_to",
    ]
    for key in training_arg_keys:
        value = getattr(args, key)
        if value is not None:
            config["training_args"][key] = value

    if args.output_dir is None:
        # The run directory has to identify the dataset, otherwise two runs that differ
        # only by --dataset overwrite each other's checkpoints.
        config["training_args"]["output_dir"] = build_output_dir(
            config, modality=args.modality, query_kind=query_kind, extra=name_extras(config)
        )
    return config


def name_extras(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "easy": config["easy_negative_value"] if "easy_negative_value" in config else None,
        "V": config["V"] if "V" in config else None,
        "transform": config["distance_transform"] if "distance_transform" in config else None,
        "note": config["note"] if "note" in config else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    # --query-kind supersedes --use-synthetic-data; the boolean is kept for older commands.
    query_kind = args.query_kind or ("synthetic" if args.use_synthetic_data else "original")
    config = load_config(args, query_kind)
    train_config = config["training_args"]
    training_style = config["training_style"]
    modality = args.modality

    os.environ["WANDB_LOG_MODEL"] = "false"
    is_main_process = int(os.environ["LOCAL_RANK"]) <= 0 if "LOCAL_RANK" in os.environ else True
    wandb_name = build_run_name(config, modality=modality, query_kind=query_kind, extra=name_extras(config))
    wandb_run_id = None
    if is_main_process:
        run = wandb.init(
            project=config["wandb_project"],
            group=config["wandb_group"] if "wandb_group" in config else None,
            name=wandb_name,
        )
        wandb_run_id = run.id

    print(f"Loading model: {config['model_name']}")
    model = SentenceTransformer(
        config["model_name"],
        trust_remote_code=(modality == "multimodal"),
        truncate_dim=args.truncate_dim,
    )

    print(f"Loading dataset from {config['dataset']}")
    dataset = load_from_disk(config["dataset"])
    query_key = QUERY_COLUMNS[query_kind]
    if query_key not in dataset.column_names:
        raise ValueError(f"Requested query field '{query_key}' not found in columns: {dataset.column_names}")

    dataset = dataset.rename_column(query_key, "anchor")
    for column in QUERY_COLUMNS.values():
        if column in dataset.column_names:
            dataset = dataset.remove_columns([column])
    dataset = dataset.rename_column("positive_example", "positive")
    dataset = dataset.rename_column("negative_example", "negative")
    dataset = dataset.filter(lambda x: x["positive"] != x["negative"] and bool(x["anchor"]))

    # Default matches the paper's main grid; the validation sweep selected V=40 on both
    # modalities (see paper.sh's ablation rows).
    V = config["V"] if "V" in config else 40
    if V <= 0:
        raise ValueError(f"V must be > 0, got {V}")
    # Easy negatives sit beyond the measured scale; to_training_labels enforces that.
    easy_negative_value = float(
        config["easy_negative_value"] if "easy_negative_value" in config
        else default_easy_negative_distance(DEFAULT_MAX_DISTANCE)
    )

    if training_style in TRIPLET_STYLES:
        cols_to_keep = ["anchor", "positive", "negative"]
        for column in ["query_distance", "negative_example_source", "split"]:
            if column in dataset.column_names:
                cols_to_keep.append(column)
        dataset = dataset.select_columns(cols_to_keep)
    elif training_style in LABELED_STYLES:
        dataset, _ = to_training_labels(
            dataset,
            V=V,
            easy_negative_distance=easy_negative_value,
            transform=config["distance_transform"] if "distance_transform" in config else DistanceTransform.LINEAR.value,
            transform_alpha=float(config["distance_transform_alpha"]) if "distance_transform_alpha" in config else 5.0,
        )
        dataset = dataset.cast_column("label", Value("float"))
        label_min, label_max = min(dataset["label"]), max(dataset["label"])
        assert 0.0 <= label_min and label_max <= 1.0, (label_min, label_max)
    else:
        raise ValueError(f"Invalid training style: {training_style}")

    train_dataset, eval_dataset, test_dataset = split_dataset(dataset, seed=args.split_seed)
    train_dataset = select_query_fraction(train_dataset, args.train_fraction, args.split_seed, "train")
    print(f"Dataset loaded: {len(train_dataset)} train, {len(eval_dataset)} eval, {len(test_dataset)} test examples")

    raw_eval_dataset = eval_dataset
    eval_batch_size = int(train_config["per_device_max_batch_size"])

    if args.eval_only:
        selected = {"train": train_dataset, "val": eval_dataset, "test": test_dataset}[args.eval_split]
        evaluate_model(
            model,
            selected,
            modality=modality,
            split_name=args.eval_split,
            batch_size=eval_batch_size,
            log_predictions_table_name=args.log_predictions_table,
            output_dir=train_config["output_dir"],
        )
        if wandb.run is not None:
            wandb.finish()
        return

    if training_style in PAIR_STYLES:
        is_cosent = training_style == TrainingStyle.COSENT.value
        train_dataset = build_pair_dataset(train_dataset, is_cosent)
        eval_dataset = build_pair_dataset(eval_dataset, is_cosent)

    cuda_count = max(1, torch.cuda.device_count())
    per_device_train_batch_size = min(
        train_config["per_device_max_batch_size"],
        train_config["global_batch_size"] // cuda_count,
    )

    easy_label = transform_normalized_distance(
        easy_negative_value / V,
        config["distance_transform"] if "distance_transform" in config else DistanceTransform.LINEAR.value,
        float(config["distance_transform_alpha"]) if "distance_transform_alpha" in config else 5.0,
    )
    loss = build_loss(model, training_style, easy_label=easy_label,
                      batch_size=per_device_train_batch_size)
    gradient_accumulation_steps = train_config["global_batch_size"] // (per_device_train_batch_size * cuda_count)
    if per_device_train_batch_size * cuda_count * gradient_accumulation_steps != train_config["global_batch_size"]:
        raise ValueError(
            f"global_batch_size {train_config['global_batch_size']} is not divisible by "
            f"per-device batch {per_device_train_batch_size} x {cuda_count} device(s)"
        )
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be >= 1")

    # The sampler is a function of the training style, not a knob. Every in-batch-negative
    # loss (infonce family, batch-wide graded losses) mislabels a duplicate: the dataset has
    # two rows per query (hard + random negative), and a batch holding both would present the
    # twin's identical positive as a negative with target 0. NO_DUPLICATES prevents that.
    # CoSENT is the one exception: after to_pairs an anchor appears in both its positive and
    # its negative pair, and NO_DUPLICATES would admit only one of the two per batch.
    if training_style == TrainingStyle.COSENT.value:
        batch_sampler = BatchSamplers.BATCH_SAMPLER
    else:
        batch_sampler = BatchSamplers.NO_DUPLICATES

    training_args = SentenceTransformerTrainingArguments(
        output_dir=train_config["output_dir"],
        num_train_epochs=train_config["num_train_epochs"],
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=train_config["learning_rate"],
        warmup_ratio=train_config["warmup_ratio"],
        lr_scheduler_type=train_config["lr_scheduler_type"],
        bf16=train_config["bf16"],
        batch_sampler=batch_sampler,
        eval_strategy=train_config["eval_strategy"],
        save_strategy=train_config["save_strategy"],
        save_total_limit=train_config["save_total_limit"],
        logging_steps=train_config["logging_steps"],
        report_to=train_config["report_to"],
    )

    swap_pos_neg = training_style == TrainingStyle.OURS_MSE_REVERSED.value
    train_dataset_for_trainer = prepare_dataset_for_trainer(train_dataset, swap_pos_neg=swap_pos_neg)
    eval_dataset_for_trainer = prepare_dataset_for_trainer(eval_dataset, swap_pos_neg=swap_pos_neg)

    if training_style == TrainingStyle.INFONCE.value:
        train_dataset_for_trainer = drop_mined_negative(train_dataset_for_trainer)
        eval_dataset_for_trainer = drop_mined_negative(eval_dataset_for_trainer)

    if modality == "multimodal":
        if training_style in PAIR_STYLES:
            image_columns = ["sentence_B"]
        elif training_style == TrainingStyle.INFONCE.value:
            image_columns = ["positive"]
        else:
            image_columns = ["positive", "negative"]
        train_dataset_for_trainer = add_image_transform(train_dataset_for_trainer, image_columns)
        eval_dataset_for_trainer = add_image_transform(eval_dataset_for_trainer, image_columns)
        # SentenceTransformers tries to auto-build text-only model-card widgets from the
        # transformed dataset; PIL values under a string-like schema crash widget generation.
        if getattr(model, "model_card_data", None) is not None and not model.model_card_data.widget:
            model.model_card_data.widget = [{"text": "multimodal image retrieval"}]

    print(f"Trainer train dataset columns: {train_dataset_for_trainer.column_names}")
    print(f"Trainer eval dataset columns: {eval_dataset_for_trainer.column_names}")
    print(f"Final evaluator dataset columns (val): {raw_eval_dataset.column_names}")

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_for_trainer,
        eval_dataset=eval_dataset_for_trainer,
        loss=loss,
    )
    trainer.train()

    if is_main_process:
        if wandb.run is None and wandb_run_id is not None:
            wandb.init(
                project=config["wandb_project"],
                group=config["wandb_group"] if "wandb_group" in config else None,
                name=wandb_name,
                id=wandb_run_id,
                resume="allow",
            )
        # Save before evaluating: a crash in the val evaluation must not discard a
        # completed training run.
        model.save_pretrained(os.path.join(train_config["output_dir"], "final"))
        evaluate_model(
            model,
            raw_eval_dataset,
            modality=modality,
            split_name="val",
            batch_size=eval_batch_size,
            log_predictions_table_name=args.log_predictions_table,
            output_dir=train_config["output_dir"],
            wandb_step=trainer.state.global_step + 1,
        )
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
