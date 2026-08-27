from collections import defaultdict
from enum import Enum
import argparse
import json
import logging
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
from utils.run_naming import build_output_dir, build_run_name


logger = logging.getLogger(__name__)


# Easy negatives carry this instead of a measured feature distance (see preprocess_*.py).
EASY_NEGATIVE_SENTINEL = -1


class TrainingStyle(Enum):
    BASELINE_TRIPLET = "baseline-triplet"
    INFONCE = "infonce"
    COSENT = "cosent"
    OURS_MSE = "ours-mse"
    OURS_MSE_REVERSED = "ours-mse-reversed"
    CLASSIC_MSE = "classic-mse"


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
        if encode_fn_name != "document":
            return super().embed_inputs(
                model,
                sentences,
                encode_fn_name=encode_fn_name,
                prompt_name=prompt_name,
                prompt=prompt,
                **kwargs,
            )

        images = [load_rgb_image(path) if isinstance(path, str) else path for path in sentences]
        try:
            return super().embed_inputs(
                model,
                images,
                encode_fn_name=encode_fn_name,
                prompt_name=prompt_name,
                prompt=prompt,
                **kwargs,
            )
        finally:
            for image in images:
                if hasattr(image, "close"):
                    image.close()


def prep_ds_for_ir_eval(dataset: Dataset, query_key: str, pos_key: str, neg_key: str, show_progress: bool = True):
    corpus_items = sorted(set(list(dataset[pos_key]) + list(dataset[neg_key])))
    corpus = {str(i): path for i, path in enumerate(corpus_items)}
    reverse_corpus = {path: cid for cid, path in corpus.items()}

    queries = {}
    relevant_docs = defaultdict(set)
    query_to_id = {}

    iterator = tqdm(range(len(dataset)), desc="Preparing image IR evaluation") if show_progress else range(len(dataset))
    for i in iterator:
        ex = dataset[i]
        query = ex[query_key]
        if query not in query_to_id:
            query_id = str(len(query_to_id))
            query_to_id[query] = query_id
            queries[query_id] = query

        query_id = query_to_id[query]
        relevant_docs[query_id].add(reverse_corpus[ex[pos_key]])

    return queries, corpus, {key: set(value) for key, value in relevant_docs.items()}


def encode_images(model: SentenceTransformer, image_paths: List[str], batch_size: int, show_progress_bar: bool):
    embeddings = []
    iterator = range(0, len(image_paths), batch_size)
    if show_progress_bar:
        iterator = tqdm(iterator, desc="Encoding images")

    for start in iterator:
        paths = image_paths[start : start + batch_size]
        images = [load_rgb_image(path) for path in paths]
        try:
            batch_embeddings = model.encode(
                images,
                batch_size=batch_size,
                convert_to_tensor=True,
                show_progress_bar=False,
            )
            embeddings.append(batch_embeddings)
        finally:
            for image in images:
                image.close()

    return torch.cat(embeddings, dim=0) if embeddings else torch.empty(0)


def evaluate_model(
    model: SentenceTransformer,
    dataset: Dataset,
    split_name: str = "val",
    batch_size: int = 32,
    log_predictions_table_name: str | None = None,
    output_dir: str | None = None,
    wandb_step: int | None = None,
):
    print(f"Evaluating multimodal retrieval on {split_name}...")
    results = {}

    query_embeddings = model.encode(
        dataset["anchor"],
        batch_size=batch_size,
        convert_to_tensor=True,
        show_progress_bar=True,
    )
    positive_embeddings = encode_images(model, dataset["positive"], batch_size=batch_size, show_progress_bar=True)
    negative_embeddings = encode_images(model, dataset["negative"], batch_size=batch_size, show_progress_bar=True)

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
        sources = dataset["negative_example_source"]
        easy_mask = torch.tensor([source == "random" for source in sources], device=neg_cosine_scores.device)
        hard_mask = ~easy_mask

        if torch.any(easy_mask):
            results["avg_easy_neg_cosine_sim"] = torch.mean(neg_cosine_scores[easy_mask]).item()
            results["triplet_cosine_accuracy_easy"] = torch.mean(is_triplet_correct[easy_mask]).item()
            print(f"Easy Triplet Cosine Accuracy: {results['triplet_cosine_accuracy_easy']}")

        if torch.any(hard_mask):
            results["avg_hard_neg_cosine_sim"] = torch.mean(neg_cosine_scores[hard_mask]).item()
            results["triplet_cosine_accuracy_hard"] = torch.mean(is_triplet_correct[hard_mask]).item()
            print(f"Hard Triplet Cosine Accuracy: {results['triplet_cosine_accuracy_hard']}")

    queries, corpus, relevant_docs = prep_ds_for_ir_eval(dataset, "anchor", "positive", "negative", show_progress=True)
    ks = [1, 5, 10, 50, 100, 1000]
    max_corpus_k = max(1, len(corpus))
    ks = [min(k, max_corpus_k) for k in ks]
    ks = sorted(set(ks))

    ir_evaluator = ImageCorpusInformationRetrievalEvaluator(
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
    results.update({key.replace("cosine_", ""): value for key, value in ir_results.items()})

    prefixed_results = {f"{split_name}/{key}": value for key, value in results.items()}

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, f"eval_{split_name}_metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(prefixed_results, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"Saved {split_name} metrics to {metrics_path}")

    if wandb.run is not None:
        # The final retrieval evaluator runs outside the HF trainer. Update both
        # history and summary so the metrics remain visible even if there are no
        # later trainer logs to roll them into W&B's summary.
        wandb.log(prefixed_results, step=wandb_step, commit=True)
        wandb.run.summary.update(prefixed_results)
        if output_dir is not None:
            wandb.save(metrics_path, policy="now")

    return results


def prepare_dataset_for_trainer(dataset: Dataset, swap_pos_neg: bool = False) -> Dataset:
    columns_to_remove = [
        "query_distance",
        "negative_example_source",
        "distance_source",
        "positive_product_id",
        "negative_product_id",
        "item",
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
    removable_columns = [column for column in columns_to_remove if column in dataset.column_names]
    if removable_columns:
        dataset = dataset.remove_columns(removable_columns)
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


def parse_bool(value: str) -> bool:
    return str(value).lower() in ["true", "1", "t", "y", "yes"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="Path to processed image dataset directory")
    parser.add_argument("--use-synthetic-data", nargs="?", const=True, type=parse_bool, default=True)
    parser.add_argument("--use-rephrased-query", nargs="?", const=True, type=parse_bool, default=False)
    parser.add_argument("--training-style", type=str, default=None, help="baseline-triplet, ours-mse, ours-mse-reversed, or classic-mse")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--easy-negative-value", type=int, default=None)
    parser.add_argument("--V", type=int, default=None)
    parser.add_argument("--note", type=str, default=None, help="Free-form experiment tag; becomes part of the run name and output dir")
    parser.add_argument(
        "--distance-transform",
        type=str,
        default=None,
        choices=[transform.value for transform in DistanceTransform],
    )
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
    parser.add_argument("--batch-sampler", type=str, default=None)
    parser.add_argument("--eval-strategy", type=str, default=None)
    parser.add_argument("--save-strategy", type=str, default=None)
    parser.add_argument("--save-total-limit", type=int, default=None)
    parser.add_argument("--logging-steps", type=int, default=None)
    parser.add_argument("--report-to", type=str, default=None)
    parser.add_argument("--log-predictions-table", type=str, default=None)
    parser.add_argument("--eval-only", action="store_true", help="Evaluate the model on the image dataset without training")
    parser.add_argument("--eval-split", choices=["train", "val", "test"], default="test")
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Fraction of the training split to use for image fine-tuning",
    )
    parser.add_argument("--split-seed", type=int, default=42)
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> Dict[str, Any]:
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["model_name"] = "sentence-transformers/clip-ViT-B-32"
    config.setdefault("training_style", TrainingStyle.BASELINE_TRIPLET.value)
    config.setdefault("training_args", {})

    if args.model_name is not None:
        config["model_name"] = args.model_name
    if args.dataset is not None:
        config["dataset"] = args.dataset
    if args.training_style is not None:
        config["training_style"] = args.training_style
    if args.wandb_project is not None:
        config["wandb_project"] = args.wandb_project
    if args.wandb_group is not None:
        config["wandb_group"] = args.wandb_group
    if args.easy_negative_value is not None:
        config["easy_negative_value"] = args.easy_negative_value
    if args.V is not None:
        config["V"] = args.V
    if args.note is not None:
        config["note"] = args.note
    if args.distance_transform is not None:
        config["distance_transform"] = args.distance_transform
    if args.distance_transform_alpha is not None:
        config["distance_transform_alpha"] = args.distance_transform_alpha

    training_arg_keys = [
        "output_dir",
        "num_train_epochs",
        "global_batch_size",
        "per_device_max_batch_size",
        "learning_rate",
        "warmup_ratio",
        "lr_scheduler_type",
        "bf16",
        "batch_sampler",
        "eval_strategy",
        "save_strategy",
        "save_total_limit",
        "logging_steps",
        "report_to",
    ]
    for key in training_arg_keys:
        value = getattr(args, key)
        if value is not None:
            config["training_args"][key] = value

    if args.output_dir is None:
        # Include the dataset in the path, otherwise runs that differ only by --dataset
        # collide in models/.
        query_label = {"original_query": "original", "nl_query": "synthetic", "rephrased_query": "rephrased"}.get(
            select_query_key(args), "original"
        )
        config["training_args"]["output_dir"] = build_output_dir(
            config,
            modality="multimodal",
            query_kind=query_label,
            extra={
                "easy": config.get("easy_negative_value"),
                "V": config.get("V"),
                "transform": config.get("distance_transform"),
                "note": config.get("note"),
            },
        )

    return config


def select_query_key(args: argparse.Namespace) -> str:
    if args.use_rephrased_query:
        return "rephrased_query"
    return "nl_query" if args.use_synthetic_data else "original_query"


def split_dataset_by_query(dataset: Dataset, seed: int = 42):
    unique_queries = list(dict.fromkeys(dataset["anchor"]))
    if len(unique_queries) < 3:
        raise ValueError("Need at least 3 unique queries for train/eval/test split")

    train_queries, temp_queries = train_test_split(unique_queries, test_size=0.2, random_state=seed)
    eval_queries, test_queries = train_test_split(temp_queries, test_size=0.5, random_state=seed)

    train_queries = set(train_queries)
    eval_queries = set(eval_queries)
    test_queries = set(test_queries)

    train_dataset = dataset.filter(lambda x: x["anchor"] in train_queries)
    eval_dataset = dataset.filter(lambda x: x["anchor"] in eval_queries)
    test_dataset = dataset.filter(lambda x: x["anchor"] in test_queries)
    return train_dataset, eval_dataset, test_dataset


def select_query_fraction(dataset: Dataset, fraction: float, seed: int, split_name: str) -> Dataset:
    if not 0 < fraction <= 1:
        raise ValueError(f"train_fraction must be in (0, 1], got {fraction}")
    if fraction == 1:
        return dataset

    unique_queries = list(dict.fromkeys(dataset["anchor"]))
    selected_count = max(1, int(round(len(unique_queries) * fraction)))
    selected_queries, _ = train_test_split(
        unique_queries,
        train_size=selected_count,
        random_state=seed,
    )
    selected_queries = set(selected_queries)
    selected_dataset = dataset.filter(lambda x: x["anchor"] in selected_queries)
    print(
        f"Using {len(selected_dataset)}/{len(dataset)} {split_name} examples "
        f"from {selected_count}/{len(unique_queries)} queries (fraction={fraction})"
    )
    return selected_dataset


def main():
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    config = load_config(args)
    query_key = select_query_key(args)
    train_config = config["training_args"]

    os.environ["WANDB_LOG_MODEL"] = "false"

    is_main_process = int(os.environ.get("LOCAL_RANK", -1)) <= 0
    wandb_project = config.get("wandb_project", "Rec2Vec")
    wandb_group = config.get("wandb_group", None)
    query_label = {"original_query": "original", "nl_query": "synthetic", "rephrased_query": "rephrased"}.get(
        query_key, query_key
    )
    name_extras = {
        "easy": config.get("easy_negative_value"),
        "V": config.get("V"),
        "transform": config.get("distance_transform"),
        "note": config.get("note"),
    }
    wandb_name = build_run_name(config, modality="multimodal", query_kind=query_label, extra=name_extras)
    wandb_run_id = None

    if is_main_process:
        run = wandb.init(
            project=wandb_project,
            group=wandb_group,
            name=wandb_name,
        )
        wandb_run_id = run.id

    print(f"Loading multimodal model: {config['model_name']}")
    model = SentenceTransformer(
        config["model_name"],
        trust_remote_code=True,
        truncate_dim=args.truncate_dim,
    )

    print(f"Loading dataset from {config['dataset']}")
    dataset = load_from_disk(config["dataset"])
    if query_key not in dataset.column_names:
        raise ValueError(f"Requested query field '{query_key}' not found in columns: {dataset.column_names}")

    dataset = dataset.rename_column(query_key, "anchor")
    for column in ["original_query", "nl_query", "rephrased_query"]:
        if column in dataset.column_names:
            dataset = dataset.remove_columns([column])

    dataset = dataset.rename_column("positive_example", "positive")
    dataset = dataset.rename_column("negative_example", "negative")
    dataset = dataset.filter(lambda x: x["positive"] != x["negative"] and bool(x["anchor"]))

    V = config.get("V", 20)
    easy_negative_value = int(config.get("easy_negative_value", 10))
    distance_transform = config.get("distance_transform", DistanceTransform.LINEAR.value)
    distance_transform_alpha = float(config.get("distance_transform_alpha", 5.0))
    if V <= 0:
        raise ValueError(f"V must be > 0, got {V}")

    if config["training_style"] in (TrainingStyle.BASELINE_TRIPLET.value, TrainingStyle.INFONCE.value):
        cols_to_keep = ["anchor", "positive", "negative"]
        for column in ["query_distance", "negative_example_source"]:
            if column in dataset.column_names:
                cols_to_keep.append(column)
        dataset = dataset.select_columns(cols_to_keep)
    elif config["training_style"] in [
        TrainingStyle.OURS_MSE.value,
        TrainingStyle.OURS_MSE_REVERSED.value,
        TrainingStyle.CLASSIC_MSE.value,
        TrainingStyle.COSENT.value,
    ]:
        dataset = dataset.rename_column("query_distance", "label")
        # -1 is the easy-negative sentinel, not a real distance; map it to the distance
        # we actually want easy negatives trained toward.
        dataset = dataset.map(
            lambda x: {"label": easy_negative_value if x["label"] == EASY_NEGATIVE_SENTINEL else x["label"]}
        )
        dataset = dataset.map(
            lambda x: {
                "label": transform_normalized_distance(
                    x["label"] / V,
                    distance_transform,
                    distance_transform_alpha,
                )
            }
        )
        dataset = dataset.cast_column("label", Value("float"))
    else:
        raise ValueError(f"Invalid training style: {config['training_style']}")

    train_dataset, eval_dataset, test_dataset = split_dataset_by_query(dataset, seed=args.split_seed)
    train_dataset = select_query_fraction(train_dataset, args.train_fraction, args.split_seed, "train")
    print(f"Dataset loaded: {len(train_dataset)} train, {len(eval_dataset)} eval, {len(test_dataset)} test examples")

    raw_eval_dataset = eval_dataset
    eval_split_datasets = {
        "train": train_dataset,
        "val": eval_dataset,
        "test": test_dataset,
    }

    if args.eval_only:
        selected_eval_dataset = eval_split_datasets[args.eval_split]
        evaluate_model(
            model,
            selected_eval_dataset,
            split_name=args.eval_split,
            batch_size=int(train_config.get("per_device_max_batch_size", 32)),
            log_predictions_table_name=args.log_predictions_table,
            output_dir=train_config["output_dir"],
        )
        if wandb.run is not None:
            wandb.finish()
        return

    if config["training_style"] in (TrainingStyle.CLASSIC_MSE.value, TrainingStyle.COSENT.value):
        def to_pairs(batch):
            anchors = []
            others = []
            labels = []
            for anchor, positive, negative, label in zip(
                batch["anchor"], batch["positive"], batch["negative"], batch["label"]
            ):
                anchors.append(anchor)
                others.append(positive)
                labels.append(1.0)
                anchors.append(anchor)
                others.append(negative)
                labels.append(1.0 - label)
            return {"sentence_A": anchors, "sentence_B": others, "label": labels}

        train_dataset = train_dataset.map(to_pairs, batched=True, remove_columns=train_dataset.column_names)
        eval_dataset = eval_dataset.map(to_pairs, batched=True, remove_columns=eval_dataset.column_names)

    if config["training_style"] == TrainingStyle.BASELINE_TRIPLET.value:
        loss = losses.TripletLoss(
            model=model,
            distance_metric=losses.TripletDistanceMetric.COSINE,
            triplet_margin=0.2,
        )
    elif config["training_style"] == TrainingStyle.INFONCE.value:
        # Standard contrastive objective: cross-entropy over in-batch negatives.
        loss = losses.MultipleNegativesRankingLoss(model=model)
    elif config["training_style"] == TrainingStyle.COSENT.value:
        # Same graded label as ours-mse, used ordinally instead of regressed.
        loss = losses.CoSENTLoss(model=model)
    elif config["training_style"] in [TrainingStyle.OURS_MSE.value, TrainingStyle.OURS_MSE_REVERSED.value]:
        loss = losses.MarginMSELoss(model=model, similarity_fct=util.pairwise_cos_sim)
    elif config["training_style"] == TrainingStyle.CLASSIC_MSE.value:
        loss = losses.CosineSimilarityLoss(model=model)
    else:
        raise ValueError(f"Invalid training style: {config['training_style']}")

    cuda_count = max(1, torch.cuda.device_count())
    per_device_train_batch_size = min(
        train_config["per_device_max_batch_size"],
        train_config["global_batch_size"] // cuda_count,
    )
    gradient_accumulation_steps = train_config["global_batch_size"] // (per_device_train_batch_size * cuda_count)
    if gradient_accumulation_steps <= 0:
        raise ValueError("gradient_accumulation_steps must be >= 1")

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
        batch_sampler=BatchSamplers[train_config["batch_sampler"]],
        eval_strategy=train_config["eval_strategy"],
        save_strategy=train_config["save_strategy"],
        save_total_limit=train_config["save_total_limit"],
        logging_steps=train_config["logging_steps"],
        report_to=train_config.get("report_to", "none"),
    )

    swap_pos_neg = config["training_style"] == TrainingStyle.OURS_MSE_REVERSED.value
    train_dataset_for_trainer = prepare_dataset_for_trainer(train_dataset, swap_pos_neg=swap_pos_neg)
    eval_dataset_for_trainer = prepare_dataset_for_trainer(eval_dataset, swap_pos_neg=swap_pos_neg)

    if config["training_style"] in (TrainingStyle.CLASSIC_MSE.value, TrainingStyle.COSENT.value):
        train_dataset_for_trainer = add_image_transform(train_dataset_for_trainer, ["sentence_B"])
        eval_dataset_for_trainer = add_image_transform(eval_dataset_for_trainer, ["sentence_B"])
    else:
        train_dataset_for_trainer = add_image_transform(train_dataset_for_trainer, ["positive", "negative"])
        eval_dataset_for_trainer = add_image_transform(eval_dataset_for_trainer, ["positive", "negative"])

    print(f"Trainer train dataset columns: {train_dataset_for_trainer.column_names}")
    print(f"Trainer eval dataset columns: {eval_dataset_for_trainer.column_names}")
    print(f"Final evaluator dataset columns (val): {raw_eval_dataset.column_names}")

    # SentenceTransformers tries to auto-build text-only model-card widgets from
    # the transformed dataset. For image columns, the transform returns PIL images
    # while the underlying feature schema still looks string-like, which can crash
    # widget generation before training starts.
    if getattr(model, "model_card_data", None) is not None and not model.model_card_data.widget:
        model.model_card_data.widget = [{"text": "multimodal image retrieval"}]

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
                project=wandb_project,
                group=wandb_group,
                name=wandb_name,
                id=wandb_run_id,
                resume="allow",
            )
        model.save_pretrained(os.path.join(train_config["output_dir"], "final"))
        evaluate_model(
            model,
            raw_eval_dataset,
            split_name="val",
            batch_size=per_device_train_batch_size,
            log_predictions_table_name=args.log_predictions_table,
            output_dir=train_config["output_dir"],
            wandb_step=trainer.state.global_step + 1,
        )
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
