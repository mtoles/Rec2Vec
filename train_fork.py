# TODO: include all examples from original dataset, not just those appearing in pos/neg pairs asdf asdf

from datasets import load_from_disk
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
)
# from sentence_transformers.losses import MultipleNegativesRankingLoss
from sentence_transformers import losses
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator, InformationRetrievalEvaluator

import argparse
import yaml
import pandas as pd
from collections import defaultdict
from enum import Enum
import os
import wandb
from tqdm import tqdm
import torch.distributed as dist
import torch

tqdm.pandas()

class TrainingStyle(Enum):
    BASELINE_TRIPLET = "baseline-triplet"
    OURS_MSE = "ours-mse"

def prep_ds_for_ir_eval(dataset, query_key, pos_key, neg_key, show_progress=True):
    corpus_items = list(set(list(dataset[pos_key]) + list(dataset[neg_key])))
    corpus = dict(enumerate(corpus_items))
    reverse_corpus = {v: k for k, v in corpus.items()}

    queries = {i: dataset[i][query_key] for i in range(len(dataset))}
    relevant_docs = defaultdict(list)

    iterator = tqdm(range(len(dataset)), desc="Preparing dataset for IR evaluation") if show_progress else range(len(dataset))
    for i in iterator:
        ex = dataset[i]
        relevant_corpus_id = reverse_corpus[ex[pos_key]]
        relevant_docs[i].append(relevant_corpus_id)

    return queries, corpus, relevant_docs


def evaluate_model(model, dataset):
    # (Optional) Evaluate the trained model on the test set
    triplet_evaluator = TripletEvaluator(
        anchors=dataset["anchor"],
        positives=dataset["positive"],
        negatives=dataset["negative"],
    )
    triplet_score = triplet_evaluator(model)
    print("Triplet score:", triplet_score)

    queries, corpus, relevant_docs = prep_ds_for_ir_eval(dataset, "anchor", "positive", "negative", show_progress=True)
    ks = [50, 100, 1000]
    ir_evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        mrr_at_k=ks,
        map_at_k=ks,
        precision_recall_at_k=ks,
        ndcg_at_k=ks,
        accuracy_at_k=ks,
    )
    print("Information Retrieval Evaluator:")
    results = ir_evaluator(model)
    print(results)

    # Calculate average negative cosine similarity
    print("Calculating Average Negative Cosine Similarity...")
    query_embeddings = model.encode(dataset["anchor"], convert_to_tensor=True, show_progress_bar=True)
    negative_embeddings = model.encode(dataset["negative"], convert_to_tensor=True, show_progress_bar=True)
    neg_cosine_scores = torch.nn.functional.cosine_similarity(query_embeddings, negative_embeddings)
    avg_neg_cosine = torch.mean(neg_cosine_scores).item()
    results["avg_neg_cosine_sim"] = avg_neg_cosine
    print(f"Average Negative Cosine Similarity: {avg_neg_cosine}")
    
    # Log all metrics with cosine_ prefix removed
    print("\nInformation Retrieval Metrics (without prefix):")
    metrics_to_log = {}
    for metric_name, metric_value in results.items():
        # Remove the cosine_ prefix if present
        clean_metric_name = metric_name.replace("cosine_", "")
        metrics_to_log[f"test/{clean_metric_name}"] = metric_value
    
    metrics_to_log["test/triplet_cosine_accuracy"] = triplet_score["cosine_accuracy"]

    # Log to wandb only if initialized (main process only)
    if wandb.run is not None:
        wandb.log(metrics_to_log)

    # calculate mrr

    return 


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="Path to processed dataset directory")
    parser.add_argument("--use-synthetic-data", action="store_true", help="Use synthetic data (nl_query) instead of original_query")
    parser.add_argument("--training-style", type=str, default=None, help="Training style: baseline-triplet or ours-mse")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    
    # Model configuration arguments
    parser.add_argument("--model-name", type=str, default=None, help="Model name to use for training")
    
    # Training arguments
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for model checkpoints")
    parser.add_argument("--num-train-epochs", type=int, default=None, help="Number of training epochs")
    parser.add_argument("--global-batch-size", type=int, default=None, help="Training batch size per device")
    parser.add_argument("--per-device-max-batch-size", type=int, default=None, help="Maximum batch size per device")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--warmup-ratio", type=float, default=None, help="Warmup ratio")
    parser.add_argument("--lr-scheduler-type", type=str, default=None, help="Learning rate scheduler type")
    parser.add_argument("--bf16", type=lambda x: x.lower() == 'true', default=None, help="Use bfloat16 precision (true/false)")
    parser.add_argument("--batch-sampler", type=str, default=None, help="Batch sampler type")
    parser.add_argument("--eval-strategy", type=str, default=None, help="Evaluation strategy")
    parser.add_argument("--save-strategy", type=str, default=None, help="Save strategy")
    parser.add_argument("--save-total-limit", type=int, default=None, help="Maximum number of checkpoints to keep")
    parser.add_argument("--logging-steps", type=int, default=None, help="Number of steps between logging")
    parser.add_argument("--report-to", type=str, default=None, help="Reporting destination (e.g., wandb, tensorboard)")
    
    args = parser.parse_args()
    # Override config with CLI arguments (CLI args take precedence over config.yaml)
    # Load configuration from file (defaults come from here)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize wandb only on the main process
    if int(os.environ.get("LOCAL_RANK", -1)) <= 0:
        wandb.init(project="Recalogic", name=f"train_{config['model_name']}_{config['dataset']}_{config['training_style']}_{config['training_args']['num_train_epochs']}_{config['training_args']['global_batch_size']}_{config['training_args']['learning_rate']}_{config['training_args']['warmup_ratio']}_{config['training_args']['lr_scheduler_type']}_{config['training_args']['bf16']}")
    
    # Override top-level config parameters
    if args.model_name is not None:
        config["model_name"] = args.model_name
    
    if args.dataset is not None:
        config["dataset"] = args.dataset
    
    if args.training_style is not None:
        config["training_style"] = args.training_style
    
    # Override training_args parameters
    training_arg_keys = [
        'output_dir', 'num_train_epochs', 'global_batch_size', 'per_device_max_batch_size',
        'learning_rate', 'warmup_ratio', 'lr_scheduler_type', 'bf16',
        'batch_sampler', 'eval_strategy', 'save_strategy', 'save_total_limit',
        'logging_steps', 'report_to'
    ]
    for key in training_arg_keys:
        arg_value = getattr(args, key)
        if arg_value is not None:
            config["training_args"][key] = arg_value
    
    # Use the merged config for training
    train_config = config["training_args"]
    
    # Disable wandb model upload
    os.environ["WANDB_LOG_MODEL"] = "false"

    query_key = "nl_query" if args.use_synthetic_data else "original_query"

    # 1. Load a model to finetune with
    model = SentenceTransformer(config["model_name"])

    # 2. Load the preprocessed dataset
    print(f"Loading dataset from {config['dataset']}")
    dataset = load_from_disk(config["dataset"])
    
    # Rename query column and apply training style transformations
    dataset = dataset.rename_column(query_key, "query")
    if "original_query" in dataset.column_names:
        dataset = dataset.remove_columns(["original_query"])
    if "nl_query" in dataset.column_names:
        dataset = dataset.remove_columns(["nl_query"])
    
    if config["training_style"] == TrainingStyle.BASELINE_TRIPLET.value:
        dataset = dataset.rename_column("query", "anchor")
        dataset = dataset.rename_column("positive_example", "positive")
        dataset = dataset.rename_column("negative_example", "negative")
        # Keep only easy examples (query_distance == -1) for baseline
        dataset = dataset.filter(lambda x: x["query_distance"] == -1)
        dataset = dataset.remove_columns(["query_distance"])
    elif config["training_style"] == TrainingStyle.OURS_MSE.value:
        dataset = dataset.rename_column("query_distance", "label")
    else:
        raise ValueError(f"Invalid training style: {config['training_style']}")
    
    # Split into train/eval/test (80/10/10)
    split = dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = split["train"]
    temp_test = split["test"].train_test_split(test_size=0.5, seed=42)
    eval_dataset = temp_test["train"]
    test_dataset = temp_test["test"]
    
    print(f"Dataset loaded: {len(train_dataset)} train, {len(eval_dataset)} eval, {len(test_dataset)} test examples")

    # 4. Define a loss function
    # loss = MultipleNegativesRankingLoss(model)
    if config["training_style"] == TrainingStyle.BASELINE_TRIPLET.value:
        loss = losses.TripletLoss(model=model)
    elif config["training_style"] == TrainingStyle.OURS_MSE.value:
        loss = losses.MarginMSELoss(model=model)
    else:
        raise ValueError(f"Invalid training style: {config['training_style']}")

    # 5. (Optional) Specify training arguments
    per_device_train_batch_size = min(train_config["per_device_max_batch_size"], train_config["global_batch_size"] // torch.cuda.device_count())
    gradient_accumulation_steps = train_config["global_batch_size"] // (per_device_train_batch_size * torch.cuda.device_count())
    assert train_config["global_batch_size"] // (per_device_train_batch_size * torch.cuda.device_count()) == train_config["global_batch_size"] / (per_device_train_batch_size * torch.cuda.device_count()), f"Global batch size {train_config['global_batch_size']} is not divisible by the product of per-device batch size {per_device_train_batch_size} and the number of devices {torch.cuda.device_count()}"
    assert gradient_accumulation_steps > 0, f"Gradient accumulation steps {gradient_accumulation_steps} is less than 1"

    training_args = SentenceTransformerTrainingArguments(
        # Required parameter:
        output_dir=train_config["output_dir"],
        # Optional training parameters:
        num_train_epochs=train_config["num_train_epochs"],
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=train_config["learning_rate"],
        warmup_ratio=train_config["warmup_ratio"],
        lr_scheduler_type=train_config["lr_scheduler_type"],
        # fp16=True,  # Set to False if you get an error that your GPU can't run on FP16
        bf16=train_config["bf16"],  # Set to True if you have a GPU that supports BF16
        batch_sampler=BatchSamplers[train_config["batch_sampler"]],  # MultipleNegativesRankingLoss benefits from no duplicate samples in a batch
        # Optional tracking/debugging parameters:
        eval_strategy=train_config["eval_strategy"],
        save_strategy=train_config["save_strategy"],
        save_total_limit=train_config["save_total_limit"],
        logging_steps=train_config["logging_steps"],
        report_to=train_config.get("report_to", "none"),
        # run_name="mpnet-base-all-nli-triplet",  # Will be used in W&B if `wandb` is installed
    )

    # 6. Create a dev evaluator & evaluate the base model


    # 7. Create a trainer & train
    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,#.remove_columns(["query_distance"]),
        eval_dataset=eval_dataset,#.remove_columns(["query_distance"]),
        loss=loss,
        # evaluator=dev_evaluator,
    )

    trainer.train()

    # 8. Evaluate the trained model
    if int(os.environ.get("LOCAL_RANK", -1)) <= 0:
        evaluate_model(model, test_dataset)

        # 9. Save the trained model
        model.save_pretrained("models/sbert-contrastive/final")


if __name__ == "__main__":
    main()