# TODO: include all examples from original dataset, not just those appearing in pos/neg pairs asdf asdf

from datasets import load_from_disk, Dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
    util,
    losses
)
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers.evaluation import TripletEvaluator, InformationRetrievalEvaluator
from datasets import Value
import argparse
import logging
import yaml
import pandas as pd
from collections import defaultdict
from enum import Enum
import os
import wandb
from tqdm import tqdm
import torch.distributed as dist
import torch
from utils.distance_transform import DistanceTransform, transform_normalized_distance
from utils.run_naming import build_output_dir, build_run_name

tqdm.pandas()

# Easy negatives carry this instead of a measured feature distance (see preprocess_*.py).
EASY_NEGATIVE_SENTINEL = -1


class TrainingStyle(Enum):
    BASELINE_TRIPLET = "baseline-triplet"
    INFONCE = "infonce"
    COSENT = "cosent"
    OURS_MSE = "ours-mse"
    OURS_MSE_REVERSED = "ours-mse-reversed"
    CLASSIC_MSE = "classic-mse"

def prep_ds_for_ir_eval(dataset, query_key, pos_key, neg_key, show_progress=True):
    corpus_items = list(set(list(dataset[pos_key]) + list(dataset[neg_key])))
    corpus = dict(enumerate(corpus_items))
    reverse_corpus = {v: k for k, v in corpus.items()}

    queries = {}
    relevant_docs = defaultdict(set)
    
    # Map query string to a unique query ID
    query_to_id = {}

    iterator = tqdm(range(len(dataset)), desc="Preparing dataset for IR evaluation") if show_progress else range(len(dataset))
    for i in iterator:
        ex = dataset[i]
        q_str = ex[query_key]
        
        if q_str not in query_to_id:
            q_id = str(len(query_to_id))
            query_to_id[q_str] = q_id
            queries[q_id] = q_str
            
        q_id = query_to_id[q_str]
        relevant_corpus_id = reverse_corpus[ex[pos_key]]
        relevant_docs[q_id].add(relevant_corpus_id)

    # Convert sets back to lists for the evaluator
    relevant_docs = {k: list(v) for k, v in relevant_docs.items()}

    return queries, corpus, relevant_docs


def evaluate_model(model, dataset, split_name="test", log_predictions_table_name=None):
    # (Optional) Evaluate the trained model on the requested split
    triplet_evaluator = TripletEvaluator(
        anchors=dataset["anchor"],
        positives=dataset["positive"],
        negatives=dataset["negative"],
    )
    triplet_score = triplet_evaluator(model)
    print("Triplet score:", triplet_score)

    queries, corpus, relevant_docs = prep_ds_for_ir_eval(dataset, "anchor", "positive", "negative", show_progress=True)
    ks = [1, 5, 10, 50, 100, 1000]
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
    
    # Combined average (original)
    avg_neg_cosine = torch.mean(neg_cosine_scores).item()
    results["avg_neg_cosine_sim"] = avg_neg_cosine
    print(f"Average Negative Cosine Similarity (Combined): {avg_neg_cosine}")

    # Calculate average positive cosine similarity
    print("Calculating Average Positive Cosine Similarity...")
    pos_cosine_scores = torch.nn.functional.cosine_similarity(query_embeddings, model.encode(dataset["positive"], convert_to_tensor=True, show_progress_bar=True))
    avg_pos_cosine = torch.mean(pos_cosine_scores).item()
    results["avg_pos_cosine_sim"] = avg_pos_cosine
    print(f"Average Positive Cosine Similarity: {avg_pos_cosine}")

    # Calculate triplet accuracy (cosine)
    is_triplet_correct = (pos_cosine_scores > neg_cosine_scores).float()
    triplet_acc = torch.mean(is_triplet_correct).item()
    results["manual_triplet_cosine_accuracy"] = triplet_acc
    print(f"Manual Triplet Cosine Accuracy: {triplet_acc}")

    # Split by Hard/Easy negatives
    easy_mask = None
    hard_mask = None
    
    if "negative_example_source" in dataset.column_names:
        sources = dataset["negative_example_source"]
        # Easy negatives typically have source "random", while hard negatives are from ESCI labels
        easy_mask = torch.tensor([s == "random" for s in sources], device=neg_cosine_scores.device)
        hard_mask = ~easy_mask
    else:
        # Fallback to label-based logic if source column is missing
        label_key = "label" if "label" in dataset.column_names else "query_distance"
        if label_key in dataset.column_names:
            labels_list = dataset[label_key]
            labels_tensor = torch.tensor(labels_list, device=neg_cosine_scores.device, dtype=torch.float)
            if label_key == "query_distance":
                # Raw distances: easy negatives still carry the sentinel.
                easy_mask = torch.isclose(labels_tensor, torch.tensor(float(EASY_NEGATIVE_SENTINEL), device=labels_tensor.device))
            else:
                # Already remapped/scaled: easy negatives sit at the largest target distance.
                max_label_tensor = torch.max(labels_tensor)
                easy_mask = labels_tensor >= max_label_tensor - 1e-5
            hard_mask = ~easy_mask

    if easy_mask is not None and torch.any(easy_mask):
        avg_easy_neg_cosine = torch.mean(neg_cosine_scores[easy_mask]).item()
        results["avg_easy_neg_cosine_sim"] = avg_easy_neg_cosine
        print(f"Average Easy Negative Cosine Similarity: {avg_easy_neg_cosine}")

        easy_triplet_acc = torch.mean(is_triplet_correct[easy_mask]).item()
        results["triplet_cosine_accuracy_easy"] = easy_triplet_acc
        print(f"Easy Triplet Cosine Accuracy: {easy_triplet_acc}")
    
    if hard_mask is not None and torch.any(hard_mask):
        avg_hard_neg_cosine = torch.mean(neg_cosine_scores[hard_mask]).item()
        results["avg_hard_neg_cosine_sim"] = avg_hard_neg_cosine
        print(f"Average Hard Negative Cosine Similarity: {avg_hard_neg_cosine}")

        hard_triplet_acc = torch.mean(is_triplet_correct[hard_mask]).item()
        results["triplet_cosine_accuracy_hard"] = hard_triplet_acc
        print(f"Hard Triplet Cosine Accuracy: {hard_triplet_acc}")
    
    # Log all metrics with cosine_ prefix removed
    print("\nInformation Retrieval Metrics (without prefix):")
    metrics_to_log = {}
    for metric_name, metric_value in results.items():
        # Remove the cosine_ prefix if present
        clean_metric_name = metric_name.replace("cosine_", "")
        metrics_to_log[f"{split_name}/{clean_metric_name}"] = metric_value
    
    metrics_to_log[f"{split_name}/triplet_cosine_accuracy"] = triplet_score["cosine_accuracy"]

    # Log to wandb only if initialized (main process only)
    if wandb.run is not None:
        wandb.log(metrics_to_log)
        
        if log_predictions_table_name:
            # Suppress wandb warning about serializing large strings
            logging.getLogger("wandb.sdk").addFilter(lambda record: "Serializing object" not in record.getMessage())
            
            print(f"Generating top-50 predictions for table: {log_predictions_table_name}")
            
            # Create a map for metadata (query_text, prod_text) -> (type, label)
            # We need to iterate over the dataset to find positive/negative definitions
            # dataset has columns: anchor, positive, negative, and potentially 'label'
            metadata_map = {}
            print("Building metadata map...")
            has_label = "label" in dataset.column_names
            label_key = "label" if has_label else "query_distance"
            
            # Easy negatives are the sentinel in the raw column, and the largest target
            # distance once labels have been remapped/scaled.
            easy_label = EASY_NEGATIVE_SENTINEL if label_key == "query_distance" else max(dataset[label_key])

            for row in dataset:
                q = row["anchor"]
                p = row["positive"]
                n = row["negative"]
                label = row[label_key]
                
                # Logic for labeling:
                # A label at `easy_label` means "Easy Negative", anything else is a
                # "Hard Negative". If not in the map, it will be "Random" (handled in
                # format_prediction).
                # We can't easily map back from query ID to all negatives without iterating,
                # so we map (q, n) -> ("Hard Negative" or "Easy Negative", label)
                if str(label) == str(easy_label):
                    neg_type = "Easy Negative"
                else:
                    neg_type = "Hard Negative"

                metadata_map[(q, n)] = (neg_type, label)
                
                # Positives are always Positive
                metadata_map[(q, p)] = ("Positive", "N/A")
            
            print(f"Metadata map size: {len(metadata_map)}")
            if len(metadata_map) > 0:
                dk = list(metadata_map.keys())[0]
                print(f"Sample map key: {dk}, type of query: {type(dk[0])}, type of text: {type(dk[1])}")
            
            # Use the corpus and queries already prepared
            corpus_ids = list(corpus.keys())
            corpus_texts = [corpus[i] for i in corpus_ids]
            
            query_ids = list(queries.keys())
            query_texts = [queries[i] for i in query_ids]

            print(f"Encoding {len(corpus_texts)} corpus texts and {len(query_texts)} queries...")
            corpus_embeddings = model.encode(corpus_texts, convert_to_tensor=True, show_progress_bar=True)
            query_embeddings = model.encode(query_texts, convert_to_tensor=True, show_progress_bar=True)
            
            print("Computing cosine similarities...")
            cos_scores = util.cos_sim(query_embeddings, corpus_embeddings)
            
            # Get top 50
            k = min(50, len(corpus_texts))
            top_results = torch.topk(cos_scores, k=k)
            
            print("Creating WandB table...")
            table_data = []
            
            for i, q_id in enumerate(query_ids):
                # q_id matches the i-th row in query_embeddings
                
                # Retrieve ground truth (corpus IDs relevant to this query)
                gt_corpus_ids = relevant_docs.get(q_id, [])
                gt_texts = [corpus[cid] for cid in gt_corpus_ids if cid in corpus]
                
                # Retrieve predictions
                pred_indices = top_results.indices[i].tolist()
                pred_texts = [corpus_texts[idx] for idx in pred_indices]
                
                # Format texts for logging (no truncation as requested)
                # And include metadata
                def format_prediction(text, score, query_text):
                    meta_type = "Random" # Default if not in map
                    meta_dist = "N/A"
                    
                    if (query_text, text) in metadata_map:
                        meta_type, meta_dist = metadata_map[(query_text, text)]
                    #else:
                        # Debug logic for first few randoms
                        # if random.random() < 0.001: 
                        #     print(f"DEBUG: Not found in map: query='{query_text[:20]}...', text='{text[:20]}...'")
                    
                    # No truncation
                    return f"{text}\n[Score: {score:.4f}] [Type: {meta_type}] [Dist: {meta_dist}]"

                formatted_preds = []
                for idx, score in zip(pred_indices, top_results.values[i].tolist()):
                     pred_text = corpus_texts[idx]
                     formatted_preds.append(format_prediction(pred_text, score, query_texts[i]))

                # For ground truth, we don't have scores from top_results unless they are retrieved.
                # Just show text and type
                def format_gt(text, query_text):
                     meta_type = "Positive"
                     # Check if it is in map (it should be!)
                     if (query_text, text) not in metadata_map:
                         print(f"WARNING: GT not found in map for query: {query_text[:50]}...")
                         print(f"GT text: {text[:50]}...")
                         
                     # No truncation
                     return f"{text}\n[Type: {meta_type}]"

                formatted_gt = [format_gt(t, query_texts[i]) for t in gt_texts]

                table_data.append([
                    query_texts[i],
                    "\n\n".join(formatted_gt),
                    "\n\n".join(formatted_preds)
                ])
            
            columns = ["Query", "Ground Truth", "Top 50 Predictions"]
            try:
                wandb_table = wandb.Table(data=table_data, columns=columns)
                wandb.log({log_predictions_table_name: wandb_table})
                print(f"Logged table '{log_predictions_table_name}' to WandB.")
            except Exception as e:
                print(f"Failed to log table to WandB: {e}")

    # calculate mrr

    return 


def prepare_dataset_for_trainer(dataset, swap_pos_neg=False):
    columns_to_remove = [
        "query_distance",
        "negative_example_source",
        "rephrased_query",
    ]
    removable_columns = [col for col in columns_to_remove if col in dataset.column_names]
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
        reordered += [col for col in dataset.column_names if col not in reordered]
        dataset = dataset.select_columns(reordered)
    return dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default=None, help="Path to processed dataset directory")
    parser.add_argument("--use-synthetic-data", nargs='?', const=True, type=lambda x: str(x).lower() in ['true', '1', 't', 'y', 'yes'], default=False, help="Use synthetic data (nl_query) instead of original_query")
    parser.add_argument("--training-style", type=str, default=None, help="Training style: baseline-triplet, ours-mse, ours-mse-reversed, or classic-mse")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument("--easy-negative-value", type=int, default=None, help="Value for easy negatives")
    parser.add_argument("--V", type=int, default=None, help="Value for V")
    parser.add_argument("--note", type=str, default=None, help="Free-form experiment tag; becomes part of the run name and output dir")
    parser.add_argument(
        "--distance-transform",
        type=str,
        default=None,
        choices=[transform.value for transform in DistanceTransform],
        help="Shaping applied to the normalized distance label",
    )
    parser.add_argument("--distance-transform-alpha", type=float, default=None, help="Alpha for log/exp transforms")
    parser.add_argument("--wandb-project", type=str, default="Rec2Vec", help="Wandb project name")
    parser.add_argument("--wandb-group", type=str, default=None, help="Wandb group name (acts like a folder in UI)")
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
    parser.add_argument("--log-predictions-table", type=str, default=None, help="Name of the wandb table to log top-50 predictions")
    
    args = parser.parse_args()
    # Override config with CLI arguments (CLI args take precedence over config.yaml)
    # Load configuration from file (defaults come from here)
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override top-level config parameters
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

    query_kind = "synthetic" if args.use_synthetic_data else "original"
    name_extras = {
        "multiplier": config.get("hard_negative_multiplier"),
        "easy": config.get("easy_negative_value"),
        "V": config.get("V"),
        "transform": config.get("distance_transform"),
        "note": config.get("note"),
    }
    # The run directory has to identify the dataset, otherwise two runs that differ only
    # by --dataset overwrite each other's checkpoints.
    if args.output_dir is None:
        config["training_args"]["output_dir"] = build_output_dir(
            config, modality="text", query_kind=query_kind, extra=name_extras
        )

    # Initialize wandb only on the main process
    if int(os.environ.get("LOCAL_RANK", -1)) <= 0:
        wandb_name = build_run_name(config, modality="text", query_kind=query_kind, extra=name_extras)
        wandb.init(
            project=config.get("wandb_project", "Rec2Vec"),
            group=config.get("wandb_group", None),
            name=wandb_name
        )
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
    
    dataset = dataset.rename_column("query", "anchor")
    dataset = dataset.rename_column("positive_example", "positive")
    dataset = dataset.rename_column("negative_example", "negative")

    # remove rows where positive is the same as negative
    dataset = dataset.filter(lambda x: x["positive"] != x["negative"])

    V = config.get("V", 25)
    easy_negative_value = int(config.get("easy_negative_value", 15))
    distance_transform = config.get("distance_transform", DistanceTransform.LINEAR.value)
    distance_transform_alpha = float(config.get("distance_transform_alpha", 5.0))
    
    if config["training_style"] in (TrainingStyle.BASELINE_TRIPLET.value, TrainingStyle.INFONCE.value):
        cols_to_keep = ["anchor", "positive", "negative"]
        if "query_distance" in dataset.column_names:
            cols_to_keep.append("query_distance")
        if "negative_example_source" in dataset.column_names:
            cols_to_keep.append("negative_example_source")
        if "split" in dataset.column_names:
            cols_to_keep.append("split")
        dataset = dataset.select_columns(cols_to_keep)
    elif config["training_style"] in (TrainingStyle.OURS_MSE.value, TrainingStyle.OURS_MSE_REVERSED.value, TrainingStyle.CLASSIC_MSE.value, TrainingStyle.COSENT.value):
        dataset = dataset.rename_column("query_distance", "label")
        # -1 is the easy-negative sentinel in both pipelines, not a real distance.
        # Map it to the actual distance we want easy negatives trained toward.
        n_easy = sum(1 for v in dataset["label"] if v == EASY_NEGATIVE_SENTINEL)
        print(f"Remapping {n_easy}/{len(dataset)} easy-negative labels to {easy_negative_value}")
        dataset = dataset.map(lambda x: {"label": easy_negative_value if x["label"] == EASY_NEGATIVE_SENTINEL else x["label"]})
        # scale, then shape the normalized distance (linear leaves it untouched)
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
    
    if "split" in dataset.column_names:
        # Leakage-free split precomputed by utils/leakage_split.py: no query, product, or
        # ESCI item is shared across splits.
        train_dataset = dataset.filter(lambda x: x["split"] == "train")
        eval_dataset = dataset.filter(lambda x: x["split"] == "validation")
        test_dataset = dataset.filter(lambda x: x["split"] == "test")
        aux_cols = [c for c in ("split", "item", "positive_id", "negative_id") if c in dataset.column_names]
        train_dataset = train_dataset.remove_columns(aux_cols)
        eval_dataset = eval_dataset.remove_columns(aux_cols)
        test_dataset = test_dataset.remove_columns(aux_cols)
    else:
        # Legacy query-only split: documents still leak across splits.
        import pandas as pd
        unique_queries = pd.Series(dataset["anchor"]).drop_duplicates().tolist()

        # Split queries 80/10/10
        from sklearn.model_selection import train_test_split
        train_queries, temp_queries = train_test_split(unique_queries, test_size=0.2, random_state=42)
        eval_queries, test_queries = train_test_split(temp_queries, test_size=0.5, random_state=42)

        # Convert to sets for faster lookup
        train_queries_set = set(train_queries)
        eval_queries_set = set(eval_queries)
        test_queries_set = set(test_queries)

        # Filter original dataset based on queries
        train_dataset = dataset.filter(lambda x: x["anchor"] in train_queries_set)
        eval_dataset = dataset.filter(lambda x: x["anchor"] in eval_queries_set)
        test_dataset = dataset.filter(lambda x: x["anchor"] in test_queries_set)


    print(f"Dataset loaded: {len(train_dataset)} train, {len(eval_dataset)} eval, {len(test_dataset)} test examples")

    # 4. Define a loss function
    # loss = MultipleNegativesRankingLoss(model)

    raw_eval_dataset = eval_dataset

    if config["training_style"] in (TrainingStyle.CLASSIC_MSE.value, TrainingStyle.COSENT.value):
        # CoSENT scores every pair in a batch against every other, so a duplicated positive
        # pair counts twice in the comparison set. The dataset holds two rows per triple
        # (hard negative, easy negative) sharing one positive, so emit it from the hard row
        # only. CosineSimilarityLoss scores each pair independently and is left as it was.
        dedup_positive = (
            config["training_style"] == TrainingStyle.COSENT.value
            and "negative_example_source" in train_dataset.column_names
        )

        def to_pairs(batch):
            anchors = []
            others = []
            labels = []
            # iterate over batch size
            n = len(batch['anchor'])
            for i in range(n):
                a = batch['anchor'][i]
                p = batch['positive'][i]
                neg = batch['negative'][i]
                l = batch['label'][i]
                is_easy = dedup_positive and batch['negative_example_source'][i] == "random"

                # Positive pair: (anchor, positive) -> sim 1.0
                if not is_easy:
                    anchors.append(a)
                    others.append(p)
                    labels.append(1.0)
                
                # Negative pair: (anchor, negative) -> sim 1.0 - label
                # (Assuming label is distance-based scaled [0,1])
                anchors.append(a)
                others.append(neg)
                labels.append(1.0 - l)
            
            return {"sentence_A": anchors, "sentence_B": others, "label": labels}

        train_dataset = train_dataset.map(to_pairs, batched=True, remove_columns=train_dataset.column_names)
        eval_dataset = eval_dataset.map(to_pairs, batched=True, remove_columns=eval_dataset.column_names)

    if config["training_style"] == TrainingStyle.BASELINE_TRIPLET.value:
        loss = losses.TripletLoss(model=model,distance_metric=losses.TripletDistanceMetric.COSINE, triplet_margin=0.2)
    elif config["training_style"] == TrainingStyle.INFONCE.value:
        # The standard bi-encoder objective: cross-entropy over in-batch negatives, with
        # our labeled hard negative as the extra column. Ordering only, no graded signal.
        loss = losses.MultipleNegativesRankingLoss(model=model)
    elif config["training_style"] == TrainingStyle.COSENT.value:
        # Consumes the same graded label as ours-mse but ordinally: every pair with a
        # higher label must score higher. Separates "graded supervision" from
        # "margin regression" as the source of any gain.
        loss = losses.CoSENTLoss(model=model)
    elif config["training_style"] in (TrainingStyle.OURS_MSE.value, TrainingStyle.OURS_MSE_REVERSED.value):
        loss = losses.MarginMSELoss(model=model,similarity_fct=util.pairwise_cos_sim)
    elif config["training_style"] == TrainingStyle.CLASSIC_MSE.value:
        loss = losses.CosineSimilarityLoss(model=model)
    else:
        raise ValueError(f"Invalid training style: {config['training_style']}")

    # 5. (Optional) Specify training arguments
    per_device_train_batch_size = min(train_config["per_device_max_batch_size"], train_config["global_batch_size"] // torch.cuda.device_count())
    gradient_accumulation_steps = train_config["global_batch_size"] // (per_device_train_batch_size * torch.cuda.device_count())
    assert train_config["global_batch_size"] // (per_device_train_batch_size * torch.cuda.device_count()) == train_config["global_batch_size"] / (per_device_train_batch_size * torch.cuda.device_count()), f"Global batch size {train_config['global_batch_size']} is not divisible by the product of per-device batch size {per_device_train_batch_size} and the number of devices {torch.cuda.device_count()}"
    assert gradient_accumulation_steps > 0, f"Gradient accumulation steps {gradient_accumulation_steps} is less than 1"

    # NO_DUPLICATES keeps a batch free of repeated values, which MultipleNegativesRankingLoss
    # needs so an anchor's own positive is never used as an in-batch negative. CoSENT is the
    # opposite case: its loss is a comparison over the pairs that share a batch, and both pairs
    # of a query repeat that query's text, so NO_DUPLICATES would place them in different
    # batches and silently drop every within-query comparison.
    batch_sampler = BatchSamplers[train_config["batch_sampler"]]
    if config["training_style"] == TrainingStyle.COSENT.value and batch_sampler == BatchSamplers.NO_DUPLICATES:
        print("cosent: overriding batch_sampler NO_DUPLICATES -> BATCH_SAMPLER")
        batch_sampler = BatchSamplers.BATCH_SAMPLER

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
        batch_sampler=batch_sampler,
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
    swap_pos_neg = config["training_style"] == TrainingStyle.OURS_MSE_REVERSED.value
    train_dataset_for_trainer = prepare_dataset_for_trainer(train_dataset, swap_pos_neg=swap_pos_neg)
    eval_dataset_for_trainer = prepare_dataset_for_trainer(eval_dataset, swap_pos_neg=swap_pos_neg)

    print(f"Trainer train dataset columns: {train_dataset_for_trainer.column_names}")
    print(f"Trainer eval dataset columns: {eval_dataset_for_trainer.column_names}")
    print(f"Final evaluator dataset columns (val): {raw_eval_dataset.column_names}")

    # print("test dataset columns: ", test_dataset.column_names)
    # print("train dataset columns: ", train_dataset.column_names)
    # print("eval dataset columns: ", eval_dataset.column_names)

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset_for_trainer,
        eval_dataset=eval_dataset_for_trainer,
        loss=loss,
        # evaluator=dev_evaluator,
    )

    trainer.train()

    # 8. Evaluate the trained model
    if int(os.environ.get("LOCAL_RANK", -1)) <= 0:
        # Save before evaluating: a crash in the val evaluation must not discard a
        # completed training run.
        model.save_pretrained(os.path.join(train_config["output_dir"], "final"))

        evaluate_model(
            model,
            raw_eval_dataset,
            split_name="val",
            log_predictions_table_name=args.log_predictions_table,
        )


if __name__ == "__main__":
    main()
