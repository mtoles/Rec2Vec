#!/usr/bin/env python3
"""
Logical Operators in First Stage Recall

This project benchmarks and trains models for accurate first-stage recall on products
when queries contain multiple logical operators (and, or, not) using the ESCI dataset.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional, Callable
import json
from datasets import load_dataset
from tqdm import tqdm
import openai
from jsonschema import validate, ValidationError
import pandas as pd
import os
import argparse
import random
from random import randint
import numpy as np
import re
from copy import deepcopy
from multiprocessing import Pool, cpu_count
from functools import partial

from utils.sat import generate_random_sat_fn, generate_simple_sat_fn
from utils.retry import retry_with_fallback, print_cost_report, is_gemini_model


tqdm.pandas()
random.seed(42)
np.random.seed(42)

# Configuration
N_FEATURES = 5
ESCI_DATASET_URL = "https://huggingface.co/datasets/tasksource/esci"


# Global logger
logger = logging.getLogger(__name__)




def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("recall_pipeline.log"), logging.StreamHandler()],
    )

    # Disable httpx logs to reduce noise
    logging.getLogger("httpx").setLevel(logging.WARNING)

    # Use the global logger
    logger.info("Logging initialized for Logical Recall Pipeline")

    return logger


def load_esci_dataset(
    n_examples: Optional[int] = None,
    split: str = "train",
) -> List[Dict[str, Any]]:
    """
    Load and filter the ESCI dataset from HuggingFace or from cached JSON file.

    Args:
        n_examples: Optional limit on number of examples to process (for testing)
        split: Dataset split to load ("train", "test", etc.)

    Returns:
        List of processed examples with query and product information
    """
    # Load the ESCI dataset from HuggingFace
    dataset = load_dataset("tasksource/esci")[split]
    if n_examples:
        dataset = dataset.select(range(n_examples))

    # Convert to df
    df = pd.DataFrame(dataset)
    print(f"irrelevant examples: {df[df['esci_label'] == 'Irrelevant']}")
    print(f"substitute examples: {df[df['esci_label'] == 'Substitute']}")
    print(f"exact examples: {df[df['esci_label'] == 'Exact']}")
    logger.info(f"Loaded {len(df)} examples from {split} dataset")
    subs_df = df[
        df["esci_label"].progress_apply(lambda x: x in ["Substitute", "Irrelevant"])
    ]

    # Create a lookup dictionary for exact matches - O(n) preprocessing
    exact_lookup = {}
    for idx, row in df[df["esci_label"] == "Exact"].iterrows():
        query_id = row["query_id"]
        if query_id not in exact_lookup:
            exact_lookup[query_id] = idx
    
    # Map to exact_id using dictionary lookup - O(1) per lookup
    subs_df["exact_id"] = subs_df["query_id"].map(exact_lookup)
    subs_df = subs_df[subs_df["exact_id"].notna()]
    subs_df["exact_id"] = subs_df["exact_id"].astype(int)

    logger.info(
        f"Reduced {split} dataset from {len(df)} to {len(subs_df)} ({len(subs_df) / len(df) * 100:.2f}%)"
    )

    # now that we have the index of the exact product, we can return the list of tuples
    ds_list = []
    product_cols = ["product_id", "product_title", "product_text"]
    for _, row in tqdm(subs_df.iterrows(), total=len(subs_df)):
        positive_product = dict(df.iloc[row["exact_id"]][product_cols])
        hard_neg_product = dict(row[product_cols])
        # pick a random easy negative
        easy_neg_id = df.sample(1).index[0]
        easy_neg_product = dict(df.iloc[easy_neg_id][product_cols])
        ds_list.append(
            {
                "query": row["query"],
                "positive_product": positive_product,
                "hard_neg_product": hard_neg_product,
                "easy_neg_product": easy_neg_product,
            }
        )

    return ds_list


def infer_item_and_features(query: str, model_id: str) -> Optional[Dict[str, Any]]:
    """
    Extract features from the query using LLM with JSON schema validation and retry logic.

    Args:
        query: Natural language query

    Returns:
        Dict containing 'item' and 'features' keys with validated JSON structure
    """
    # Define the expected JSON schema
    schema = {
        "type": "object",
        "properties": {
            "item": {"type": "string"},
            "features": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["item"],
        "additionalProperties": False,
    }

    prompt = "Query: {query}\n\nWhat are the item and features(s) (if any) mentioned in this query? The item is the product. Features describe the product. For example, 'white shirt' is the item and 'white' is the feature. Return ONLY JSON: {{'item': 'extracted_item', 'features': ['extracted_feature1', 'extracted_feature2', ...]}}"

    messages = [{"role": "user", "content": prompt.format(query=query)}]

    def validate_response(content: str) -> bool:
        """Validate that the response is valid JSON and matches the schema."""
        try:
            parsed_response = json.loads(content)
            validate(instance=parsed_response, schema=schema)
            return True
        except (json.JSONDecodeError, ValidationError):
            return False

    response = retry_with_fallback(
        messages=messages,
        validation_func=validate_response,
        max_retries=5,
        fallback_value=None,
        model_id=model_id,
    )

    if response is None:
        return None #{"item": query, "features": []}

    # Parse the validated response
    parsed_response = json.loads(response)
    return parsed_response


def get_common_and_differentiating_features(
    positive_product: Dict[str, Any], substitute_irrelevant_product: Dict[str, Any], model_id: str
) -> Optional[Tuple[List[str], List[str], List[str], List[str]]]:
    """
    Get the common and differentiating features between the positive and substitute/irrelevant products.
    """
    similar_prompt = "### Product A:\n\n{positive_product}\n\n### Product B:\n\n{substitute_irrelevant_product}\n\nList up to {N_COMMON_FEATURES} features common to both products (\"common_features\"), up to {N_FEATURES} features unique to product A (\"unique_features_a\"), up to {N_FEATURES} features unique to product B (\"unique_features_b\"), and up to {N_FEATURES} features that do not apply to either product (\"neither_features\")). When generating common_features, ensure they are present in both products in some way. For example, if product A is \"low fat\" and product B in \"low sugar\", then \"low calories\" could be in common_features. If there are fewer than {N_COMMON_FEATURES} common features, generate as many as possible. Do not assume anything about products not written in the description. When generating neither_features, ensure they are opposite or mutually exclusive with features in one or both of the products. For example, if product A is \"red\" and product B is \"orange\", then \"blue\" could be in neither_features. Do not generate negated features, such as \"not red\". Use your imagination and create diverse neither_features. All features should be objective and no more than 5 words. Return ONLY JSON: {{'common_features': ['feature1', 'feature2', ...], 'unique_features_a': ['feature1', 'feature2', ...], 'unique_features_b': ['feature1', 'feature2', ...], 'neither_features': ['feature1', 'feature2', ...]}}"
    messages = [
        {
            "role": "user",
            "content": similar_prompt.format(
                positive_product=positive_product,
                substitute_irrelevant_product=substitute_irrelevant_product,
                N_FEATURES=N_FEATURES,
                N_COMMON_FEATURES=N_FEATURES,
            ),
        }
    ]
    schema = {
        "type": "object",
        "properties": {
            "common_features": {"type": "array", "items": {"type": "string"}},
            "unique_features_a": {"type": "array", "items": {"type": "string"}},
            "unique_features_b": {"type": "array", "items": {"type": "string"}},
            "neither_features": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "common_features",
            "unique_features_a",
            "unique_features_b",
            "neither_features",
        ],
        "additionalProperties": False,
    }

    def validate_response(content: str) -> bool:
        """Validate that the response is valid JSON and matches the schema."""
        try:
            parsed_response = json.loads(content)
            validate(instance=parsed_response, schema=schema)
            return True
        except (json.JSONDecodeError, ValidationError):
            return False

    response = retry_with_fallback(
        messages=messages,
        validation_func=validate_response,
        max_retries=5,
        fallback_value=None,
        model_id=model_id,
    )
    
    if response is None:
        return None, None, None, None
        
    try:
        parsed_response = json.loads(response)
        return (
            parsed_response["common_features"],
            parsed_response["unique_features_a"],
            parsed_response["unique_features_b"],
            parsed_response["neither_features"],
        )
    except json.JSONDecodeError:
        return None, None, None, None


def fn_seq_to_nl(seq: List[str], features: List[str]) -> str:
    """
    Convert a sequence of tokens to a natural language description.
    Replace args[i] with features[i]
    """
    output = []
    for token in seq:
        if re.match(r"args\[(\d+)\]", token):
            index = int(re.match(r"args\[(\d+)\]", token).group(1))
            output.append(features[index])
        else:
            output.append(token)
    return " ".join(output)


def generate_example(
    item: str,
    common_features: List[str],
    unique_pos_features: List[str],
    unique_neg_features: List[str],
    neither_features: List[str],
    max_distance: int,
) -> Dict[str, Any]:
    """
    Generate a query fn based on a subset of features such that exactly one product satisfies the query function.
    """
    # all_features = (
    #     common_features + unique_pos_features + unique_neg_features + neither_features
    # )

    # pos_bin_features = (
    #     [True] * len(common_features)
    #     + [True] * len(unique_pos_features)
    #     + [False] * len(unique_neg_features)
    #     + [False] * len(neither_features)
    # )
    # neg_bin_features = (
    #     [True] * len(common_features)
    #     + [False] * len(unique_pos_features)
    #     + [True] * len(unique_neg_features)
    #     + [False] * len(neither_features)
    # )

    # Randomly pick a target distance for this example

    # common, pos, neg, neither
    query_distance = randint(1, max_distance)
    # features_indices = random.sample(list(range(len(pos_bin_features))), query_distance)
    # sample query_distance features from pos_bin_features and negations of neg_bin_features
    n_pos_features = min(randint(0, query_distance), len(unique_pos_features))
    n_neg_features = min(query_distance - n_pos_features, len(unique_neg_features))
    n_common_features = min(randint(0, query_distance), len(common_features))
    n_neither_features = min(randint(0, query_distance), len(neither_features))

    selected_pos_features_indices    = random.sample(list(range(len(unique_pos_features))), n_pos_features)
    selected_neg_features_indices    = random.sample(list(range(len(unique_neg_features))), n_neg_features)
    selected_common_features_indices = random.sample(list(range(len(common_features))), n_common_features)
    selected_neither_features_indices = random.sample(list(range(len(neither_features))), n_neither_features)

    selected_pos_features = [unique_pos_features[i] for i in selected_pos_features_indices]
    selected_neg_features = [unique_neg_features[i] for i in selected_neg_features_indices]
    selected_common_features = [common_features[i] for i in selected_common_features_indices]
    selected_neither_features = [neither_features[i] for i in selected_neither_features_indices]

    # pos_features = [pos_bin_features[i] for i in features_indices]
    # neg_features = [neg_bin_features[i] for i in features_indices]
    # common_features = [common_features[i] for i in features_indices]
    # query_fn, source_code, seq = generate_random_sat_fn(query_len)
    # query_fn, source_code, seq = generate_simple_sat_fn(pos_features=pos_features, neg_features=neg_features, common_features=common_features)
    nl_query = f"I am looking for: \"{item}\" that has: {', '.join(selected_pos_features + selected_common_features)}; and does not have: {', '.join(selected_neg_features + selected_neither_features)}"

    return {
        "item": item,
        "nl_query": nl_query,
        "selected_pos_features": selected_pos_features,
        "selected_neg_features": selected_neg_features,
        "selected_common_features": selected_common_features,
        "selected_neither_features": selected_neither_features,
        "query_distance": query_distance,
        # Store the full generated features
        "full_common_features": common_features,
        "full_unique_pos_features": unique_pos_features,
        "full_unique_neg_features": unique_neg_features,
        "full_neither_features": neither_features,
    }

        # calculate nl stuff

        # if query_fn(*pos_features) != query_fn(*neg_features):
        #     nl_features = [all_features[i] for i in features_indices]
        #     nl_query = fn_seq_to_nl(seq, nl_features)
        #     pos_edit_distance = calculate_edit_distance(query_fn, pos_features)
        #     neg_edit_distance = calculate_edit_distance(query_fn, neg_features)

        #     assert pos_edit_distance == 0 or neg_edit_distance == 0
        #     assert pos_edit_distance != neg_edit_distance

        #     # Check if we found the target distance
        #     if (
        #         pos_edit_distance == target_distance
        #         or neg_edit_distance == target_distance
        #     ):
        #         return {
        #             "query_fn": query_fn,
        #             "source_code": source_code,
        #             "pos_features": (
        #                 pos_features if pos_edit_distance == 0 else neg_features
        #             ),
        #             "neg_features": (
        #                 neg_features if pos_edit_distance == 0 else pos_features
        #             ),
        #             "nl_query": nl_query,
        #             "pos_edit_distance": pos_edit_distance,
        #             "neg_edit_distance": neg_edit_distance,
        #         }

        # i += 1
        # if i > 1000:
        #     raise ValueError("Failed to generate a query fn")


def process_single_example(example: Dict[str, Any], max_distance: int, model_id: str) -> Dict[str, Any]:
    """
    Process a single example through the feature extraction and query generation pipeline.
    
    Args:
        example: Single example from the dataset
        max_distance: Maximum distance for query generation
        model_id: Model ID to use for inference
        
    Returns:
        Processed example with generated features and query
    """
    query = example["query"]
    
    # Step 2: Feature Extraction Pipeline
    query_extraction_result = infer_item_and_features(query, model_id=model_id)
    if query_extraction_result is None:
        return None
    query_features = query_extraction_result["features"] if "features" in query_extraction_result else []
    query_item = query_extraction_result["item"]

    (
        common_features,
        unique_pos_features,
        unique_neg_features,
        neither_features,
    ) = get_common_and_differentiating_features(
        example["positive_product"], example["hard_neg_product"], model_id=model_id
    )
    if common_features is None or unique_pos_features is None or unique_neg_features is None or neither_features is None:
        return None
    generated_example = generate_example(
        item=query_item,
        common_features=common_features,
        unique_pos_features=unique_pos_features,
        unique_neg_features=unique_neg_features,
        neither_features=neither_features,
        max_distance=max_distance,
    )
    
    # Check there are no overlapping keys between example and generated_example
    if set(example.keys()) & set(generated_example.keys()):
        raise ValueError("Overlapping keys between example and generated_example")
    
    # Rename example["query"] to example["original_query"]
    example["original_query"] = example.pop("query")
    
    example.update(generated_example)
    return example


def process_wrapper(args_tuple):
    """Wrapper function for multiprocessing that can be pickled."""
    example, max_distance, model_id = args_tuple
    return process_single_example(example, max_distance, model_id)


def calculate_edit_distance(query_fn: Callable, product_features: List[bool]) -> int:
    """
    Calculate minimum number of feature edits needed for product to satisfy query.

    Args:
        query_fn: Boolean query function
        product_features: array of boolean values representing whether the product has the feature

    Returns:
        Minimum edit distance (integer)
    """
    n_features = len(product_features)
    closest_feature_set = None
    min_edit_distance = float("inf")
    # calculate all possible feature combinations
    for i in range(2**n_features):
        feature_set = [bool(i & (1 << j)) for j in range(n_features)]
        if query_fn(*feature_set):
            distance = sum(
                1 for j in range(n_features) if feature_set[j] != product_features[j]
            )
            if distance < min_edit_distance:
                min_edit_distance = distance
                closest_feature_set = feature_set
    assert min_edit_distance != float("inf")
    return min_edit_distance


def load_dataset_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """
    Load dataset from a JSONL file.

    Args:
        filepath: Path to the JSONL file

    Returns:
        List of example records
    """
    import os

    if not os.path.exists(filepath):
        logger.warning(f"Dataset file {filepath} does not exist")
        return []

    dataset = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                dataset.append(json.loads(line.strip()))

    logger.info(f"Loaded {len(dataset)} examples from {filepath}")
    return dataset


def save_dataset_jsonl(dataset: List[Dict[str, Any]], filepath: str):
    """
    Save the final dataset to a JSONL file.

    Args:
        dataset: List of example records
        filepath: Output file path
    """

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    with open(filepath, "w", encoding="utf-8") as f:
        for example in dataset:
            # _example = deepcopy(example)
            # _example.pop("query_fn")
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    logger.info(f"Dataset saved to {filepath} with {len(dataset)} examples")


def generate_dataset_summary_md(dataset: List[Dict[str, Any]], output_filepath: str, n_examples: int = 10):
    """
    Generate a markdown summary of the first n examples from the dataset.
    
    Args:
        dataset: List of example records
        output_filepath: Path for the markdown file
        n_examples: Number of examples to include in summary (default: 10)
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)
    
    # Take first n examples
    examples_to_summarize = dataset[:n_examples]
    
    with open(output_filepath, "w", encoding="utf-8") as f:
        f.write("# Dataset Summary\n\n")
        f.write(f"This document summarizes the first {len(examples_to_summarize)} examples from the dataset.\n\n")
        
        for i, example in enumerate(examples_to_summarize, 1):
            f.write(f"## Example {i}\n\n")
            
            # Original query
            f.write(f"**Original Query:** {example.get('original_query', 'N/A')}\n\n")
            
            # Generated query
            f.write(f"**Generated Query:** {example.get('nl_query', 'N/A')}\n\n")
            
            # Positive product
            pos_product = example.get('positive_product', {})
            f.write(f"**Positive Product:**\n")
            f.write(f"- **Title:** {pos_product.get('product_title', 'N/A')}\n")
            f.write(f"- **Description:** {pos_product.get('product_text', 'N/A')}\n\n")
            
            # Hard negative product
            hard_neg_product = example.get('hard_neg_product', {})
            f.write(f"**Hard Negative Product:**\n")
            f.write(f"- **Title:** {hard_neg_product.get('product_title', 'N/A')}\n")
            f.write(f"- **Description:** {hard_neg_product.get('product_text', 'N/A')}\n\n")
            
            # Full features
            f.write(f"**Full Generated Features:**\n")
            f.write(f"- **Common Features:** {', '.join(example.get('full_common_features', []))}\n")
            f.write(f"- **Unique Positive Features:** {', '.join(example.get('full_unique_pos_features', []))}\n")
            f.write(f"- **Unique Negative Features:** {', '.join(example.get('full_unique_neg_features', []))}\n")
            f.write(f"- **Neither Features:** {', '.join(example.get('full_neither_features', []))}\n\n")
            
            # Selected features (for comparison)
            f.write(f"**Selected Features (used in query):**\n")
            f.write(f"- **Selected Positive:** {', '.join(example.get('selected_pos_features', []))}\n")
            f.write(f"- **Selected Negative:** {', '.join(example.get('selected_neg_features', []))}\n")
            f.write(f"- **Selected Common:** {', '.join(example.get('selected_common_features', []))}\n")
            f.write(f"- **Selected Neither:** {', '.join(example.get('selected_neither_features', []))}\n")
            f.write(f"- **Query Distance:** {example.get('query_distance', 'N/A')}\n\n")
            
            f.write("---\n\n")
    
    logger.info(f"Dataset summary saved to {output_filepath}")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Logical Operators in First Stage Recall - Process ESCI dataset"
    )
    parser.add_argument(
        "--n-examples",
        type=int,
        default=None,
        help="Number of examples to process (default: None for all examples)",
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=10,
        help="Maximum distance to generate examples for",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="gpt-5-nano",
        help="Model ID to use for inference (default: gpt-5-nano)",
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    # ============================================================================
    # SETUP AND CONFIGURATION
    # ============================================================================
    # Parse command line arguments
    args = parse_arguments()

    # Setup logging
    model_id = args.model_id


    logger = setup_logging()
    logger.info(f"Starting Logical Recall Pipeline with n_examples={args.n_examples}, model_id={model_id}")

    # ============================================================================
    # CHECK FOR CACHED OUTPUT FILE (EARLY EXIT TO SAVE TIME)
    # ============================================================================
    output_file = f"dataset/feature-distance-dataset_{model_id}_{args.n_examples}.jsonl"
    if os.path.exists(output_file):
        logger.info(f"JSONL file already exists: {output_file}. Loading from cache and skipping all processing.")
        examples = load_dataset_jsonl(output_file)
        logger.info(f"Loaded {len(examples)} examples from {output_file}")
    else:
        # ============================================================================
        # DATA LOADING (only if not cached)
        # ============================================================================
        logger.info("Loading dataset from HuggingFace...")
        train_ds = load_esci_dataset(n_examples=args.n_examples, split="train")

        # ============================================================================
        # FEATURE EXTRACTION AND QUERY GENERATION
        # ============================================================================
        logger.info("Starting feature extraction and query generation...")

        # Determine number of workers based on model type
        if is_gemini_model(model_id):
            num_workers = int(os.getenv("N_WORKERS", "1"))
        else:
            num_workers = 1
        
        logger.info(f"Using {num_workers} worker(s) for parallel processing (model: {model_id})")

        if num_workers == 1:
            # Sequential processing for non-Gemini models
            logger.info("Using sequential processing")
            examples = []
            for i, example in tqdm(enumerate(train_ds or []), total=len(train_ds)):
                query = example["query"]

                if i % 100 == 0:
                    logger.info(f"Processing example {i}...")

                # Step 2: Feature Extraction Pipeline
                query_extraction_result = infer_item_and_features(query, model_id=model_id)
                query_features = query_extraction_result["features"] if "features" in query_extraction_result else []
                query_item = query_extraction_result["item"]

                (
                    common_features,
                    unique_pos_features,
                    unique_neg_features,
                    neither_features,
                ) = get_common_and_differentiating_features(
                    example["positive_product"], example["hard_neg_product"], model_id=model_id
                )
                generated_example = generate_example(
                    item=query_item,
                    common_features=common_features,
                    unique_pos_features=unique_pos_features,
                    unique_neg_features=unique_neg_features,
                    neither_features=neither_features,
                    max_distance=args.max_distance,
                )
                # check there are no overlapping keys between example and generated_example
                if set(example.keys()) & set(generated_example.keys()):
                    raise ValueError("Overlapping keys between example and generated_example")
                # rename example["query"] to example["original_query"]
                example["original_query"] = example.pop("query")

                example.update(generated_example)
                examples.append(example)
        else:
            # Parallel processing for Gemini models
            logger.info(f"Using parallel processing with {num_workers} workers")
            
            # Process examples in parallel using simple Pool
            logger.info(f"Processing {len(train_ds)} examples with {num_workers} workers")
            
            # Prepare arguments for multiprocessing
            args_list = [(example, args.max_distance, model_id) for example in train_ds]
            
            with Pool(processes=num_workers) as pool:
                # Use map with chunksize for better performance
                results = list(tqdm(
                    pool.imap(process_wrapper, args_list, chunksize=1),
                    total=len(args_list),
                    desc="Processing examples"
                ))
                examples = [result for result in results if result is not None]
            
            # Filter out None values (failed processing)
            examples = [ex for ex in examples if ex is not None]
            logger.info(f"Successfully processed {len(examples)} out of {len(train_ds) if train_ds else 0} examples")

        # ============================================================================
        # SAVE PROCESSED DATASET TO CACHE
        # ============================================================================
        logger.info("Saving dataset...")
        save_dataset_jsonl(examples, output_file)
        logger.info(f"Dataset saved to {output_file}")

    # ============================================================================
    # GENERATE MARKDOWN SUMMARY
    # ============================================================================
    logger.info("Generating dataset summary...")
    summary_file = f"dataset/feature-distance-dataset_{model_id}_{args.n_examples}_summary.md"
    generate_dataset_summary_md(examples, summary_file, n_examples=10)
    logger.info(f"Dataset summary saved to {summary_file}")

    # ============================================================================
    # PROCESS AND SAVE AS HUGGINGFACE DATASET (FOR TRAINING)
    # ============================================================================
    logger.info("Processing dataset for training...")
    from datasets import Dataset as HFDataset, concatenate_datasets
    
    raw_dataset = HFDataset.from_list(examples)
    
    # Create hard examples dataset
    def add_hard_examples(example):
        return {
            "original_query": example["original_query"],
            "nl_query": example["nl_query"],
            "positive_example": f"Product Title: {example['positive_product']['product_title']}\nProduct Description: {example['positive_product']['product_text']}",
            "negative_example": f"Product Title: {example['hard_neg_product']['product_title']}\nProduct Description: {example['hard_neg_product']['product_text']}",
            "query_distance": float(example["query_distance"])
        }
    
    # Create easy examples dataset  
    def add_easy_examples(example):
        return {
            "original_query": example["original_query"],
            "nl_query": example["nl_query"],
            "positive_example": f"Product Title: {example['positive_product']['product_title']}\nProduct Description: {example['positive_product']['product_text']}",
            "negative_example": f"Product Title: {example['easy_neg_product']['product_title']}\nProduct Description: {example['easy_neg_product']['product_text']}",
            "query_distance": -1.0
        }
    
    hard_dataset = raw_dataset.map(add_hard_examples, desc="Processing hard examples", remove_columns=raw_dataset.column_names)
    easy_dataset = raw_dataset.map(add_easy_examples, desc="Processing easy examples", remove_columns=raw_dataset.column_names)
    processed_dataset = concatenate_datasets([hard_dataset, easy_dataset])
    
    # Save processed dataset
    dataset_output_dir = f"dataset/processed/feature-distance-dataset_{model_id}_{args.n_examples}"
    os.makedirs(os.path.dirname(dataset_output_dir), exist_ok=True)
    processed_dataset.save_to_disk(dataset_output_dir)
    logger.info(f"Processed dataset saved to {dataset_output_dir}")

    # ============================================================================
    # FINAL CLEANUP AND LOGGING
    # ============================================================================
    dataset_size = len(examples) if examples else 0
    logger.info(f"Dataset creation completed. Generated {dataset_size} examples.")

    # Print API cost report
    print_cost_report()

    # TODO: Clean up temporary files and resources

    return examples


if __name__ == "__main__":
    main()
