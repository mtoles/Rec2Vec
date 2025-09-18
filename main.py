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

from utils.sat import generate_random_sat_fn
from utils.retry import retry_with_fallback, print_cost_report


tqdm.pandas()
random.seed(42)
np.random.seed(42)

# Configuration
N_FEATURES = 5
MAX_QUERY_LEN = 5
ESCI_DATASET_URL = "https://huggingface.co/datasets/tasksource/esci"
# MODEL_ID = "gpt-5-mini"
MODEL_ID = "gpt-5-nano"

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
    import os

    # Define the JSON file path (include n_examples in filename if specified)
    if n_examples:
        json_filepath = f"dataset/{split}_{n_examples}.jsonl"
    else:
        json_filepath = f"dataset/{split}.jsonl"

    # Check if JSON file exists
    if os.path.exists(json_filepath):
        logger.info(f"Loading dataset from cached file: {json_filepath}")
        return load_dataset_jsonl(json_filepath)

    logger.info(f"JSON file {json_filepath} not found. Loading from HuggingFace...")

    # Load the ESCI dataset from HuggingFace
    dataset = load_dataset("tasksource/esci")[split]
    if n_examples:
        dataset = dataset.select(range(n_examples))

    # Convert to df
    df = pd.DataFrame(dataset)
    logger.info(f"Loaded {len(df)} examples from {split} dataset")
    subs_df = df[
        df["esci_label"].progress_apply(lambda x: x in ["Substitute", "Irrelevant"])
    ]

    def find_exact(query_id):
        hits = df[(df["query_id"] == query_id) & (df["esci_label"] == "Exact")]
        if len(hits) > 0:
            return hits.sample(1).index[0]
        else:
            return None

    subs_df["exact_id"] = subs_df["query_id"].progress_apply(find_exact)
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

    # Save the processed dataset to JSON for future use
    logger.info(f"Saving processed {split} dataset to {json_filepath}")
    save_dataset_jsonl(ds_list, json_filepath)

    return ds_list


def infer_item_and_features(query: str) -> Dict[str, Any]:
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
        "required": ["item", "features"],
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
        model_id=MODEL_ID,
    )

    if response is None:
        return {"item": query, "features": []}

    # Parse the validated response
    parsed_response = json.loads(response)
    return parsed_response


def get_common_and_differentiating_features(
    positive_product: Dict[str, Any], substitute_irrelevant_product: Dict[str, Any]
) -> Tuple[List[str], List[str]]:
    """
    Get the common and differentiating features between the positive and substitute/irrelevant products.
    """
    similar_prompt = "Product A: {positive_product}\nProduct B: {substitute_irrelevant_product}\n\nList up to {N_FEATURES} features common to both products, {N_FEATURES} features unique to product A, and {N_FEATURES} Features unique to product B, and {N_FEATURES} features that do not apply to either product. Features should be objective and no more than 5 words. Return ONLY JSON: {{'common_features': ['feature1', 'feature2', ...], 'unique_features_a': ['feature1', 'feature2', ...], 'unique_features_b': ['feature1', 'feature2', ...], 'neither_features': ['feature1', 'feature2', ...]}}"
    messages = [
        {
            "role": "user",
            "content": similar_prompt.format(
                positive_product=positive_product,
                substitute_irrelevant_product=substitute_irrelevant_product,
                N_FEATURES=N_FEATURES,
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
        model_id=MODEL_ID,
    )
    parsed_response = json.loads(response)
    return (
        parsed_response["common_features"],
        parsed_response["unique_features_a"],
        parsed_response["unique_features_b"],
        parsed_response["neither_features"],
    )


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
    common_features: List[str],
    unique_pos_features: List[str],
    unique_neg_features: List[str],
    neither_features: List[str],
) -> Dict[str, Any]:
    """
    Generate a query fn based on a subset of features such that exactly one product satisfies the query function.
    """
    all_features = (
        common_features + unique_pos_features + unique_neg_features + neither_features
    )

    pos_bin_features = (
        [True] * len(common_features)
        + [True] * len(unique_pos_features)
        + [False] * len(unique_neg_features)
        + [False] * len(neither_features)
    )
    neg_bin_features = (
        [True] * len(common_features)
        + [False] * len(unique_pos_features)
        + [True] * len(unique_neg_features)
        + [False] * len(neither_features)
    )
    i = 0
    while True:
        # common, pos, neg, neither
        query_len = randint(1, MAX_QUERY_LEN)
        features_indices = random.sample(list(range(len(pos_bin_features))), query_len)
        pos_features = [pos_bin_features[i] for i in features_indices]
        neg_features = [neg_bin_features[i] for i in features_indices]
        query_fn, source_code, seq = generate_random_sat_fn(query_len)

        # calculate nl stuff

        if query_fn(*pos_features) != query_fn(*neg_features):
            nl_features = [all_features[i] for i in features_indices]
            nl_query = fn_seq_to_nl(seq, nl_features)
            pos_edit_distance = calculate_edit_distance(query_fn, pos_features)
            neg_edit_distance = calculate_edit_distance(query_fn, neg_features)
            assert pos_edit_distance == 0 or neg_edit_distance == 0
            assert pos_edit_distance != neg_edit_distance

            return {
                "query_fn": query_fn,
                "source_code": source_code,
                "pos_features": (
                    pos_features if pos_edit_distance == 0 else neg_features
                ),
                "neg_features": (
                    neg_features if pos_edit_distance == 0 else pos_features
                ),
                "nl_query": nl_query,
                "pos_edit_distance": pos_edit_distance,
                "neg_edit_distance": neg_edit_distance,
            }

        i += 1
        if i > 1000:
            raise ValueError("Failed to generate a query fn")


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
            _example = deepcopy(example)
            _example.pop("query_fn")
            f.write(json.dumps(_example, ensure_ascii=False) + "\n")

    logger.info(f"Dataset saved to {filepath} with {len(dataset)} examples")


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
    return parser.parse_args()


def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_arguments()

    # Step 0: Setup and Configuration
    logger = setup_logging()
    logger.info(f"Starting Logical Recall Pipeline with n_examples={args.n_examples}")

    # Step 1: Data Loading and Preprocessing
    train_ds = load_esci_dataset(n_examples=args.n_examples, split="train")

    final_examples = []

    # Process each query group (simplified loop structure for now)
    logger.info("Starting feature extraction and query generation...")

    # QUIT HERE - Data loading and filtering is complete, feature extraction not yet implemented
    logger.info(
        "Data loading and filtering complete. Exiting before feature extraction."
    )
    # return train_ds
    examples = []

    for i, example in tqdm(enumerate(train_ds or []), total=len(train_ds)):
        query = example["query"]

        if i % 100 == 0:
            logger.info(f"Processing example {i}...")

        # Step 2: Feature Extraction Pipeline
        query_extraction_result = infer_item_and_features(query)
        query_features = query_extraction_result["features"]
        query_item = query_extraction_result["item"]

        (
            common_features,
            unique_pos_features,
            unique_neg_features,
            neither_features,
        ) = get_common_and_differentiating_features(
            example["positive_product"], example["hard_neg_product"]
        )
        example = generate_example(
            common_features,
            unique_pos_features,
            unique_neg_features,
            neither_features,
        )
        example["item"] = query_item
        example["features"] = query_features
        example["common_features"] = common_features
        example["unique_pos_features"] = unique_pos_features
        example["unique_neg_features"] = unique_neg_features
        example["neither_features"] = neither_features
        examples.append(example)

    # Print cost report after processing
    # print_cost_report()

    # raise NotImplementedError("Not implemented")

    # Step 6: Output Generation
    logger.info("Saving dataset...")
    output_file = f"dataset/feature-distance-dataset_{MODEL_ID}_{args.n_examples}.jsonl"

    save_dataset_jsonl(examples, output_file)
    logger.info(f"Dataset saved to {output_file}")

    # Step 7: Cleanup and Logging
    dataset_size = len(examples) if examples else 0
    logger.info(f"Dataset creation completed. Generated {dataset_size} examples.")

    # Print API cost report
    print_cost_report()

    # TODO: Clean up temporary files and resources

    return examples


if __name__ == "__main__":
    main()
