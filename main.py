#!/usr/bin/env python3
"""
Logical Operators in First Stage Recall

This project benchmarks and trains models for accurate first-stage recall on products
when queries contain multiple logical operators (and, or, not) using the ESCI dataset.
"""

import logging
from typing import Dict, List, Tuple, Any, Optional
import json
from datasets import load_dataset
from tqdm import tqdm
import openai
import jsonschema
from jsonschema import validate, ValidationError
import pandas as pd
import os

tqdm.pandas()


# Configuration
N_FEATURES = 4
ESCI_DATASET_URL = "https://huggingface.co/datasets/tasksource/esci"
MODEL_ID = "gpt-5-mini"

# Global logger
logger = logging.getLogger(__name__)


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("recall_pipeline.log"), logging.StreamHandler()],
    )

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


def filter_esci_labels(dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Filter dataset to remove Complement labels, keep Exact, Substitute, Irrelevant.

    Args:
        dataset: Raw ESCI dataset

    Returns:
        Filtered dataset without Complement labels
    """
    if not dataset:
        return []

    valid_labels = {"Exact", "Substitute", "Irrelevant"}
    filtered_data = [
        example for example in dataset if example.get("esci_label") in valid_labels
    ]

    return filtered_data


def extract_features_from_query(query: str) -> Dict[str, Any]:
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

    extraction_prompt = "Query: {query}\n\nWhat are the item and features(s) (if any) mentioned in this query? The item is the product. Features describe the product. For example, 'white shirt' is the item and 'white' is the feature. Return ONLY JSON: {{'item': 'extracted_item', 'features': ['extracted_feature1', 'extracted_feature2', ...]}}"

    max_retries = 5

    for attempt in range(max_retries):
        try:
            response = openai.chat.completions.create(
                model=MODEL_ID,
                messages=[
                    {"role": "user", "content": extraction_prompt.format(query=query)}
                ],
                response_format={"type": "json_object"},
            )

            # Parse JSON response
            parsed_response = json.loads(response.choices[0].message.content)

            # Validate against schema
            validate(instance=parsed_response, schema=schema)

            logger.info(
                f"Successfully extracted features from query on attempt {attempt + 1}"
            )
            return parsed_response

        except (json.JSONDecodeError, ValidationError) as e:
            logger.warning(f"JSON validation failed on attempt {attempt + 1}: {str(e)}")
            if attempt == max_retries - 1:
                logger.error(
                    f"Failed to extract valid features after {max_retries} attempts"
                )
                # Return a fallback structure
                return {"item": query, "features": []}

    # This should never be reached, but just in case
    return {"item": query, "features": []}


def generate_features_from_products(
    query: str,
    exact_product: Dict[str, Any],
    substitute_irrelevant_product: Dict[str, Any],
    query_features: List[str],
) -> List[str]:
    """
    Generate 1-4 features that differentiate Exact from Substitute/Irrelevant products.

    Args:
        query: Original query
        exact_product: Product with esci_label='Exact'
        substitute_irrelevant_product: Product with esci_label in ['Substitute', 'Irrelevant']
        query_features: Features already identified in the query

    Returns:
        List of differentiating features (total features = query_features + generated)
    """
    pass


def determine_feature_applicability(
    product: Dict[str, Any], features: List[str]
) -> Dict[str, bool]:
    """
    Determine which features apply to a given product using LLM.

    Args:
        product: Product data
        features: List of features to evaluate

    Returns:
        Dict mapping feature name to boolean indicating if it applies
    """
    pass


def generate_random_boolean_expression(features: List[str]) -> str:
    """
    Generate a random boolean SAT-style expression using all features exactly once.

    Grammar allows:
    - f_1...f_n used only once each
    - Parentheses must be properly closed
    - And, Or, Not, () operators

    Args:
        features: List of feature names

    Returns:
        Boolean expression as string
    """
    pass


def boolean_expression_to_natural_language(expression: str, features: List[str]) -> str:
    """
    Convert boolean expression to natural language description.

    Args:
        expression: Boolean expression string
        features: List of feature names

    Returns:
        Natural language query description
    """
    pass


def evaluate_boolean_expression(
    expression: str, feature_values: Dict[str, bool]
) -> bool:
    """
    Evaluate a boolean expression given feature values.

    Args:
        expression: Boolean expression string
        feature_values: Dict mapping feature names to boolean values

    Returns:
        Result of evaluating the expression
    """
    pass


def calculate_edit_distance(
    query_expression: str, product_features: Dict[str, bool]
) -> int:
    """
    Calculate minimum number of feature edits needed for product to satisfy query.

    Args:
        query_expression: Boolean query expression
        product_features: Dict of feature name to boolean values for product

    Returns:
        Minimum edit distance (integer)
    """
    pass


def select_easy_negative(
    dataset: List[Dict[str, Any]], positive_product: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Select an easy negative product randomly from the dataset.

    Args:
        dataset: Full dataset to choose from
        positive_product: The positive product to avoid selecting

    Returns:
        Randomly selected easy negative product
    """
    pass


def create_example_record(
    query: str,
    query_expression: str,
    positive_product: Dict[str, Any],
    hard_negative_product: Dict[str, Any],
    easy_negative_product: Dict[str, Any],
    features: List[str],
    edit_distance: int,
) -> Dict[str, Any]:
    """
    Create a complete example record for the dataset.

    Returns:
        Dict containing query, products, features, distance, etc.
    """
    pass


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
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    logger.info(f"Dataset saved to {filepath} with {len(dataset)} examples")


def add_train_test_split(
    dataset: List[Dict[str, Any]], train_ratio: float = 0.8
) -> List[Dict[str, Any]]:
    """
    Add train/test split column to dataset.

    Args:
        dataset: List of example records
        train_ratio: Fraction of examples for training

    Returns:
        Dataset with split column added
    """
    pass


def generate_summary_statistics(dataset: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Generate summary statistics for the dataset.

    Args:
        dataset: Final processed dataset

    Returns:
        Dict containing various statistics
    """
    pass


def generate_example_samples(
    dataset: List[Dict[str, Any]], n_samples: int = 5
) -> List[Dict[str, Any]]:
    """
    Generate sample examples for the report.

    Args:
        dataset: Final processed dataset
        n_samples: Number of examples to sample

    Returns:
        List of sample examples
    """
    pass


def create_report(dataset: List[Dict[str, Any]], output_path: str):
    """
    Generate a comprehensive report with statistics and examples.

    Args:
        dataset: Final processed dataset
        output_path: Path to save the report
    """
    pass


def setup_openai_client():
    """
    Setup OpenAI client for GPT-5 API.

    Returns:
        Configured OpenAI client object
    """
    pass


def parallel_openai_call(prompts: List[str]) -> List[str]:
    """
    Make parallel calls to GPT-5 for efficiency.

    Args:
        prompts: List of prompts to process

    Returns:
        List of responses
    """
    pass


def process_esci_dataset(
    n_examples: Optional[int] = None,
    output_file: str = "logical_recall_dataset.jsonl",
    report_file: str = "dataset_report.md",
) -> List[Dict[str, Any]]:
    """
    Main processing function that orchestrates the entire pipeline.

    Args:
        n_examples: Optional limit on examples to process (for testing)
        output_file: Output JSONL file path
        report_file: Output report file path

    Returns:
        Final processed dataset
    """
    pass


def main():
    """Main entry point."""
    # Step 0: Setup and Configuration
    logger = setup_logging()
    logger.info("Starting Logical Recall Pipeline")

    # TODO: Parse command line arguments (n_examples, output paths, etc.)
    # TODO: Set random seeds for reproducibility
    openai_client = setup_openai_client()
    logger.info("OpenAI GPT-5 client initialized")

    # Step 1: Data Loading and Preprocessing
    logger.info("Loading and filtering ESCI dataset...")
    train_ds = load_esci_dataset(n_examples=10000, split="train")  # Limit for testing
    logger.info(f"Loaded and filtered {len(train_ds)} train examples")

    # TODO: Group examples by query_id to find Exact + Substitute/Irrelevant pairs
    # TODO: Validate that we have proper pairs for training data generation

    final_examples = []

    # Process each query group (simplified loop structure for now)
    logger.info("Starting feature extraction and query generation...")

    # QUIT HERE - Data loading and filtering is complete, feature extraction not yet implemented
    logger.info(
        "Data loading and filtering complete. Exiting before feature extraction."
    )
    # return train_ds

    for i, example in enumerate(train_ds or []):
        query = example["query"]

        if i % 100 == 0:
            logger.info(f"Processing example {i}...")

        # Step 2: Feature Extraction Pipeline
        query_extraction_result = extract_features_from_query(query)
        query_features = query_extraction_result["features"]
        query_item = query_extraction_result["item"]

        # Find exact and substitute/irrelevant products for this query
        # TODO: Implement proper grouping logic
        exact_product = example  # Placeholder
        sub_irr_product = example  # Placeholder

        generated_features = generate_features_from_products(
            query, exact_product, sub_irr_product, query_features
        )
        all_features = query_features + generated_features

        # Determine feature applicability for products
        exact_features = determine_feature_applicability(exact_product, all_features)
        sub_irr_features = determine_feature_applicability(
            sub_irr_product, all_features
        )

        # Step 3: Boolean Query Generation
        boolean_expression = generate_random_boolean_expression(all_features)
        natural_query = boolean_expression_to_natural_language(
            boolean_expression, all_features
        )

        # Step 4: Product Selection and Distance Calculation
        # Check if exact product satisfies the boolean query
        if evaluate_boolean_expression(boolean_expression, exact_features):
            positive_product = exact_product

            # Check if substitute/irrelevant doesn't satisfy (making it hard negative)
            if not evaluate_boolean_expression(boolean_expression, sub_irr_features):
                hard_negative_product = sub_irr_product

                # Select easy negative
                easy_negative_product = select_easy_negative(train_ds, positive_product)

                # Calculate edit distance
                edit_distance = calculate_edit_distance(
                    boolean_expression, sub_irr_features
                )

                # Step 5: Dataset Assembly - Create example record
                example_record = create_example_record(
                    natural_query,
                    boolean_expression,
                    positive_product,
                    hard_negative_product,
                    easy_negative_product,
                    all_features,
                    edit_distance,
                )

                final_examples.append(example_record)

    # Add train/test split
    logger.info(f"Generated {len(final_examples)} valid examples")
    final_dataset = add_train_test_split(final_examples)
    logger.info("Train/test split added to dataset")

    # Step 6: Output Generation
    logger.info("Saving dataset and generating report...")
    save_dataset_jsonl(final_dataset, "logical_recall_dataset.jsonl")
    logger.info("Dataset saved to logical_recall_dataset.jsonl")

    create_report(final_dataset, "dataset_report.md")
    logger.info("Report saved to dataset_report.md")

    # Step 7: Cleanup and Logging
    dataset_size = len(final_dataset) if final_dataset else 0
    logger.info(f"Dataset creation completed. Generated {dataset_size} examples.")
    # TODO: Clean up temporary files and resources

    return final_dataset


if __name__ == "__main__":
    main()
