#!/usr/bin/env python3
"""
Build image-to-image triplets for DeepFashion In-Shop.

Primary source: Hugging Face dataset `Marqo/deepfashion-inshop`.
Fallback source: original DeepFashion In-Shop annotation files.

Raw JSONL output is image-native:
    nl_query, positive_product, hard_neg_product, easy_neg_product

Processed dataset output mirrors main.py:
    original_query, nl_query, positive_example, negative_example,
    negative_example_source, query_distance
"""

import argparse
import csv
import json
import logging
import os
import random
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

from datasets import Dataset as HFDataset, concatenate_datasets, load_dataset
from dotenv import load_dotenv
from jsonschema import ValidationError, validate
from tqdm import tqdm

from utils.retry import get_cost_summary, print_cost_report


DEFAULT_HF_DATASET = "Marqo/deepfashion-inshop"
DEFAULT_MODEL_ID = "gemini-2.5-flash"
N_FEATURES = 5
RANDOM_SEED = 42
# Sentinel, not a real distance: easy negatives are unrelated, so there is no feature
# distance to measure. Training remaps it to `easy_negative_value`. Kept identical to the
# text pipeline so both datasets use one convention.
# A random negative's violation count was never measured; it is absent, not small. Keep it
# null so negative_example_source stays the only marker for an easy negative.
EASY_NEGATIVE_DISTANCE = None
HARD_NEGATIVE_DISTANCE = 1.0

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler("recall_pipeline.log"), logging.StreamHandler()],
    )


def ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def safe_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def derive_product_id(item_id: str) -> str:
    """Strip DeepFashion view suffixes so views of the same SKU group together."""
    value = str(item_id)
    # Examples: MEN_Denim_id_00000080_01_1_front -> MEN_Denim_id_00000080_01
    return re.sub(r"_\d+_(front|side|back|full|flat|additional)$", "", value)


def record_category(record: Dict[str, Any], category_key: str) -> str:
    value = record.get(category_key)
    if value is None or value == "":
        return "unknown"
    return str(value).lower()


def materialize_hf_image(image: Any, image_id: str, output_dir: str) -> str:
    """Persist a HF Image/PIL value to a stable local path for downstream training."""
    os.makedirs(output_dir, exist_ok=True)

    if isinstance(image, str):
        return image

    if isinstance(image, dict):
        if image.get("path") and os.path.exists(image["path"]):
            return image["path"]
        image = image.get("bytes") or image.get("array") or image

    extension = "jpg"
    pil_image = image
    if hasattr(pil_image, "format") and pil_image.format:
        extension = pil_image.format.lower().replace("jpeg", "jpg")

    output_path = os.path.join(output_dir, f"{safe_filename(image_id)}.{extension}")
    if os.path.exists(output_path):
        return output_path

    if not hasattr(pil_image, "save"):
        raise TypeError(f"Unsupported HF image object for {image_id}: {type(image)!r}")

    if getattr(pil_image, "mode", None) not in ("RGB", "L"):
        pil_image = pil_image.convert("RGB")
    pil_image.save(output_path)
    return output_path


def load_hf_deepfashion(
    dataset_name: str,
    split: str,
    image_output_dir: str,
    row_limit: Optional[int],
) -> List[Dict[str, Any]]:
    ds = load_dataset(dataset_name, split=split)
    if row_limit is not None:
        ds = ds.select(range(min(row_limit, len(ds))))

    records: List[Dict[str, Any]] = []
    for row in tqdm(ds, desc="Loading HF DeepFashion rows"):
        item_id = str(row.get("item_ID", row.get("item_id", "")))
        if not item_id:
            continue

        image_path = materialize_hf_image(row["image"], item_id, image_output_dir)
        product_id = derive_product_id(item_id)
        records.append(
            {
                "image_id": item_id,
                "image_path": image_path,
                "item_id": item_id,
                "product_id": product_id,
                "split": "data",
                "category1": str(row.get("category1", "unknown")),
                "category2": str(row.get("category2", "unknown")),
                "category3": str(row.get("category3", "unknown")),
                "color": str(row.get("color", "unknown")),
                "description": str(row.get("description", "")),
                "text": str(row.get("text", "")),
            }
        )

    logger.info("Loaded %d records from %s[%s]", len(records), dataset_name, split)
    return records


def parse_table(path: str, expected_min_columns: int) -> List[List[str]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing required annotation file: {path}")

    rows: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    for line in lines[2:]:
        parts = line.split()
        if len(parts) >= expected_min_columns:
            rows.append(parts)
    return rows


def clothes_type_name(clothes_type: Optional[str]) -> str:
    return {"1": "upper_body", "2": "lower_body", "3": "full_body"}.get(str(clothes_type), "unknown")


def load_original_deepfashion(data_root: str, image_root: Optional[str]) -> List[Dict[str, Any]]:
    eval_path = os.path.join(data_root, "Eval", "list_eval_partition.txt")
    bbox_path = os.path.join(data_root, "Anno", "list_bbox_inshop.txt")
    image_root = image_root or os.path.join(data_root, "Img")

    bbox_by_image: Dict[str, Dict[str, Any]] = {}
    if os.path.exists(bbox_path):
        for parts in parse_table(bbox_path, expected_min_columns=7):
            image_name, clothes_type, pose_type, x1, y1, x2, y2 = parts[:7]
            bbox_by_image[image_name] = {
                "clothes_type": clothes_type,
                "category2": clothes_type_name(clothes_type),
                "pose_type": pose_type,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
            }

    records: List[Dict[str, Any]] = []
    for parts in parse_table(eval_path, expected_min_columns=3):
        image_name, item_id, split = parts[:3]
        meta = bbox_by_image.get(image_name, {})
        records.append(
            {
                "image_id": image_name,
                "image_path": os.path.join(image_root, image_name),
                "item_id": item_id,
                "product_id": item_id,
                "split": split,
                "category1": "unknown",
                "category2": meta.get("category2", "unknown"),
                "category3": "unknown",
                "color": "unknown",
                "description": "",
                "text": "",
                "clothes_type": meta.get("clothes_type", "unknown"),
                "pose_type": meta.get("pose_type", "unknown"),
                "bbox": meta.get("bbox", []),
            }
        )

    logger.info("Loaded %d records from original DeepFashion annotations", len(records))
    return records


def grouped(records: Iterable[Dict[str, Any]], key: str) -> Dict[str, List[Dict[str, Any]]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[str(record[key])].append(record)
    return groups


def choose_negative(
    rng: random.Random,
    records: List[Dict[str, Any]],
    anchor: Dict[str, Any],
    category_key: str,
    prefer_same_category: bool,
    match_color_for_hard_negative: bool,
) -> Optional[Dict[str, Any]]:
    anchor_product = str(anchor["product_id"])
    anchor_category = record_category(anchor, category_key)
    anchor_color = str(anchor.get("color", "unknown")).lower()

    candidates = []
    for row in records:
        if str(row["product_id"]) == anchor_product:
            continue
        same_category = record_category(row, category_key) == anchor_category
        if prefer_same_category != same_category:
            continue
        if match_color_for_hard_negative and prefer_same_category:
            if str(row.get("color", "unknown")).lower() != anchor_color:
                continue
        candidates.append(row)

    if not candidates and match_color_for_hard_negative and prefer_same_category:
        return choose_negative(
            rng,
            records,
            anchor,
            category_key,
            prefer_same_category=True,
            match_color_for_hard_negative=False,
        )

    if not candidates:
        return None
    return rng.choice(candidates)


def build_product_examples(
    records: List[Dict[str, Any]],
    rng: random.Random,
) -> List[Dict[str, Any]]:
    by_product = grouped(records, "product_id")
    positives: List[Dict[str, Any]] = []

    for product_rows in by_product.values():
        # We only need one representative image now because the anchor is nl_query.
        positives.append(rng.choice(product_rows))

    rng.shuffle(positives)
    return positives


def make_raw_example(
    positive: Dict[str, Any],
    hard_negative: Dict[str, Any],
    easy_negative: Dict[str, Any],
    category_key: str,
) -> Dict[str, Any]:
    item = record_category(positive, category_key)
    return {
        "original_query": "",
        "nl_query": "",
        "rephrased_query": "",
        "query_source_image": positive["image_path"],
        "positive_product": positive,
        "hard_neg_product": hard_negative,
        "easy_neg_product": easy_negative,
        "positive_product_id": positive["product_id"],
        "hard_negative_product_id": hard_negative["product_id"],
        "easy_negative_product_id": easy_negative["product_id"],
        "category_key": category_key,
        "item": item,
        "positive_category": item,
        "hard_negative_category": record_category(hard_negative, category_key),
        "easy_negative_category": record_category(easy_negative, category_key),
        "negative_example_source": f"same_{category_key}_different_product",
        "hard_negative_distance_source": "placeholder_for_vlm",
        "query_distance": HARD_NEGATIVE_DISTANCE,
        "easy_negative_query_distance": EASY_NEGATIVE_DISTANCE,
        "selected_pos_features": [],
        "selected_neg_features": [],
        "selected_common_features": [],
        "selected_neither_features": [],
        "full_common_features": [],
        "full_unique_pos_features": [],
        "full_unique_neg_features": [],
        "full_neither_features": [],
    }


def product_prompt_metadata(record: Dict[str, Any]) -> Dict[str, Any]:
    keys = ["product_id", "item_id", "category1", "category2", "category3", "color", "description"]
    return {key: record.get(key, "") for key in keys if record.get(key, "")}


def validate_feature_response(content: str) -> bool:
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

    try:
        parsed_response = json.loads(content)
        validate(instance=parsed_response, schema=schema)
        forbidden_prefixes = ["no ", "not ", "without ", "doesn't have ", "does not have "]
        for key in ["common_features", "unique_features_a", "unique_features_b", "neither_features"]:
            for feature in parsed_response.get(key, []):
                if any(str(feature).lower().startswith(prefix) for prefix in forbidden_prefixes):
                    return False
        return True
    except (json.JSONDecodeError, ValidationError):
        return False


def get_visual_common_and_differentiating_features(
    positive: Dict[str, Any],
    hard_negative: Dict[str, Any],
    model_id: str,
    max_retries: int = 5,
) -> Optional[Tuple[List[str], List[str], List[str], List[str]]]:
    if not model_id.startswith("gemini-"):
        raise ValueError("Image VLM scoring currently supports Gemini models only.")

    similar_prompt = """
### Product A:

The first image is Product A. Metadata: {positive_product}

### Product B:

The second image is Product B. Metadata: {hard_negative_product}

List up to {N_COMMON_FEATURES} visual features common to both products ("common_features"), up to {N_FEATURES} visual features unique to product A ("unique_features_a"), up to {N_FEATURES} visual features unique to product B ("unique_features_b"), and up to {N_FEATURES} visual features that do not apply to either product ("neither_features").

When generating common_features, ensure they are visible in both products in some way. If there are fewer than {N_COMMON_FEATURES} common features, generate as many as possible. Do not assume anything not visible in the images or stated in metadata.

When generating neither_features, ensure they are opposite or mutually exclusive with visual features in one or both products. For example, if product A is red and product B is orange, then "blue" could be in neither_features. Must not generate negated features, such as "not red", "no sleeves", "without pattern", or "doesn't have x". Dont use "not" or "without". Use your imagination and create diverse neither_features. All features should be objective and no more than 5 words.

Return ONLY JSON: {{"common_features": ["feature1", "feature2"], "unique_features_a": ["feature1", "feature2"], "unique_features_b": ["feature1", "feature2"], "neither_features": ["feature1", "feature2"]}}
""".strip()

    prompt = similar_prompt.format(
        positive_product=product_prompt_metadata(positive),
        hard_negative_product=product_prompt_metadata(hard_negative),
        N_FEATURES=N_FEATURES,
        N_COMMON_FEATURES=N_FEATURES,
    )

    # Imported at the call site: the VLM helper is only needed when generating hard-negative
    # distances, so reusing this module's dataset builders does not require it to exist.
    from utils.retry import retry_vlm_with_fallback

    response = retry_vlm_with_fallback(
        prompt=prompt,
        image_paths=[positive["image_path"], hard_negative["image_path"]],
        validation_func=validate_feature_response,
        max_retries=max_retries,
        fallback_value=None,
        model_id=model_id,
    )
    if response is None:
        return None

    parsed = json.loads(response)
    return (
        parsed["common_features"],
        parsed["unique_features_a"],
        parsed["unique_features_b"],
        parsed["neither_features"],
    )


def generate_visual_distance_example(
    item: str,
    common_features: List[str],
    unique_pos_features: List[str],
    unique_neg_features: List[str],
    neither_features: List[str],
    max_distance: int,
    rng: random.Random,
) -> Dict[str, Any]:
    query_distance = rng.randint(1, max_distance)
    n_pos_features = min(rng.randint(0, query_distance), len(unique_pos_features))
    n_neg_features = min(query_distance - n_pos_features, len(unique_neg_features))
    n_common_features = min(rng.randint(0, query_distance), len(common_features))
    n_neither_features = min(rng.randint(0, query_distance), len(neither_features))

    if not (n_pos_features or n_common_features):
        n_pos_features = 1 if unique_pos_features else 0
        n_common_features = 1 if not n_pos_features and common_features else 0

    if not (n_neg_features or n_neither_features):
        n_neg_features = 1 if unique_neg_features else 0
        n_neither_features = 1 if not n_neg_features and neither_features else 0

    selected_pos_features = rng.sample(unique_pos_features, n_pos_features)
    selected_neg_features = rng.sample(unique_neg_features, n_neg_features)
    selected_common_features = rng.sample(common_features, n_common_features)
    selected_neither_features = rng.sample(neither_features, n_neither_features)
    nl_query = (
        f'I am looking for: "{item}" that has: '
        f"{', '.join(selected_pos_features + selected_common_features)}; "
        f"and does not have: {', '.join(selected_neg_features + selected_neither_features)}"
    )

    return {
        "item": item,
        "nl_query": nl_query,
        "selected_pos_features": selected_pos_features,
        "selected_neg_features": selected_neg_features,
        "selected_common_features": selected_common_features,
        "selected_neither_features": selected_neither_features,
        "query_distance": float(len(selected_pos_features) + len(selected_neg_features)),
        "full_common_features": common_features,
        "full_unique_pos_features": unique_pos_features,
        "full_unique_neg_features": unique_neg_features,
        "full_neither_features": neither_features,
    }


def apply_vlm_distances(
    examples: List[Dict[str, Any]],
    model_id: str,
    max_distance: int,
    seed: int,
    keep_failures: bool,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    scored_examples: List[Dict[str, Any]] = []

    for example in tqdm(examples, desc="Scoring hard negatives with VLM"):
        features = get_visual_common_and_differentiating_features(
            example["positive_product"],
            example["hard_neg_product"],
            model_id=model_id,
        )
        if features is None:
            if keep_failures:
                scored_examples.append(example)
            continue

        common_features, unique_pos_features, unique_neg_features, neither_features = features
        generated = generate_visual_distance_example(
            item=example["item"],
            common_features=common_features,
            unique_pos_features=unique_pos_features,
            unique_neg_features=unique_neg_features,
            neither_features=neither_features,
            max_distance=max_distance,
            rng=rng,
        )
        example.update(generated)
        example["hard_negative_distance_source"] = model_id
        scored_examples.append(example)

    logger.info("VLM-scored %d of %d examples", len(scored_examples), len(examples))
    return scored_examples


def build_triplets(
    records: List[Dict[str, Any]],
    max_examples: Optional[int],
    category_key: str,
    match_color_for_hard_negative: bool,
    seed: int,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    positives = build_product_examples(records, rng)

    examples: List[Dict[str, Any]] = []
    for positive in tqdm(positives, desc="Constructing product triplets"):
        hard_negative = choose_negative(
            rng,
            records,
            positive,
            category_key,
            prefer_same_category=True,
            match_color_for_hard_negative=match_color_for_hard_negative,
        )
        easy_negative = choose_negative(
            rng,
            records,
            positive,
            category_key,
            prefer_same_category=False,
            match_color_for_hard_negative=False,
        )
        if hard_negative is None or easy_negative is None:
            continue
        examples.append(make_raw_example(positive, hard_negative, easy_negative, category_key))
        if max_examples is not None and len(examples) >= max_examples:
            break

    logger.info("Constructed %d image triplets", len(examples))
    return examples


def save_jsonl(examples: List[Dict[str, Any]], output_file: str) -> None:
    ensure_parent_dir(output_file)
    with open(output_file, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    logger.info("Saved raw JSONL dataset to %s", output_file)


def markdown_image(path: str, summary_file: str, alt: str) -> str:
    if not path:
        return ""
    if os.path.exists(path):
        rel_path = os.path.relpath(path, start=os.path.dirname(summary_file) or ".")
    else:
        rel_path = path
    return f"![{alt}]({rel_path})"


def metadata_lines(record: Dict[str, Any]) -> List[str]:
    keys = [
        "image_id",
        "product_id",
        "item_id",
        "category1",
        "category2",
        "category3",
        "color",
        "split",
    ]
    return [f"- **{key}:** {record.get(key, '')}" for key in keys if record.get(key, "") != ""]


def write_image_block(f, title: str, image_path: str, record: Dict[str, Any], summary_file: str) -> None:
    f.write(f"### {title}\n\n")
    f.write(f"**Path:** `{image_path}`\n\n")
    f.write(markdown_image(image_path, summary_file, title.lower().replace(" ", "-")))
    f.write("\n\n")
    for line in metadata_lines(record):
        f.write(f"{line}\n")
    if record.get("description"):
        f.write(f"- **description:** {record['description']}\n")
    f.write("\n")


def generate_summary_md(examples: List[Dict[str, Any]], summary_file: str, n_examples: int) -> None:
    ensure_parent_dir(summary_file)
    examples_to_write = examples[:n_examples]

    with open(summary_file, "w", encoding="utf-8") as f:
        f.write("# DeepFashion Image Triplet Summary\n\n")
        f.write(f"Showing {len(examples_to_write)} of {len(examples)} raw triplets.\n\n")
        f.write(
            "Each example has one generated natural-language query, one positive product, "
            "one same-category hard negative product, and one different-category easy negative product.\n\n"
        )

        for i, example in enumerate(examples_to_write, start=1):
            f.write(f"## Example {i}\n\n")
            f.write(f"- **Generated Query:** {example.get('nl_query', '')}\n")
            f.write(f"- **positive_product_id:** {example['positive_product_id']}\n")
            f.write(f"- **hard_negative_product_id:** {example['hard_negative_product_id']}\n")
            f.write(f"- **easy_negative_product_id:** {example['easy_negative_product_id']}\n")
            f.write(f"- **category_key:** {example['category_key']}\n")
            f.write(f"- **item:** {example['item']}\n")
            f.write(f"- **positive_category:** {example['positive_category']}\n")
            f.write(f"- **hard_negative_category:** {example['hard_negative_category']}\n")
            f.write(f"- **easy_negative_category:** {example['easy_negative_category']}\n")
            f.write(f"- **hard_negative_distance_source:** {example['hard_negative_distance_source']}\n")
            f.write(f"- **query_distance:** {example['query_distance']}\n")
            f.write(f"- **easy_negative_query_distance:** {example['easy_negative_query_distance']}\n\n")
            f.write(f"- **selected_pos_features:** {', '.join(example.get('selected_pos_features', []))}\n")
            f.write(f"- **selected_neg_features:** {', '.join(example.get('selected_neg_features', []))}\n")
            f.write(f"- **selected_common_features:** {', '.join(example.get('selected_common_features', []))}\n")
            f.write(f"- **selected_neither_features:** {', '.join(example.get('selected_neither_features', []))}\n")
            f.write(f"- **full_common_features:** {', '.join(example.get('full_common_features', []))}\n")
            f.write(f"- **full_unique_pos_features:** {', '.join(example.get('full_unique_pos_features', []))}\n")
            f.write(f"- **full_unique_neg_features:** {', '.join(example.get('full_unique_neg_features', []))}\n")
            f.write(f"- **full_neither_features:** {', '.join(example.get('full_neither_features', []))}\n\n")

            write_image_block(f, "Positive Product", example["positive_product"]["image_path"], example["positive_product"], summary_file)
            write_image_block(
                f,
                "Hard Negative Product",
                example["hard_neg_product"]["image_path"],
                example["hard_neg_product"],
                summary_file,
            )
            write_image_block(
                f,
                "Easy Negative Product",
                example["easy_neg_product"]["image_path"],
                example["easy_neg_product"],
                summary_file,
            )
            f.write("---\n\n")

    logger.info("Saved markdown summary to %s", summary_file)


def save_processed_dataset(examples: List[Dict[str, Any]], output_dir: str) -> None:
    raw_dataset = HFDataset.from_list(examples)

    def add_hard_examples(example: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "original_query": example.get("original_query", ""),
            "nl_query": example["nl_query"],
            "rephrased_query": example.get("rephrased_query", ""),
            "positive_example": example["positive_product"]["image_path"],
            "negative_example": example["hard_neg_product"]["image_path"],
            "positive_product_id": example["positive_product_id"],
            "negative_product_id": example["hard_negative_product_id"],
            "item": example["item"],
            "positive_category": example["positive_category"],
            "negative_category": example["hard_negative_category"],
            "negative_example_source": example["negative_example_source"],
            "query_distance": float(example["query_distance"]),
            "distance_source": example["hard_negative_distance_source"],
            "selected_pos_features": example.get("selected_pos_features", []),
            "selected_neg_features": example.get("selected_neg_features", []),
            "selected_common_features": example.get("selected_common_features", []),
            "selected_neither_features": example.get("selected_neither_features", []),
            "full_common_features": example.get("full_common_features", []),
            "full_unique_pos_features": example.get("full_unique_pos_features", []),
            "full_unique_neg_features": example.get("full_unique_neg_features", []),
            "full_neither_features": example.get("full_neither_features", []),
        }

    def add_easy_examples(example: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "original_query": example.get("original_query", ""),
            "nl_query": example["nl_query"],
            "rephrased_query": example.get("rephrased_query", ""),
            "positive_example": example["positive_product"]["image_path"],
            "negative_example": example["easy_neg_product"]["image_path"],
            "positive_product_id": example["positive_product_id"],
            "negative_product_id": example["easy_negative_product_id"],
            "item": example["item"],
            "positive_category": example["positive_category"],
            "negative_category": example["easy_negative_category"],
            "negative_example_source": "random",
            "query_distance": (None if example["easy_negative_query_distance"] is None
                               else float(example["easy_negative_query_distance"])),
            "distance_source": "easy_negative_constant",
            "selected_pos_features": example.get("selected_pos_features", []),
            "selected_neg_features": example.get("selected_neg_features", []),
            "selected_common_features": example.get("selected_common_features", []),
            "selected_neither_features": example.get("selected_neither_features", []),
            "full_common_features": example.get("full_common_features", []),
            "full_unique_pos_features": example.get("full_unique_pos_features", []),
            "full_unique_neg_features": example.get("full_unique_neg_features", []),
            "full_neither_features": example.get("full_neither_features", []),
        }

    hard_dataset = raw_dataset.map(add_hard_examples, desc="Processing hard image negatives", remove_columns=raw_dataset.column_names)
    easy_dataset = raw_dataset.map(add_easy_examples, desc="Processing easy image negatives", remove_columns=raw_dataset.column_names)
    processed_dataset = concatenate_datasets([hard_dataset, easy_dataset])

    os.makedirs(os.path.dirname(output_dir), exist_ok=True)
    processed_dataset.save_to_disk(output_dir)
    logger.info("Saved processed image dataset to %s", output_dir)


def append_cost_record(
    costs_file: str,
    args: argparse.Namespace,
    output_file: str,
    output_dir: str,
    summary_file: str,
    raw_examples: int,
    processed_rows: int,
) -> None:
    ensure_parent_dir(costs_file)
    summary = get_cost_summary()
    fieldnames = [
        "timestamp_utc",
        "source",
        "hf_dataset",
        "hf_split",
        "model_id",
        "no_vlm",
        "n_examples",
        "row_limit",
        "raw_examples",
        "processed_rows",
        "total_api_calls",
        "total_tokens_used",
        "total_cost",
        "avg_cost_per_call",
        "avg_tokens_per_call",
        "output_file",
        "processed_output_dir",
        "summary_file",
    ]
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": args.source,
        "hf_dataset": args.hf_dataset,
        "hf_split": args.hf_split,
        "model_id": args.model_id,
        "no_vlm": args.no_vlm,
        "n_examples": args.n_examples,
        "row_limit": args.row_limit,
        "raw_examples": raw_examples,
        "processed_rows": processed_rows,
        "total_api_calls": summary["total_api_calls"],
        "total_tokens_used": summary["total_tokens_used"],
        "total_cost": f"{summary['total_cost']:.8f}",
        "avg_cost_per_call": f"{summary['avg_cost_per_call']:.8f}",
        "avg_tokens_per_call": f"{summary['avg_tokens_per_call']:.2f}",
        "output_file": output_file,
        "processed_output_dir": output_dir,
        "summary_file": "" if args.no_summary else summary_file,
    }
    write_header = not os.path.exists(costs_file) or os.path.getsize(costs_file) == 0
    with open(costs_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    logger.info("Appended API cost record to %s", costs_file)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build DeepFashion In-Shop image triplets")
    parser.add_argument("--source", choices=["hf", "original"], default="hf")
    parser.add_argument("--hf-dataset", default=DEFAULT_HF_DATASET)
    parser.add_argument("--hf-split", default="data")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--max-distance", type=int, default=10)
    parser.add_argument("--no-vlm", action="store_true", help="Skip VLM hard-negative distance generation")
    parser.add_argument(
        "--keep-vlm-failures",
        action="store_true",
        help="Keep placeholder-distance examples when VLM scoring fails",
    )
    parser.add_argument("--data-root", default=None, help="Original DeepFashion In-Shop root, required for --source original")
    parser.add_argument("--image-root", default=None, help="Original image root. Defaults to DATA_ROOT/Img")
    parser.add_argument("--materialize-images-dir", default="dataset/images/deepfashion-inshop")
    parser.add_argument("--row-limit", type=int, default=None, help="Optional HF row limit before triplet construction")
    parser.add_argument("--category-key", default="category2", choices=["category1", "category2", "category3", "clothes_type"])
    parser.add_argument("--match-color-for-hard-negative", action="store_true")
    parser.add_argument("--n-examples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--processed-output-dir", default=None)
    parser.add_argument("--summary-file", default=None)
    parser.add_argument("--summary-examples", type=int, default=10)
    parser.add_argument("--no-summary", action="store_true")
    parser.add_argument("--costs-file", default="dataset/api_costs.csv")
    return parser.parse_args()


def main() -> List[Dict[str, Any]]:
    args = parse_arguments()
    setup_logging()
    random.seed(args.seed)

    if args.source == "hf":
        records = load_hf_deepfashion(
            dataset_name=args.hf_dataset,
            split=args.hf_split,
            image_output_dir=args.materialize_images_dir,
            row_limit=args.row_limit,
        )
    else:
        if not args.data_root:
            raise ValueError("--data-root is required when --source original")
        records = load_original_deepfashion(args.data_root, args.image_root)

    examples = build_triplets(
        records=records,
        max_examples=args.n_examples,
        category_key=args.category_key,
        match_color_for_hard_negative=args.match_color_for_hard_negative,
        seed=args.seed,
    )

    if not args.no_vlm:
        examples = apply_vlm_distances(
            examples=examples,
            model_id=args.model_id,
            max_distance=args.max_distance,
            seed=args.seed,
            keep_failures=args.keep_vlm_failures,
        )

    size_label = args.n_examples if args.n_examples is not None else len(examples)
    output_file = args.output_file or f"dataset/deepfashion-inshop-image-triplets_{args.source}_{size_label}.jsonl"
    output_dir = args.processed_output_dir or f"dataset/processed/deepfashion-inshop-image-triplets_{args.source}_{size_label}"
    summary_file = args.summary_file or f"dataset/deepfashion-inshop-image-triplets_{args.source}_{size_label}_summary.md"

    save_jsonl(examples, output_file)
    if not args.no_summary:
        generate_summary_md(examples, summary_file, args.summary_examples)
    save_processed_dataset(examples, output_dir)

    processed_rows = len(examples) * 2
    append_cost_record(
        costs_file=args.costs_file,
        args=args,
        output_file=output_file,
        output_dir=output_dir,
        summary_file=summary_file,
        raw_examples=len(examples),
        processed_rows=processed_rows,
    )

    logger.info("Done. Raw triplets: %d; processed rows: %d", len(examples), processed_rows)
    print_cost_report()
    return examples


if __name__ == "__main__":
    main()
