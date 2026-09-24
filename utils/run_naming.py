"""Shared naming for run directories and wandb runs.

Both train_text.py and train_multimodal.py write checkpoints under models/, so the
directory name has to carry enough of the config to tell two runs apart -- most
importantly the dataset, which used to be missing from the path entirely.
"""

import re


def slugify(value) -> str:
    """Filesystem/wandb-safe token: no slashes, no spaces, no repeated separators."""
    slug = re.sub(r"[^0-9A-Za-z._-]+", "-", str(value)).strip("-")
    return re.sub(r"-{2,}", "-", slug)


def dataset_tag(dataset_path: str) -> str:
    """Short identifier for a processed dataset directory.

    'dataset/processed/feature-distance-dataset_gemini-2.5-flash-lite_None/'
        -> 'feature-distance-dataset_gemini-2.5-flash-lite_None'
    """
    trimmed = str(dataset_path).rstrip("/")
    return slugify(trimmed.split("/")[-1] or trimmed)


def dataset_size_tag(dataset_path: str) -> str:
    """The trailing size token of a dataset name ('None', '10000', ...)."""
    trimmed = str(dataset_path).rstrip("/").replace("_fixed_distance", "")
    return trimmed.split("/")[-1].split("_")[-1]


def build_run_name(config, modality: str, query_kind: str, extra=None) -> str:
    """Stable run name shared by the output directory and the wandb run.

    e.g. text__all-mpnet-base-v2__ours-mse__feature-distance-dataset_gemini-2.5-flash-lite_None__synthetic__easy-15
    """
    parts = [
        slugify(modality),
        slugify(str(config["model_name"]).split("/")[-1]),
        slugify(config["training_style"]),
        dataset_tag(config["dataset"]),
        slugify(query_kind),
    ]
    for key, value in (extra or {}).items():
        if value is not None:
            parts.append(f"{slugify(key)}-{slugify(value)}")
    return "__".join(part for part in parts if part)


def build_output_dir(config, modality: str, query_kind: str, extra=None, root: str = "models") -> str:
    return f"{root}/{build_run_name(config, modality, query_kind, extra)}"
