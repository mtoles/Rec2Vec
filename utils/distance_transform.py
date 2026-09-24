"""Shaping applied to normalized feature distances before they become MSE labels.

Shared by the text and multimodal trainers so both accept the same
--distance-transform values.
"""

import math
from enum import Enum


class DistanceTransform(Enum):
    LINEAR = "linear"
    QUADRATIC = "quadratic"
    SQRT = "sqrt"
    LOG = "log"
    EXP = "exp"


def transform_normalized_distance(value: float, transform_name: str, transform_alpha: float = 5.0) -> float:
    value = max(0.0, min(1.0, float(value)))

    if transform_name == DistanceTransform.LINEAR.value:
        return value
    if transform_name == DistanceTransform.QUADRATIC.value:
        return value**2
    if transform_name == DistanceTransform.SQRT.value:
        return math.sqrt(value)
    if transform_name == DistanceTransform.LOG.value:
        if transform_alpha <= 0:
            raise ValueError("distance_transform_alpha must be > 0 for log transform")
        return math.log1p(transform_alpha * value) / math.log1p(transform_alpha)
    if transform_name == DistanceTransform.EXP.value:
        if transform_alpha <= 0:
            raise ValueError("distance_transform_alpha must be > 0 for exp transform")
        return math.expm1(transform_alpha * value) / math.expm1(transform_alpha)

    raise ValueError(f"Invalid distance transform: {transform_name}")
