from __future__ import annotations

import numpy as np

from sfa.sfa_halfnormal import HalfNormalSFA
from sfa.sfa_truncated import TruncatedNormalSFA

MODEL_ALIASES = {
    "half_normal": "half_normal",
    "halfnormal": "half_normal",
    "hn": "half_normal",
    "truncated_normal": "truncated_normal",
    "truncatednormal": "truncated_normal",
    "tn": "truncated_normal",
}


def normalise_model_type(model_type: str) -> str:
    key = model_type.lower().replace("-", "_")
    if key not in MODEL_ALIASES:
        raise ValueError("model_type must be 'half_normal' or 'truncated_normal'.")
    return MODEL_ALIASES[key]


def make_sfa_model(
    model_type: str,
    y: np.ndarray,
    X: np.ndarray,
    *,
    feature_names: list[str] | None = None,
):
    model = normalise_model_type(model_type)
    if model == "half_normal":
        return HalfNormalSFA(y, X, feature_names=feature_names)
    if model == "truncated_normal":
        return TruncatedNormalSFA(y, X, feature_names=feature_names)
    raise ValueError(f"Unsupported model_type: {model_type}")
