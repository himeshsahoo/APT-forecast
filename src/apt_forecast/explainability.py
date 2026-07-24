from __future__ import annotations

from typing import Any, Callable

import numpy as np


def rule_based_explanation(feature_weights: list[dict[str, Any]]) -> list[str]:
    if not feature_weights:
        return ["No known malicious indicator was mapped for this prediction."]
    return [f"{item['feature']} contributed {item['weight']} evidence point(s)." for item in feature_weights]


def shap_explanation(model: Callable[[np.ndarray], np.ndarray], background: np.ndarray, sample: np.ndarray) -> Any:
    try:
        import shap
    except ImportError as exc:
        raise RuntimeError("Install shap to enable SHAP explanations: pip install shap") from exc

    explainer = shap.KernelExplainer(model, background)
    return explainer.shap_values(sample)


def lime_explanation(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    training_data: np.ndarray,
    sample: np.ndarray,
    feature_names: list[str],
) -> Any:
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError as exc:
        raise RuntimeError("Install lime to enable LIME explanations: pip install lime") from exc

    explainer = LimeTabularExplainer(training_data, feature_names=feature_names, mode="classification")
    return explainer.explain_instance(sample, predict_fn)
