"""Training-set class-versus-rest activation separation, without a classifier."""

import numpy as np
from scipy import sparse

from . import rank_scores


def fit_class_scores(image_codes, labels):
    """Use image-level max-pooled activations, matching the archived viewer."""
    codes = sparse.csr_array(image_codes, dtype=np.float64)
    labels = np.asarray(labels)
    if codes.shape[0] != len(labels) or len(np.unique(labels)) < 2:
        raise ValueError(
            "Class selection requires aligned labels and at least two classes"
        )
    if not np.isfinite(codes.data).all():
        raise ValueError("Activations must be finite")
    total = np.asarray(codes.sum(axis=0)).reshape(-1)
    total_sq = np.asarray(codes.power(2).sum(axis=0)).reshape(-1)
    result = {}
    for label in np.unique(labels):
        positive = codes[labels == label]
        n_pos, n_neg = positive.shape[0], codes.shape[0] - positive.shape[0]
        pos_sum = np.asarray(positive.sum(axis=0)).reshape(-1)
        pos_sq = np.asarray(positive.power(2).sum(axis=0)).reshape(-1)
        pos_mean, neg_mean = pos_sum / n_pos, (total - pos_sum) / n_neg
        pos_var = np.maximum(pos_sq / n_pos - pos_mean**2, 0)
        neg_var = np.maximum((total_sq - pos_sq) / n_neg - neg_mean**2, 0)
        gap = pos_mean - neg_mean
        result[str(label)] = np.where(
            gap > 0, gap / np.sqrt(pos_var + neg_var + 1e-8), -np.inf
        )
    return result


def select_class(scores, class_name, *, top_k=20):
    if str(class_name) not in scores:
        raise ValueError(f"No training statistics for class {class_name!r}")
    return rank_scores(scores[str(class_name)], top_k=top_k, method="class_specific")
