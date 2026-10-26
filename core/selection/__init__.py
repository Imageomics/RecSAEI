"""Dimension selection shared by viewers and offline experiments."""

import numpy as np


def rank_scores(scores, *, top_k=20, method):
    values = np.asarray(scores, dtype=float).reshape(-1)
    if top_k < 1:
        raise ValueError("top_k must be positive")
    order = np.argsort(-values, kind="stable")
    return [
        {"dim": int(dim), "score": float(values[dim]), "method": method}
        for dim in order[np.isfinite(values[order])][:top_k]
    ]


def manual_dimensions(dimensions, *, d_sae):
    dims = list(dimensions)
    if not dims or any(type(dim) is not int or not 0 <= dim < d_sae for dim in dims):
        raise ValueError("Provide valid integer dimension IDs")
    if len(set(dims)) != len(dims):
        raise ValueError("Dimension IDs must be unique")
    return [{"dim": dim, "score": None, "method": "manual"} for dim in dims]
