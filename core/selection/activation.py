"""Rank feature activations over images or selected spatial patches."""

import numpy as np
from scipy import sparse

from . import rank_scores


def pool_activations(codes, *, pooling="mean"):
    if codes.ndim != 2 or codes.shape[0] == 0:
        raise ValueError("Expected nonempty [tokens, dimensions] activations")
    if pooling == "mean":
        values = codes.mean(axis=0)
    elif pooling == "max":
        values = codes.max(axis=0)
    else:
        raise ValueError("pooling must be mean or max")
    return np.asarray(values.toarray() if sparse.issparse(values) else values).reshape(
        -1
    )


def select_activation(codes, *, top_k=20, pooling="mean", patches=None):
    if patches is not None:
        if (
            not patches
            or len(set(patches)) != len(patches)
            or any(type(p) is not int or not 0 <= p < codes.shape[0] for p in patches)
        ):
            raise ValueError("Provide unique patch indices within the image grid")
        codes = codes[patches]
    return rank_scores(
        pool_activations(codes, pooling=pooling),
        top_k=top_k,
        method=f"activation_{pooling}",
    )
