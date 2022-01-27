"""Kernel function functionality."""

import jax


def gram_matrix_function(k):

    k_inner = jax.vmap(k, (0, 0), 0)
    k_outer = jax.vmap(jax.vmap(k, (0, None), 0), (None, 1), 1)

    def k_wrapped(xs, ys):

        # Single element of the Gram matrix:
        # X.shape=(d,), Y.shape=(d,) -> K.shape = ()
        if xs.ndim == ys.ndim <= 1:
            return k(xs, ys)

        # Diagonal of the Gram matrix:
        # X.shape=(N,d), Y.shape=(N,d) -> K.shape = (N,)
        if xs.shape == ys.shape:
            return k_inner(xs, ys)

        # Full Gram matrix:
        # X.shape=[N,d), Y.shape=(d,K) -> K.shape = (N,K)
        return k_outer(xs, ys)

    return k_wrapped
