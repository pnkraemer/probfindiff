"""Kernel function functionality."""


def gram_matrix_function(k):
    def k_wrapped(xs, ys):

        # Single element of the Gram matrix:
        # X.shape=(d,), Y.shape=(d,) -> K.shape = ()
        if xs.ndim == ys.ndim <= 1:
            return k(xs, ys)

    return k_wrapped
