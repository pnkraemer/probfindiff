"""Kernel function functionality."""

import jax
import jax.numpy as jnp


def vmap_gram(k):
    r"""Vectorise a kernel function such that it returns Gram matrices.

    A function :math:`k: R^d \times R^d \rightarrow R` becomes

    .. math:: \tilde{k}: R^{N\times d}\times R^{d\times M} \rightarrow R^{N, M}

    which can be used to assemble Kernel gram matrices.
    """
    k_vmapped_x = jax.vmap(k, in_axes=(0, None), out_axes=0)
    return jax.vmap(k_vmapped_x, in_axes=(None, 1), out_axes=1)


def exp_quad():
    """Exponentiated quadratic kernel."""

    @vmap_gram
    def k(x, y):
        return jnp.exp(-(x - y).dot(x - y) / 2.0)

    return k
