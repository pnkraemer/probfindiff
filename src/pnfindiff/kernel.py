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

    # Return the input again. This simplifies its usage drastically,
    # because often, e.g. derivatives of kernels are defined inline only.
    # For example, the following code
    #
    #     k = kernel.exp_quad()[1]
    #     lk = L(k, argnums=0)
    #     llk = L(lk, argnums=1)
    #     k = kernel.vmap_gram(k)
    #     lk = kernel.vmap_gram(lk)
    #     llk = kernel.vmap_gram(llk)
    #
    # becomes
    #
    #     k_batch, k = kernel.exp_quad()
    #     lk_batch, lk = kernel.vmap_gram(L(k, argnums=0))
    #     llk_batch, llk = kernel.vmap_gram(L(lk, argnums=1))
    #
    # which is so much more compact.
    return jax.vmap(k_vmapped_x, in_axes=(None, 1), out_axes=1), k


def exp_quad():
    """Exponentiated quadratic kernel."""

    @vmap_gram
    def k(x, y):
        return jnp.exp(-(x - y).dot(x - y) / 2.0)

    return k
