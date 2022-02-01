"""Kernel function functionality."""

from typing import Tuple

import jax

from pnfindiff.typing import DifferentialOperatorLike, KernelFunctionLike


def differentiate(
    k: KernelFunctionLike, *, L: DifferentialOperatorLike
) -> Tuple[KernelFunctionLike, KernelFunctionLike, KernelFunctionLike]:
    """Differentiate (and batch) a kernel function."""
    k_batch, _ = batch_gram(k)
    lk_batch, lk = batch_gram(L(k, argnums=0))
    llk_batch, _ = batch_gram(L(lk, argnums=1))
    return jax.jit(k_batch), jax.jit(lk_batch), jax.jit(llk_batch)


def batch_gram(k: KernelFunctionLike) -> Tuple[KernelFunctionLike, KernelFunctionLike]:
    r"""Vectorise a kernel function such that it returns Gram matrices.

    A function :math:`k: R^d \times R^d \rightarrow R` becomes

    .. math:: \tilde{k}: R^{N\times d}\times R^{d\times M} \rightarrow R^{N, M}

    which can be used to assemble Kernel gram matrices.
    """
    k_vmapped_x = jax.vmap(k, in_axes=(0, None), out_axes=0)

    # Return the input again. This simplifies its usage drastically,
    # because often, e.g., derivatives of kernels are defined inline only.
    # For example, the following code
    #
    #     k = kernel_zoo.exponentiated_quadratic[1]
    #     lk = L(k, argnums=0)
    #     llk = L(lk, argnums=1)
    #     k = kernel.batch_gram(k)
    #     lk = kernel.batch_gram(lk)
    #     llk = kernel.batch_gram(llk)
    #
    # becomes
    #
    #     k_batch, k = kernel_zoo.exponentiated_quadratic
    #     lk_batch, lk = kernel.batch_gram(L(k, argnums=0))
    #     llk_batch, llk = kernel.batch_gram(L(lk, argnums=1))
    #
    # which is so much more compact.
    return jax.jit(jax.vmap(k_vmapped_x, in_axes=(None, 1), out_axes=1)), jax.jit(k)
