"""Kernel function functionality."""

from typing import Tuple

import jax

from probfindiff.typing import DifferentialOperatorLike, KernelFunctionLike


def differentiate(
    k: KernelFunctionLike, *, L: DifferentialOperatorLike
) -> Tuple[KernelFunctionLike, KernelFunctionLike, KernelFunctionLike]:
    """Differentiate (and batch) a kernel function.

    Parameters
    ----------
    k
        Kernel function to be differentiated.
    L
        Differential operator to be applied.

    Returns
    -------
    :
        Triple :math:`(k, L k, L L^*k)` of differentiated kernel functions.
    """
    k_batch, _ = batch_gram(k)
    lk_batch, lk = batch_gram(L(k, argnums=0))
    llk_batch, _ = batch_gram(L(lk, argnums=1))
    return jax.jit(k_batch), jax.jit(lk_batch), jax.jit(llk_batch)


def batch_gram(k: KernelFunctionLike) -> Tuple[KernelFunctionLike, KernelFunctionLike]:
    r"""Vectorise a kernel function such that it returns Gram matrices.

    A function :math:`k: R^d \times R^d \rightarrow R` becomes

    .. math:: \tilde{k}: R^{N\times d}\times R^{d\times M} \rightarrow R^{N, M}

    which can be used to assemble Kernel gram matrices.

    Parameters
    ----------
    k
        Kernel function to be batched.

    Returns
    -------
    :
        Tuple :math:`(\tilde k, k)` of the batched kernel function and the original kernel function.
    """
    k_vmapped_x = jax.vmap(k, in_axes=(0, None), out_axes=-1)
    return jax.jit(jax.vmap(k_vmapped_x, in_axes=(None, -1), out_axes=-1)), jax.jit(k)
