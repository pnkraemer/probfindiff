"""Finite differences and collocation with Gaussian processes."""

import functools
from typing import Any, Tuple

import jax
import jax.numpy as jnp

from pnfindiff.typing import ArrayLike, KernelFunctionLike


@functools.partial(jax.jit, static_argnames=("ks",))
def non_uniform_nd(
    *,
    x: ArrayLike,
    xs: ArrayLike,
    ks: Tuple[KernelFunctionLike, KernelFunctionLike, KernelFunctionLike]
) -> Tuple[Any, Any]:
    """Finite difference coefficients for non-uniform data in multiple dimensions."""

    K, LK, LLK = prepare_gram(ks, x, xs)
    return unsymmetric(K=K, LK0=LK, LLK=LLK)


def prepare_gram(
    ks: Tuple[KernelFunctionLike, KernelFunctionLike, KernelFunctionLike],
    x: ArrayLike,
    xs: ArrayLike,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Prepare the Gram matrices that are used for collocation approaches."""
    k, lk, llk = ks
    n = xs.shape[0]
    K = k(xs, xs.T).reshape((n, n))
    LK = lk(x[None, :], xs.T).reshape((n,))
    LLK = llk(x[None, :], x[None, :].T).reshape(())
    return K, LK, LLK


@jax.jit
def unsymmetric(
    *, K: ArrayLike, LK0: ArrayLike, LLK: ArrayLike
) -> Tuple[ArrayLike, ArrayLike]:
    """Unsymmetric collocation.

    Parameters
    ----------
    K
        Gram matrix associated with :math:`k`. ``Shape (n,n)``.
    LK0
        Gram matrix associated with :math:`L k`. ``Shape (n,)``.
    LLK
        Gram matrix associated with :math:`L L^* k`. ``Shape ()``.
    """
    weights = jnp.linalg.solve(K, LK0.T).T
    unc_base = LLK - weights @ LK0.T
    return weights, unc_base


@jax.jit
def symmetric(
    *, K: ArrayLike, LK1: ArrayLike, LLK: ArrayLike
) -> Tuple[ArrayLike, ArrayLike]:
    """Symmetric collocation.

    Parameters
    ----------
    K
        Gram matrix associated with :math:`k`. ``Shape ()``.
    LK1
        Gram matrix associated with :math:`L^* k`. ``Shape (n,)``.
    LLK
        Gram matrix associated with :math:`L L^* k`. ``Shape (n,n)``.

    Returns
    -------
    :
        Weights and base-uncertainty.
    """
    weights = jnp.linalg.solve(LLK, LK1.T).T
    unc_base = K - weights @ LK1.T
    return weights, unc_base
