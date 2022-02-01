"""Finite differences and collocation with Gaussian processes."""

import functools
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp

KernelFunctionType = Callable[[Any, Any], Any]
"""Kernel function type."""


@functools.partial(jax.jit, static_argnames=("ks",))
def non_uniform_nd(
    *,
    x: Any,
    xs: Any,
    ks: Tuple[KernelFunctionType, KernelFunctionType, KernelFunctionType]
) -> Tuple[Any, Any]:
    """Finite difference coefficients for non-uniform data in multiple dimensions."""

    K, LK, LLK = prepare_gram(ks, x, xs)
    return unsymmetric(K=K, LK0=LK, LLK=LLK)


def prepare_gram(
    ks: Tuple[KernelFunctionType, KernelFunctionType, KernelFunctionType],
    x: Any,
    xs: Any,
) -> Tuple[Any, Any, Any]:
    """Prepare the Gram matrices that are used for collocation approaches."""
    k, lk, llk = ks
    n = xs.shape[0]
    K = k(xs, xs.T).reshape((n, n))
    LK = lk(x[None, :], xs.T).reshape((n,))
    LLK = llk(x[None, :], x[None, :].T).reshape(())
    return K, LK, LLK


@jax.jit
def unsymmetric(*, K: Any, LK0: Any, LLK: Any) -> Tuple[Any, Any]:
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
def symmetric(*, K: Any, LK1: Any, LLK: Any) -> Tuple[Any, Any]:
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
