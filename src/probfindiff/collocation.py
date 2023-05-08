"""Finite differences and collocation with Gaussian processes."""

import jax.numpy as jnp

from probfindiff.backend.typing import Array, ArrayLike, Callable, Tuple


def non_uniform_nd(
    *,
    x: ArrayLike,
    xs: ArrayLike,
    ks: Tuple[Callable, Callable, Callable],
    noise_variance: float,
) -> Tuple[Array, Array]:
    r"""Finite difference coefficients for non-uniform data in multiple dimensions.

    Parameters
    ----------
    x
        Where to compute the finite difference approximation. Shape ``(d,)``.
    ks
        Triple of kernel functions (:math:`\tilde k`, :math:`\tilde L k`, :math:`\tilde L L^*k`)
    xs
        Neighbourhood. Shape ``(N, d)``.
    noise_variance
        Variance of the observation noise.

    Returns
    -------
    :
        Weights and base-uncertainty. Shapes ``(n,n)``, ``(n,)``.
    """
    x = jnp.asarray(x)
    xs = jnp.asarray(xs)

    K, LK, LLK = prepare_gram(ks, x, xs)
    weights, unc_base = unsymmetric(K=K, LK0=LK, LLK=LLK, noise_variance=noise_variance)
    return weights, unc_base


def prepare_gram(
    ks: Tuple[Callable, Callable, Callable],
    x: ArrayLike,
    xs: ArrayLike,
) -> Tuple[Array, Array, Array]:
    r"""Prepare the Gram matrices that are used for collocation approaches.

    Parameters
    ----------
    ks
        Triple of kernel functions (:math:`\tilde k`, :math:`\tilde L k`, :math:`\tilde L L^*k`)
    x
        Where to compute the finite difference approximation. Shape ``(d,)``.
    xs
        Neighbourhood. Shape ``(N, d)``.

    Returns
    -------
    :
        Triple of kernel Gram matrices (:math:`K`, :math:`LK`, :math:`L L^*K`) with shapes ``(n,n)``, ``(n,)``, ``()``.
    """
    x = jnp.asarray(x)
    xs = jnp.asarray(xs)

    k, lk, llk = ks
    n = xs.shape[0]
    K = k(xs, xs.T).reshape((n, n))
    LK = lk(x[None, :], xs.T).reshape((n,))
    LLK = llk(x[None, :], x[None, :].T).reshape(())
    return K, LK, LLK


def unsymmetric(
    *,
    K: ArrayLike,
    LK0: ArrayLike,
    LLK: ArrayLike,
    noise_variance: float,
) -> Tuple[Array, Array]:
    r"""Unsymmetric collocation.

    Parameters
    ----------
    K
        Gram matrix associated with :math:`k`. Shape ``(n,n)``.
    LK0
        Gram matrix associated with :math:`L k`. Shape ``(n,)``.
    LLK
        Gram matrix associated with :math:`L L^* k`. Shape ``()``.
    noise_variance
        Variance of the observation noise.

    Returns
    -------
    :
        Weights and base-uncertainty. Shapes ``(n,n)``, ``(n,)``.
    """
    K = jnp.asarray(K)
    LK0 = jnp.asarray(LK0)
    LLK = jnp.asarray(LLK)

    noise_matrix = jnp.broadcast_to(
        noise_variance * jnp.eye(K.shape[-1]), shape=K.shape
    )
    LKt = _transpose(LK0)
    weights_t = jnp.linalg.solve(K + noise_matrix, LKt)
    weights = _transpose(weights_t)
    unc_base = LLK - weights @ LKt
    return weights, unc_base


def _transpose(LK0: ArrayLike) -> Array:
    LK0 = jnp.asarray(LK0)

    if LK0.ndim > 1:
        LKt: Array = jnp.swapaxes(LK0, -2, -1)
    else:
        LKt = LK0
    return LKt


def symmetric(
    *, K: ArrayLike, LK1: ArrayLike, LLK: ArrayLike, noise_variance: float
) -> Tuple[Array, Array]:
    r"""Symmetric collocation.

    Parameters
    ----------
    K
        Gram matrix associated with :math:`k`. Shape ``()``.
    LK1
        Gram matrix associated with :math:`L^* k`. Shape ``(n,)``.
    LLK
        Gram matrix associated with :math:`L L^* k`. Shape ``(n,n)``.
    noise_variance
        Variance of the observation noise.

    Returns
    -------
    :
        Weights and base-uncertainty. Shapes ``(n,n)``, ``(n,)``.
    """
    K = jnp.asarray(K)
    LK1 = jnp.asarray(LK1)
    LLK = jnp.asarray(LLK)

    weights = jnp.linalg.solve(LLK + noise_variance * jnp.eye(*LLK.shape), LK1.T).T
    unc_base = K - weights @ LK1.T
    return weights, unc_base
