"""Global collocation with Gaussian processes."""

import functools

import jax
import jax.numpy as jnp


@functools.partial(jax.jit, static_argnames=("ks",))
def non_uniform_nd(*, x, xs, ks):
    """Finite difference coefficients for non-uniform data in multiple dimensions."""

    K, LK, LLK = prepare_gram(ks, x, xs)
    return unsymmetric(K=K, LK0=LK, LLK=LLK)


def prepare_gram(ks, x, xs):
    """Prepare the Gram matrices that are used for collocation approaches."""
    k, lk, llk = ks
    n = xs.shape[0]
    K = k(xs, xs.T).reshape((n, n))
    LK = lk(x[None, :], xs.T).reshape((n,))
    LLK = llk(x[None, :], x[None, :].T).reshape(())
    return K, LK, LLK


@jax.jit
def unsymmetric(*, K, LK0, LLK):
    """Unsymmetric collocation."""
    weights = jnp.linalg.solve(K, LK0.T).T
    unc_base = LLK - weights @ LK0.T
    return weights, unc_base


@jax.jit
def symmetric(*, K, LK1, LLK):
    """Unsymmetric collocation."""
    weights = jnp.linalg.solve(LLK, LK1.T).T
    unc_base = K - weights @ LK1.T
    return weights, unc_base
