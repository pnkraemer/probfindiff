"""Global collocation with Gaussian processes."""

import jax.numpy as jnp


def unsymmetric(*, K, LK0, LLK):
    """Unsymmetric collocation."""
    weights = jnp.linalg.solve(K, LK0.T).T
    unc_base = LLK - weights @ LK0.T
    return weights, unc_base


def symmetric(*, K, LK1, LLK):
    """Unsymmetric collocation."""
    weights = jnp.linalg.solve(LLK, LK1.T).T
    unc_base = K - weights @ LK1.T
    return weights, unc_base
