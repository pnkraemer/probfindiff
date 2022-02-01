"""Kernel zoo."""


from functools import partial

import jax
import jax.numpy as jnp

from pnfindiff.utils import kernel


def exp_quad():
    """Exponentiated quadratic kernel."""

    @jax.jit
    def k(x, y):
        return jnp.exp(-(x - y).dot(x - y) / 2.0)

    return k


def polynomial(*, order, scale=1.0, bias=1.0):
    """Polynomial kernels."""

    @partial(jax.jit, static_argnames=("s", "b", "o"))
    def k(x, y, s=scale, b=bias, o=order):
        return (s * x.dot(y) + b) ** o

    return k
