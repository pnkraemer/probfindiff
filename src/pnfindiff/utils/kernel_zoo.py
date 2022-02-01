"""Kernel zoo."""


from functools import partial

import jax
import jax.numpy as jnp

from pnfindiff.utils import kernel


@jax.jit
def exp_quad(x, y):
    """Exponentiated quadratic kernel."""
    return jnp.exp(-(x - y).dot(x - y) / 2.0)


@partial(jax.jit, static_argnames=("scale", "order", "bias"))
def polynomial(x, y, *, order=2, scale=1.0, bias=1.0):
    """Polynomial kernels."""
    return (scale * x.dot(y) + bias) ** order
