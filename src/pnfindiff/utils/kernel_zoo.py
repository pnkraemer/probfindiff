"""Kernel zoo."""


from functools import partial

import jax
import jax.numpy as jnp


@jax.jit
def exponentiated_quadratic(x, y, input_scale=1.0, output_scale=1.0):
    """Exponentiated quadratic kernel."""
    return output_scale * jnp.exp(-input_scale * (x - y).dot(x - y) / 2.0)


@partial(jax.jit, static_argnames=("scale", "order", "bias"))
def polynomial(x, y, *, order=2, scale=1.0, bias=1.0):
    """Polynomial kernels."""
    return (scale * x.dot(y) + bias) ** order
