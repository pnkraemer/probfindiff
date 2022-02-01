"""Kernel zoo."""


from functools import partial
from typing import Any

import jax
import jax.numpy as jnp


@jax.jit
def exponentiated_quadratic(
    x: Any, y: Any, input_scale: float = 1.0, output_scale: float = 1.0
) -> Any:
    """Exponentiated quadratic kernel."""
    return output_scale * jnp.exp(-input_scale * (x - y).dot(x - y) / 2.0)


@partial(jax.jit, static_argnames=("scale", "order", "bias"))
def polynomial(
    x: Any, y: Any, *, order: int = 2, scale: float = 1.0, bias: float = 1.0
) -> Any:
    """Polynomial kernels."""
    return (scale * x.dot(y) + bias) ** order
