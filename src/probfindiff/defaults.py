"""Default constants."""

import functools

import jax.numpy as jnp

from probfindiff.backend.typing import Callable
from probfindiff.utils import kernel_zoo

NOISE_VARIANCE = 1e-14
"""Function observation noise."""

ORDER_DERIVATIVE = 1
"""Derivative order."""

ORDER_METHOD_CENTRAL = 2
"""Order of central finite difference schemes."""

ORDER_METHOD = ORDER_METHOD_CENTRAL + 1
"""Order of non-central finite difference schemes."""


def kernel(*, min_order: int) -> Callable:
    """Default kernel function."""
    return functools.partial(kernel_zoo.polynomial, p=jnp.ones((min_order,)))
