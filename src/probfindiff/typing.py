"""Common types."""
from typing import Any, Callable

from jax import Array as _Array  # temp
from jax.typing import ArrayLike as _ArrayLike  # temp

# Expose JAX's Array types.

Array = _Array
"""Array output type."""

ArrayLike = _ArrayLike
"""Array input type."""


KernelFunctionLike = Callable[[ArrayLike, ArrayLike], Array]
"""Kernel function type."""


DifferentialOperatorLike = Callable[..., Callable[..., Any]]
"""Differential operators transform functions.

The ellipsis is necessary because of keyword arguments:
    https://github.com/python/mypy/issues/1655
"""
