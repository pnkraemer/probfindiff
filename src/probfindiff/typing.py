"""Common types."""
from typing import Any, Callable

from jax import Array as _Array  # temp
from jax.typing import ArrayLike as _ArrayLike  # temp

Array = _Array

ArrayLike = Any
"""Array type. JAX's arrays cannot be assigned a strict type, so we use 'Any'."""


KernelFunctionLike = Callable[[Any, Any], Any]
"""Kernel function type."""


DifferentialOperatorLike = Callable[..., Callable[..., Any]]
"""Differential operators transform functions.

The ellipsis is necessary because of keyword arguments:
    https://github.com/python/mypy/issues/1655
"""
