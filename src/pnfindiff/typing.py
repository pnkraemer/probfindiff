"""Common types."""
from typing import Any, Callable

KernelFunctionLike = Callable[[Any, Any], Any]
"""Kernel function type."""

ArrayLike = Any
"""Array type. JAX's arrays cannot be assigned a strict type, so we use 'Any'."""
