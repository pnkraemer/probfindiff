"""Top-level API."""

from typing import Any, Callable

from pnfindiff import coefficients
from pnfindiff.typing import ArrayLike

__all__ = ["derivative", "derivative_higher"]


def derivative(f: Callable[[Any], Any], **kwargs: Any) -> ArrayLike:
    """Compute the derivative of a function based on its values."""
    coeffs, indices = coefficients.derivative(**kwargs)
    return coefficients.apply(f, coeffs=coeffs, indices=indices)


def derivative_higher(f: Callable[[Any], Any], **kwargs: Any) -> ArrayLike:
    """Compute the higher derivative of a function based on its values."""
    coeffs, indices = coefficients.derivative_higher(**kwargs)
    return coefficients.apply(f, coeffs=coeffs, indices=indices)
