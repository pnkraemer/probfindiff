"""Top-level API."""

from pnfindiff import coefficients

__all__ = ["derivative"]


def derivative(f, **kwargs):
    coeffs, indices = coefficients.derivative(**kwargs)
    return coefficients.apply(f, coeffs=coeffs, indices=indices)
