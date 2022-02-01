"""Top-level API."""

from pnfindiff import coefficients

__all__ = ["derivative", "derivative_higher"]


def derivative(f, **kwargs):
    coeffs, indices = coefficients.derivative(**kwargs)
    return coefficients.apply(f, coeffs=coeffs, indices=indices)


def derivative_higher(f, **kwargs):
    coeffs, indices = coefficients.derivative_higher(**kwargs)
    return coefficients.apply(f, coeffs=coeffs, indices=indices)
