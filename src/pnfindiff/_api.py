"""Top-level API."""

from functools import partial
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp

from pnfindiff import coefficients
from pnfindiff.typing import ArrayLike


def derivative(f: Callable[[Any], Any], **kwargs: Any) -> ArrayLike:
    """Compute the derivative of a function based on its values."""
    coeffs, indices = coefficients.derivative(**kwargs)
    return differentiate(f, coeffs=coeffs, indices=indices)


def derivative_higher(f: Callable[[Any], Any], **kwargs: Any) -> ArrayLike:
    """Compute the higher derivative of a function based on its values."""
    coeffs, indices = coefficients.derivative_higher(**kwargs)
    return differentiate(f, coeffs=coeffs, indices=indices)


@jax.jit
def differentiate(
    f: ArrayLike, *, coeffs: Tuple[ArrayLike, ArrayLike], indices: ArrayLike
) -> ArrayLike:
    """Apply a finite difference scheme to a vector of function evaluations.

    Parameters
    ----------
    f
        Array of function evaluated to be differentiated numerically. Shape ``(n,)``.
    coeffs
        PN finite difference coefficients. Shapes `` (n,k), (n,)``.
    indices
        Indices of the neighbours that shall be used for each derivative. Shape ``(n,k)``.


    Returns
    -------
    :
        Finite difference approximation and the corresponding base-uncertainty. Shapes `` (n,), (n,)``.
    """
    weights, unc_base = coeffs
    dfx = jnp.einsum("nk,nk->n", weights, f[indices])
    return dfx, unc_base


def differentiate_along_axis(
    f: ArrayLike, *, axis: int, coeffs: Tuple[ArrayLike, ArrayLike], indices: ArrayLike
) -> ArrayLike:
    """Apply a finite difference scheme along a specified axis.

    Parameters
    ----------
    f
        Array of function evaluated to be differentiated numerically. Shape ``(..., n, ...)``.
    axis
        Axis along which the scheme should be applied.
    coeffs
        PN finite difference coefficients. Shapes `` (n,k), (n,)``.
    indices
        Indices of the neighbours that shall be used for each derivative. Shape ``(n,k)``.


    Returns
    -------
    :
        Finite difference approximation and the corresponding base-uncertainty. Shapes `` (..., n, ...), (n,)``.
    """
    fd = partial(differentiate, coeffs=coeffs, indices=indices)
    return jnp.apply_along_axis(fd, axis=axis, arr=f)
