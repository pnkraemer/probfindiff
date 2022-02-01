"""Top-level API."""

from functools import partial
from typing import Any, Callable

import jax
import jax.numpy as jnp

from pnfindiff import schemes
from pnfindiff.typing import ArrayLike


def derivative(f: Callable[[Any], Any], **kwargs: Any) -> ArrayLike:
    """Compute the derivative of a function based on its values."""
    return differentiate(f, scheme=schemes.derivative(**kwargs))


def derivative_higher(f: Callable[[Any], Any], **kwargs: Any) -> ArrayLike:
    """Compute the higher derivative of a function based on its values."""
    return differentiate(f, scheme=schemes.derivative_higher(**kwargs))


@jax.jit
def differentiate(f: ArrayLike, *, scheme: schemes.FiniteDifferenceScheme) -> ArrayLike:
    """Apply a finite difference scheme to a vector of function evaluations.

    Parameters
    ----------
    f
        Array of function evaluated to be differentiated numerically. Shape ``(n,)``.
    scheme
        PN finite difference schemes.


    Returns
    -------
    :
        Finite difference approximation and the corresponding base-uncertainty. Shapes `` (n,), (n,)``.
    """
    weights, unc_base, indices = scheme
    dfx = jnp.einsum("nk,nk->n", weights, f[indices])
    return dfx, unc_base


def differentiate_along_axis(
    f: ArrayLike, *, axis: int, scheme: schemes.FiniteDifferenceScheme
) -> ArrayLike:
    """Apply a finite difference scheme along a specified axis.

    Parameters
    ----------
    f
        Array of function evaluated to be differentiated numerically. Shape ``(..., n, ...)``.
    axis
        Axis along which the scheme should be applied.
    scheme
        PN finite difference schemes.


    Returns
    -------
    :
        Finite difference approximation and the corresponding base-uncertainty. Shapes `` (..., n, ...), (n,)``.
    """

    fd = partial(differentiate, scheme=scheme)
    return jnp.apply_along_axis(fd, axis=axis, arr=f)
