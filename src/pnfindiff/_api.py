"""Top-level API."""

import functools
from collections import namedtuple
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp

from pnfindiff import collocation
from pnfindiff.typing import ArrayLike
from pnfindiff.utils import autodiff, kernel, kernel_zoo

FiniteDifferenceScheme = namedtuple(
    "FiniteDifferenceScheme",
    (
        "weights",
        "covs_marginal",
        "offset_indices",
        "order_method",
        "order_derivative",
    ),
)
"""Finite difference schemes.

A finite difference scheme consists of a weight-vector, marginal covariances, and offset indices.
"""


@jax.jit
def differentiate(fx: ArrayLike, *, scheme: FiniteDifferenceScheme) -> ArrayLike:
    """Apply a finite difference scheme to a vector of function evaluations.

    Parameters
    ----------
    fx
        Array of function evaluated to be differentiated numerically. Shape ``(n,)``.
    scheme
        PN finite difference schemes.


    Returns
    -------
    :
        Finite difference approximation and the corresponding base-uncertainty. Shapes `` (n,), (n,)``.
    """
    weights, unc_base, indices, *_ = scheme
    dfx = jnp.einsum("nk,nk->n", weights, fx[indices])
    return dfx, unc_base


def differentiate_along_axis(
    fx: ArrayLike, *, axis: int, scheme: FiniteDifferenceScheme
) -> ArrayLike:
    """Apply a finite difference scheme along a specified axis.

    Parameters
    ----------
    fx
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
    return jnp.apply_along_axis(fd, axis=axis, arr=fx)


@functools.partial(jax.jit, static_argnames=("order_derivative", "order_method"))
def backward(*, dx: float, order_derivative: int = 1, order_method: int = 2) -> Any:
    """Backward coefficients in 1d.

    Parameters
    ----------
    dx
        Step-size.
    order_derivative
        Order of the derivative.
    order_method
        Desired accuracy.

    Returns
    -------
    :
        Finite difference coefficients and base uncertainty.
    """
    offset = -jnp.arange(order_derivative + order_method, step=1)
    return from_grid(xs=offset * dx, order_derivative=order_derivative)


@functools.partial(jax.jit, static_argnames=("order_derivative", "order_method"))
def forward(*, dx: float, order_derivative: int = 1, order_method: int = 2) -> Any:
    """Forward coefficients in 1d.

    Parameters
    ----------
    dx
        Step-size.
    order_derivative
        Order of the derivative.
    order_method
        Desired accuracy.

    Returns
    -------
    :
        Finite difference coefficients and base uncertainty.
    """
    offset = jnp.arange(order_derivative + order_method, step=1)
    return from_grid(xs=offset * dx, order_derivative=order_derivative)


@functools.partial(jax.jit, static_argnames=("order_derivative", "order_method"))
def central(*, dx: float, order_derivative: int = 1, order_method: int = 2) -> Any:
    """Central coefficients in 1d.

    Parameters
    ----------
    dx
        Step-size.
    order_derivative
        Order of the derivative.
    order_method
        Desired accuracy.

    Returns
    -------
    :
        Finite difference coefficients and base uncertainty.
    """
    num_central = (2 * ((order_derivative + 1.0) / 2.0) // 2) - 1 + order_method
    num_side = num_central // 2
    offset = jnp.arange(-num_side, num_side + 1, step=1)
    return from_grid(xs=offset * dx, order_derivative=order_derivative)


@functools.partial(jax.jit, static_argnames=("order_derivative",))
def from_grid(*, xs: ArrayLike, order_derivative: int = 1) -> Any:
    """Finite difference coefficients based on an array of offset indices.

    Parameters
    ----------
    order_derivative
        Order of the derivative.
    xs
        Grid. Shape ``(n,)``.

    Returns
    -------
    :
        Finite difference coefficients and base uncertainty.
    """
    k = functools.partial(kernel_zoo.polynomial, p=jnp.ones((xs.shape[0],)))
    L = functools.reduce(autodiff.compose, [autodiff.derivative] * order_derivative)

    ks = kernel.differentiate(k=k, L=L)
    weights, cov_marginal = collocation.non_uniform_nd(
        x=xs[0].reshape((-1,)), xs=xs[:, None], ks=ks
    )
    scheme = FiniteDifferenceScheme(
        weights,
        cov_marginal,
        offset_indices=jnp.arange(xs.shape[0] + 1),
        order_derivative=order_derivative,
        order_method=len(xs),
    )
    return scheme, xs
