"""Top-level API."""

import functools
from collections import namedtuple
from functools import partial
from typing import Optional, Tuple

import jax
import jax.numpy as jnp

from probfindiff import collocation, defaults, stencil
from probfindiff.typing import Array, ArrayLike, KernelFunctionLike
from probfindiff.utils import autodiff
from probfindiff.utils import kernel as kernel_module

FiniteDifferenceScheme = namedtuple(
    "FiniteDifferenceScheme",
    (
        "weights",
        "covs_marginal",
        "order_derivative",
    ),
)
"""Finite difference schemes.

A finite difference scheme consists of a weight-vector, marginal covariances, and offset indices.
"""


@jax.jit
def differentiate(
    fx: ArrayLike, *, scheme: FiniteDifferenceScheme
) -> Tuple[Array, Array]:
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
    fx = jnp.asarray(fx)

    weights, unc_base = scheme.weights, scheme.covs_marginal
    dfx = jnp.einsum("...k,...k->...", weights, fx)  # type: ignore  # (einsum not typed)
    return dfx, unc_base


@functools.partial(jax.jit, static_argnames=["axis"])
def differentiate_along_axis(
    fx: ArrayLike, *, axis: int, scheme: FiniteDifferenceScheme
) -> Array:
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
    fx = jnp.asarray(fx)

    fd = partial(differentiate, scheme=scheme)
    differentiated: Array = jnp.apply_along_axis(fd, axis=axis, arr=fx)
    return differentiated


@functools.partial(
    jax.jit, static_argnames=("order_derivative", "order_method", "kernel")
)
def backward(
    *,
    dx: float,
    order_derivative: int = defaults.ORDER_DERIVATIVE,
    order_method: int = defaults.ORDER_METHOD,
    kernel: Optional[KernelFunctionLike] = None,
    noise_variance: float = defaults.NOISE_VARIANCE,
) -> Tuple[FiniteDifferenceScheme, Array]:
    """Backward coefficients in 1d.

    Parameters
    ----------
    dx
        Step-size.
    order_derivative
        Order of the derivative.
    order_method
        Desired accuracy.
    kernel
        Kernel function. Defines the function-model.
    noise_variance
        Variance of the observation noise.

    Returns
    -------
    :
        Finite difference coefficients and base uncertainty.
    """
    grid = stencil.backward(
        dx=dx, order_derivative=order_derivative, order_method=order_method
    )
    scheme = from_grid(
        xs=grid,
        order_derivative=order_derivative,
        kernel=kernel,
        noise_variance=noise_variance,
    )
    return scheme, grid


@functools.partial(
    jax.jit, static_argnames=("order_derivative", "order_method", "kernel")
)
def forward(
    *,
    dx: float,
    order_derivative: int = defaults.ORDER_DERIVATIVE,
    order_method: int = defaults.ORDER_METHOD,
    kernel: Optional[KernelFunctionLike] = None,
    noise_variance: float = defaults.NOISE_VARIANCE,
) -> Tuple[FiniteDifferenceScheme, Array]:
    """Forward coefficients in 1d.

    Parameters
    ----------
    dx
        Step-size.
    order_derivative
        Order of the derivative.
    order_method
        Desired accuracy.
    kernel
        Kernel function. Defines the function-model.
    noise_variance
        Variance of the observation noise.

    Returns
    -------
    :
        Finite difference coefficients and base uncertainty.
    """
    grid = stencil.forward(
        dx=dx, order_derivative=order_derivative, order_method=order_method
    )
    scheme = from_grid(
        xs=grid,
        order_derivative=order_derivative,
        kernel=kernel,
        noise_variance=noise_variance,
    )
    return scheme, grid


@functools.partial(
    jax.jit, static_argnames=("order_derivative", "order_method", "kernel")
)
def central(
    *,
    dx: float,
    order_derivative: int = defaults.ORDER_DERIVATIVE,
    order_method: int = defaults.ORDER_METHOD_CENTRAL,
    kernel: Optional[KernelFunctionLike] = None,
    noise_variance: float = defaults.NOISE_VARIANCE,
) -> Tuple[FiniteDifferenceScheme, Array]:
    """Central coefficients in 1d.

    Parameters
    ----------
    dx
        Step-size.
    order_derivative
        Order of the derivative.
    order_method
        Desired accuracy.
    kernel
        Kernel function. Defines the function-model.
    noise_variance
        Variance of the observation noise.

    Returns
    -------
    :
        Finite difference coefficients and base uncertainty.
    """
    grid = stencil.central(
        dx=dx, order_derivative=order_derivative, order_method=order_method
    )
    scheme = from_grid(
        xs=grid,
        order_derivative=order_derivative,
        kernel=kernel,
        noise_variance=noise_variance,
    )
    return scheme, grid


def from_grid(
    *,
    xs: ArrayLike,
    order_derivative: int = defaults.ORDER_DERIVATIVE,
    kernel: Optional[KernelFunctionLike] = None,
    noise_variance: float = defaults.NOISE_VARIANCE,
) -> FiniteDifferenceScheme:
    """Finite difference coefficients based on an array of offset indices.

    Parameters
    ----------
    order_derivative
        Order of the derivative.
    xs
        Grid. Shape ``(n,)``.
    kernel
        Kernel function. Defines the function-model.
    noise_variance
        Variance of the observation noise.

    Returns
    -------
    :
        Finite difference coefficients and base uncertainty.
    """
    xs = jnp.asarray(xs)

    if kernel is None:
        kernel = defaults.kernel(min_order=xs.shape[0])
    L = functools.reduce(autodiff.compose, [autodiff.derivative] * order_derivative)
    ks = kernel_module.differentiate(k=kernel, L=L)

    x = jnp.zeros_like(xs[0])
    weights, cov_marginal = collocation.non_uniform_nd(
        x=x[..., None],
        xs=xs[..., None],
        ks=ks,
        noise_variance=noise_variance,
    )
    scheme = FiniteDifferenceScheme(
        weights,
        cov_marginal,
        order_derivative=order_derivative,
    )
    return scheme
