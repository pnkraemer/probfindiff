"""One-dimensional finite difference coefficients"""

import functools
from typing import Any, Tuple

import jax
import jax.numpy as jnp

from pnfindiff import collocation
from pnfindiff.typing import ArrayLike, KernelFunctionLike
from pnfindiff.utils import autodiff, kernel, kernel_zoo


@functools.partial(jax.jit, static_argnames=("deriv", "acc"))
def backward(x: ArrayLike, *, dx: float, deriv: int = 1, acc: int = 2) -> Any:
    """Backward coefficients in 1d.

    Parameters
    ----------
    x
        Point at which to compute the backward finite difference approximation.
    dx
        Step-size.
    deriv
        Order of the derivative.
    acc
        Desired accuracy.

    Returns
    -------
    :
        Finite difference coefficients and base uncertainty.
    """
    offset = -jnp.arange(deriv + acc, step=1)
    return from_offset(x=x, dx=dx, offset=offset, deriv=deriv)


@functools.partial(jax.jit, static_argnames=("deriv", "acc"))
def forward(x: ArrayLike, *, dx: float, deriv: int = 1, acc: int = 2) -> Any:
    """Forward coefficients in 1d."""
    offset = jnp.arange(deriv + acc, step=1)
    return from_offset(x=x, dx=dx, offset=offset, deriv=deriv)


@functools.partial(jax.jit, static_argnames=("deriv", "acc"))
def center(x: ArrayLike, *, dx: float, deriv: int = 1, acc: int = 2) -> Any:
    """Forward coefficients in 1d."""
    num = (deriv + acc) // 2
    offset = jnp.arange(-num, num + 1, step=1)
    return from_offset(x=x, dx=dx, offset=offset, deriv=deriv)


@functools.partial(jax.jit, static_argnames=("deriv",))
def from_offset(x: ArrayLike, *, dx: float, offset: ArrayLike, deriv: int = 1) -> Any:
    """Forward coefficients in 1d."""
    xs = x + offset * dx
    k = kernel_zoo.exponentiated_quadratic
    L = functools.reduce(autodiff.compose, [autodiff.deriv_scalar] * deriv)

    ks = kernel.differentiate(k=k, L=L)
    return collocation.non_uniform_nd(x=jnp.array([x]), xs=xs[:, None], ks=ks)


@functools.partial(jax.jit, static_argnames=("ks",))
def non_uniform_1d(
    *,
    x: ArrayLike,
    xs: ArrayLike,
    ks: Tuple[KernelFunctionLike, KernelFunctionLike, KernelFunctionLike]
) -> Any:
    """Finite difference coefficients for non-uniform data."""
    return collocation.non_uniform_nd(x=jnp.array([x]), xs=xs[:, None], ks=ks)
