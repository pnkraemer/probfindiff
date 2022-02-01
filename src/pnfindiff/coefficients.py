"""Finite difference coefficients. (In 1d.)"""

import functools

import jax
import jax.numpy as jnp

from pnfindiff import collocation
from pnfindiff.utils import autodiff, kernel, kernel_zoo


@functools.partial(jax.jit, static_argnames=("deriv", "acc"))
def backward(x, *, dx, deriv=1, acc=2):
    """Backward coefficients in 1d."""
    offset = -jnp.arange(deriv + acc, step=1)
    return from_offset(x=x, dx=dx, offset=offset, deriv=deriv)


@functools.partial(jax.jit, static_argnames=("deriv", "acc"))
def forward(x, *, dx, deriv=1, acc=2):
    """Forward coefficients in 1d."""
    offset = jnp.arange(deriv + acc, step=1)
    return from_offset(x=x, dx=dx, offset=offset, deriv=deriv)


@functools.partial(jax.jit, static_argnames=("deriv", "acc"))
def center(x, *, dx, deriv=1, acc=2):
    """Forward coefficients in 1d."""
    num = (deriv + acc) // 2
    offset = jnp.arange(-num, num + 1, step=1)
    return from_offset(x=x, dx=dx, offset=offset, deriv=deriv)


@functools.partial(jax.jit, static_argnames=("deriv",))
def from_offset(x, *, dx, offset, deriv=1):
    """Forward coefficients in 1d."""
    xs = x + offset * dx
    _, k = kernel_zoo.exp_quad()
    L = functools.reduce(autodiff.compose, [autodiff.deriv_scalar] * deriv)

    ks = kernel.differentiate(k=k, L=L)
    return non_uniform_1d(x=x, xs=xs, ks=ks)


@functools.partial(jax.jit, static_argnames=("ks",))
def non_uniform_1d(*, x, xs, ks):
    """Finite difference coefficients for non-uniform data."""
    return non_uniform_nd(x=jnp.array([x]), xs=xs[:, None], ks=ks)


@functools.partial(jax.jit, static_argnames=("ks",))
def non_uniform_nd(*, x, xs, ks):
    """Finite difference coefficients for non-uniform data in multiple dimensions."""

    k, lk, llk = ks
    n = xs.shape[0]

    K = k(xs, xs.T).reshape((n, n))
    LK = lk(x[None, :], xs.T).reshape((n,))
    LLK = llk(x[None, :], x[None, :].T).reshape(())
    return collocation.unsymmetric(K=K, LK0=LK, LLK=LLK)
