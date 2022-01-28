"""Finite difference coefficients. (In 1d.)"""

import functools

import jax.numpy as jnp

from pnfindiff import collocation, diffops, kernel


def backward(x, *, deriv, dx, acc=2, k=None):
    """Backward coefficients in 1d."""

    ks = _differentiate_kernel(deriv, k)
    xs = x - jnp.arange(deriv + acc) * dx
    return scattered_1d(x=x, xs=xs, ks=ks)


def forward(x, *, deriv, dx, acc=2, k=None):
    """Forward coefficients in 1d."""

    ks = _differentiate_kernel(deriv, k)
    xs = x + jnp.arange(deriv + acc) * dx
    return scattered_1d(x=x, xs=xs, ks=ks)


def _differentiate_kernel(deriv, k):
    L = functools.reduce(diffops.compose, [diffops.deriv_scalar()] * deriv)
    if k is None:
        _, k = kernel.exp_quad()
    k_batch, _ = kernel.batch_gram(k)
    lk_batch, lk = kernel.batch_gram(L(k, argnums=0))
    llk_batch, _ = kernel.batch_gram(L(lk, argnums=1))
    return k_batch, lk_batch, llk_batch


def scattered_1d(*, x, xs, ks):
    """Finite difference coefficients for scattered data."""
    k, lk, llk = ks
    n = xs.shape[0]
    x = jnp.array([x])

    K = k(xs[:, None], xs[None, :]).reshape((n, n))
    LK = lk(x[:, None], xs[None, :]).reshape((n,))
    LLK = llk(x[:, None], x[None, :]).reshape(())

    return collocation.unsymmetric(K=K, LK0=LK, LLK=LLK)