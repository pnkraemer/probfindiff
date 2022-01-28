"""Tests for FD coefficients."""


from pnfindiff import coefficients, diffops, kernel
import pytest

import jax
import jax.numpy as jnp

def test_backward():

    with pytest.raises(NotImplementedError):
        coefficients.backward(deriv=1, acc=2, dx=0.1)


def test_scattered1d():
    k = lambda x, y: jnp.exp(-jnp.dot(x - y, x-y) / 2.0)

    L = diffops.grad()
    lk = L(k, argnums=0)
    llk = L(lk, argnums=1)

    k = kernel.vmap_gram(k)
    lk = kernel.vmap_gram(lk)
    llk = kernel.vmap_gram(llk)

    x = jnp.array([0.5])
    xs = jnp.array([0.5, 0.3, 0.1])
    a, b = coefficients.scattered1d(x=x, xs=xs, ks=(k, lk, llk))
    assert a.shape == (3,)
    assert b.shape == ()
    assert b > 0.0
