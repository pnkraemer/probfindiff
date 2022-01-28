"""Tests for FD coefficients."""


from pnfindiff import coefficients, diffops, kernel
import pytest

import jax
import jax.numpy as jnp


import pytest_cases

def case_backward():
    der, acc = 1, 2
    return  coefficients.backward(x=0.0, deriv=der, acc=acc, dx=1.)


def case_scattered_1d():
    k = lambda x, y: jnp.exp(-jnp.dot(x - y, x - y) / 2.0)

    L = diffops.grad()
    lk = L(k, argnums=0)
    llk = L(lk, argnums=1)

    k = kernel.vmap_gram(k)
    lk = kernel.vmap_gram(lk)
    llk = kernel.vmap_gram(llk)

    x = 0.5
    xs = jnp.array([0.5, 0.3, 0.1])
    return coefficients.scattered_1d(x=x, xs=xs, ks=(k, lk, llk))


@pytest_cases.parametrize_with_cases("res", cases=".")
def test_coeff_shapes_and_cov_pos(res):
    a, b = res
    assert a.ndim==1
    assert b.ndim == 0
    assert b > 0.0
