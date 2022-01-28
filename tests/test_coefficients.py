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
    L = diffops.grad()
    k_batch, k = kernel.exp_quad()
    lk_batch, lk = kernel.vmap_gram(L(k, argnums=0))
    llk_batch, llk = kernel.vmap_gram(L(lk, argnums=1))

    x = 0.5
    xs = jnp.array([0.5, 0.3, 0.1])
    return coefficients.scattered_1d(x=x, xs=xs, ks=(k_batch, lk_batch, llk_batch))


@pytest_cases.parametrize_with_cases("res", cases=".")
def test_coeff_shapes_and_cov_pos(res):
    a, b = res
    assert a.ndim==1
    assert b.ndim == 0
    assert b > 0.0
