"""Tests for FD coefficients."""


import jax.numpy as jnp
import pytest_cases

from pnfindiff import coefficients, diffops, kernel


def case_backward_1():
    der, acc = 1, 2
    return coefficients.backward(x=0.0, deriv=der, acc=acc, dx=1.0)


def case_backward_2():
    der, acc = 2, 2
    return coefficients.backward(x=0.0, deriv=der, acc=acc, dx=1.0)


def case_scattered_1d():
    L = diffops.grad()
    k_batch, k = kernel.exp_quad()
    lk_batch, lk = kernel.batch_gram(L(k, argnums=0))
    llk_batch, _ = kernel.batch_gram(L(lk, argnums=1))

    x = 0.5
    xs = jnp.array([0.5, 0.3, 0.1])
    return coefficients.scattered_1d(x=x, xs=xs, ks=(k_batch, lk_batch, llk_batch))


@pytest_cases.parametrize_with_cases("res", cases=".")
def test_coeff_shapes_and_cov_pos(res):
    a, b = res
    assert a.ndim == 1
    assert b.ndim == 0
    assert b > 0.0
