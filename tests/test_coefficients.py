"""Tests for FD coefficients."""


import jax.numpy as jnp
import pytest_cases

from pnfindiff import coefficients
from pnfindiff.aux import diffop, kernel


def case_backward():
    return coefficients.backward(x=0.0, deriv=1, acc=2, dx=1.0)


def case_forward():
    return coefficients.forward(x=0.0, deriv=1, acc=2, dx=1.0)


def case_central():
    return coefficients.central(x=0.0, deriv=1, acc=2, dx=1.0)


def case_from_offset():
    return coefficients.from_offset(
        x=0.0, deriv=1, offset=jnp.arange(-2.0, 3.0), dx=1.0
    )


def case_scattered_1d():
    ks = kernel.differentiate(kernel.exp_quad()[1], L=diffop.deriv_scalar)

    x = 0.5
    xs = jnp.array([0.5, 0.3, 0.1])
    return coefficients.scattered_1d(x=x, xs=xs, ks=ks)


def case_scattered_2d():
    L = diffop.laplace
    k_batch, k = kernel.exp_quad()
    lk_batch, lk = kernel.batch_gram(L(k, argnums=0))
    llk_batch, _ = kernel.batch_gram(L(lk, argnums=1))

    x = 0.5 * jnp.ones((1,))
    xs = jnp.array([0.5, 0.3, 0.1, 0.2, 0.4]).reshape((-1, 1))
    return coefficients.scattered_nd(x=x, xs=xs, ks=(k_batch, lk_batch, llk_batch))


@pytest_cases.parametrize_with_cases("res", cases=".")
def test_coeff_shapes_and_cov_pos(res):
    a, b = res
    assert a.ndim == 1
    assert b.ndim == 0
    assert b > 0.0
