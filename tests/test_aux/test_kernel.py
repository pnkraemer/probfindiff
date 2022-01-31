"""Test for kernel functionality."""


import jax.numpy as jnp
import pytest_cases

from pnfindiff.aux import diffop, kernel


def case_exp_quad():
    k = lambda x, y: jnp.exp(-(x - y).dot(x - y))
    return kernel.batch_gram(k)[0]


def case_exp_quad_builtin():
    return kernel.exp_quad()[0]


def case_differentiate_0():
    k = lambda x, y: (x - y).dot(x - y)
    return kernel.differentiate(k, L=diffop.deriv_scalar)[0]


def case_differentiate_1():
    k = lambda x, y: (x - y).dot(x - y)
    return kernel.differentiate(k, L=diffop.deriv_scalar)[1]


def case_differentiate_2():
    k = lambda x, y: (x - y).dot(x - y)
    return kernel.differentiate(k, L=diffop.deriv_scalar)[2]


def case_polynomial_builtin():
    return kernel.polynomial(order=3, bias=2)[0]


@pytest_cases.parametrize_with_cases("exp_quad", cases=".")
def test_vectorize_gram_shapes(exp_quad):
    xs = jnp.arange(8.0).reshape((4, 2))
    ys = jnp.arange(12.0).reshape((6, 2))
    assert exp_quad(xs, ys.T).shape == (4, 6)
