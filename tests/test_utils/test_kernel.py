"""Test for kernel functionality."""


import jax.numpy as jnp
import pytest_cases

from pnfindiff.utils import autodiff, kernel, kernel_zoo


def case_exp_quad():
    k = lambda x, y: jnp.exp(-(x - y).dot(x - y))
    return kernel.batch_gram(k)[0]


def case_exp_quad_builtin():
    return kernel.batch_gram(kernel_zoo.exp_quad)[0]


def case_differentiate_0():
    k = lambda x, y: (x - y).dot(x - y)
    return kernel.differentiate(k, L=autodiff.deriv_scalar)[0]


def case_differentiate_1():
    k = lambda x, y: (x - y).dot(x - y)
    return kernel.differentiate(k, L=autodiff.deriv_scalar)[1]


def case_differentiate_2():
    k = lambda x, y: (x - y).dot(x - y)
    return kernel.differentiate(k, L=autodiff.deriv_scalar)[2]


def case_polynomial_builtin():
    return kernel.batch_gram(lambda x, y: kernel_zoo.polynomial(x, y, order=3, bias=2))[
        0
    ]


@pytest_cases.parametrize_with_cases("k", cases=".")
def test_vectorize_gram_shapes(k):
    xs = jnp.arange(8.0).reshape((4, 2))
    ys = jnp.arange(12.0).reshape((6, 2))
    assert k(xs, ys.T).shape == (4, 6)
