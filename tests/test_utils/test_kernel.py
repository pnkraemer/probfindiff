"""Test for kernel functionality."""


import functools

import jax
import jax.numpy as jnp
import pytest
import pytest_cases

from probfindiff.utils import autodiff, kernel, kernel_zoo


def case_exponentiated_quadratic():
    k = lambda x, y: jnp.exp(-(x - y).dot(x - y))
    return kernel.batch_gram(k)[0]


def case_exponentiated_quadratic_builtin():
    return kernel.batch_gram(kernel_zoo.exponentiated_quadratic)[0]


def case_differentiate_0():
    k = lambda x, y: (x - y).dot(x - y)
    return kernel.differentiate(k, L=autodiff.derivative)[0]


def case_differentiate_1():
    k = lambda x, y: (x - y).dot(x - y)
    return kernel.differentiate(k, L=autodiff.derivative)[1]


def case_differentiate_2():
    k = lambda x, y: (x - y).dot(x - y)
    return kernel.differentiate(k, L=autodiff.derivative)[2]


def case_polynomial_builtin():
    k = functools.partial(kernel_zoo.polynomial, p=jnp.ones((3,)))
    return kernel.batch_gram(k)[0]


@pytest_cases.parametrize_with_cases("k", cases=".")
def test_vectorize_gram_shapes(k):
    xs = jnp.arange(8.0).reshape((4, 2))
    ys = jnp.arange(12.0).reshape((6, 2))
    assert k(xs, ys.T).shape == (4, 6)


@pytest.mark.parametrize("L, d, diffop_shape", ([jax.jacfwd, 2, (2,)],))
def test_kernel_batch_shape(L, d, diffop_shape):

    k = kernel_zoo.exponentiated_quadratic
    k_batch, lk_batch, llk_batch = kernel.differentiate(k, L=L)

    num_xs, num_ys = 4, 3
    xs = jnp.arange(1, 1 + d * num_xs, dtype=float).reshape((num_xs, d))
    ys = jnp.arange(1, 1 + d * num_ys, dtype=float).reshape((num_ys, d))

    k_shape = (num_xs, num_ys)
    assert k_batch(xs, ys.T).shape == k_shape
    assert lk_batch(xs, ys.T).shape == diffop_shape + k_shape
    assert llk_batch(xs, ys.T).shape == diffop_shape + diffop_shape + k_shape
