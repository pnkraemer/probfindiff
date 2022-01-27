"""Test for kernel functionality."""


import jax.numpy as jnp
import pytest

from pnfindiff import kernel


@pytest.fixture
def exp_quad():
    k = lambda x, y: jnp.exp(-(x - y).dot(x - y))
    return kernel.gram_matrix_function(k)


def test_gram_matrix_function_pairwise(exp_quad):
    xs = jnp.arange(4)
    assert exp_quad(xs, xs).shape == ()


def test_gram_matrix_function_diagonal(exp_quad):
    xs = jnp.arange(8).reshape((4, 2))
    assert exp_quad(xs, xs).shape == (4,)


def test_gram_matrix_function_outer(exp_quad):
    xs = jnp.arange(8).reshape((4, 2))
    ys = jnp.arange(12).reshape((6, 2))
    assert exp_quad(xs, ys.T).shape == (4, 6)
