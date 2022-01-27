"""Test for kernel functionality."""


import jax.numpy as jnp
import pytest_cases

from pnfindiff import kernel


def case_exp_quad():
    k = lambda x, y: jnp.exp(-(x - y).dot(x - y))
    return kernel.vmap_gram(k)


def case_exp_quad_builtin():
    return kernel.exp_quad()


@pytest_cases.parametrize_with_cases("exp_quad", cases=".")
def test_vectorize_gram_shapes(exp_quad):
    xs = jnp.arange(8).reshape((4, 2))
    ys = jnp.arange(12).reshape((6, 2))
    assert exp_quad(xs, ys.T).shape == (4, 6)
