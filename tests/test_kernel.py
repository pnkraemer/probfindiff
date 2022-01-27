"""Test for kernel functionality."""


import jax.numpy as jnp
import pytest_cases

from pnfindiff import kernel


def case_exp_quad():
    k = lambda x, y: jnp.exp(-(x - y).dot(x - y))
    return kernel.gram_matrix_function(k)


def case_exp_quad_builtin():
    k = lambda x, y: jnp.exp(-(x - y).dot(x - y))
    return kernel.gram_matrix_function(k)


@pytest_cases.parametrize_with_cases("exp_quad", cases=".")
def test_gram_matrix_function_pairwise(exp_quad):
    xs = jnp.arange(4)
    assert exp_quad(xs, xs).shape == ()


@pytest_cases.parametrize_with_cases("exp_quad", cases=".")
def test_gram_matrix_function_diagonal(exp_quad):
    xs = jnp.arange(8).reshape((4, 2))
    assert exp_quad(xs, xs).shape == (4,)


@pytest_cases.parametrize_with_cases("exp_quad", cases=".")
def test_gram_matrix_function_outer(exp_quad):
    xs = jnp.arange(8).reshape((4, 2))
    ys = jnp.arange(12).reshape((6, 2))
    assert exp_quad(xs, ys.T).shape == (4, 6)
