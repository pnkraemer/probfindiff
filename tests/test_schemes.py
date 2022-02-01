"""Tests for pnfindiff's top-level API."""


import jax.numpy as jnp
import pytest
import pytest_cases

import pnfindiff
from pnfindiff import schemes


@pytest.fixture(name="xs")
def fixture_xs():
    return jnp.linspace(1, 1.1, 10)


def case_derivative_higher(xs):
    return schemes.derivative_higher(xs=xs, deriv=1, num=3)


def case_derivative(xs):
    return schemes.derivative(xs=xs, num=3)


@pytest_cases.parametrize_with_cases("fd", cases=".")
def test_differentiate(fd, xs):
    f = jnp.sin(xs)
    df_approx, _ = pnfindiff.differentiate(f, scheme=fd)

    assert jnp.allclose(df_approx, jnp.cos(xs), atol=1e-4, rtol=1e-4)


@pytest_cases.parametrize_with_cases("fd", cases=".")
def test_differentiate_along_axis(fd, xs):
    ys = xs
    f = jnp.sin(xs)[:, None] * jnp.sin(ys)[None, :]
    df_dy = jnp.sin(xs)[:, None] * jnp.cos(ys)[None, :]

    df_approx, _ = pnfindiff.differentiate_along_axis(f, axis=1, scheme=fd)

    assert jnp.allclose(df_approx, df_dy, atol=1e-4, rtol=1e-4)
