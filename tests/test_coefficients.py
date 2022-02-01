"""Tests for pnfindiff's top-level API."""


import jax.numpy as jnp
import pytest
import pytest_cases

from pnfindiff import coefficients


@pytest.fixture
def xs():
    return jnp.linspace(1, 1.1, 10)


def case_derivative_higher(xs):
    return coefficients.derivative_higher(xs=xs, deriv=1, num=3)


@pytest_cases.parametrize_with_cases("fd", cases=".")
def test_apply(fd, xs):
    f = jnp.sin(xs)
    df_approx, _ = coefficients.apply(f, fd=fd)

    assert jnp.allclose(df_approx, jnp.cos(xs), atol=1e-4, rtol=1e-4)


@pytest_cases.parametrize_with_cases("fd", cases=".")
def test_apply_along_axis(fd, xs):
    ys = xs
    f = jnp.sin(xs)[:, None] * jnp.sin(ys)[None, :]
    df_dy = jnp.sin(xs)[:, None] * jnp.cos(ys)[None, :]

    df_approx, _ = coefficients.apply_along_axis(f, axis=1, fd=fd)

    assert jnp.allclose(df_approx, df_dy, atol=1e-4, rtol=1e-4)
