"""Tests for the top-level API."""
import jax.numpy as jnp
import pytest

import pnfindiff


@pytest.mark.parametrize("f,df_true", ((jnp.sin, jnp.cos),))
def test_derivative(f, df_true):
    xs = jnp.linspace(0.0, 1.0, num=12)

    fx = f(xs)
    dfx_true = df_true(xs)
    dfx_approx, _ = pnfindiff.derivative(fx, xs=xs)

    assert jnp.allclose(dfx_approx, dfx_true, atol=1e-3, rtol=1e-3)


@pytest.mark.parametrize("f,df_true", ((jnp.sin, jnp.cos),))
def test_derivative_higher(f, df_true):
    xs = jnp.linspace(0.0, 1.0, num=12)

    fx = f(xs)
    dfx_true = df_true(xs)
    dfx_approx, _ = pnfindiff.derivative_higher(fx, deriv=1, xs=xs)

    assert jnp.allclose(dfx_approx, dfx_true, atol=1e-3, rtol=1e-3)
