"""Tests for pnfindiff's top-level API."""


import jax.numpy as jnp

import pnfindiff


def test_findiff():
    x = jnp.linspace(1, 1.1, 10)
    f = jnp.sin(x)
    fd = pnfindiff.findiff(xs=x, deriv=1, num=3)

    df_approx, _ = fd(f)

    assert jnp.allclose(df_approx, jnp.cos(x), atol=1e-4, rtol=1e-4)


def test_findiff_along_axis():
    x = jnp.linspace(1, 1.1, 10)
    y = jnp.linspace(1, 1.1, 10)
    f = jnp.sin(x)[:, None] * jnp.sin(y)[None, :]
    df_dy = jnp.sin(x)[:, None] * jnp.cos(y)[None, :]

    fd = pnfindiff.findiff_along_axis(axis=1, xs=x, deriv=1, num=3)
    df_approx, _ = fd(f)

    assert jnp.allclose(df_approx, df_dy, atol=1e-4, rtol=1e-4)
