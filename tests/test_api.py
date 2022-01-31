import jax.numpy as jnp

import pnfindiff


def test_findiff():
    x = jnp.linspace(1, 1.1, 10)
    f = jnp.sin(x)

    fd = pnfindiff.findiff(xs=x, deriv=1, num=3)
    df_approx = fd(f)

    assert jnp.allclose(df_approx, jnp.cos(x), atol=1e-4, rtol=1e-4)
