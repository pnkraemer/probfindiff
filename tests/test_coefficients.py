"""Tests for FD coefficients."""


from pnfindiff import coefficients
import pytest


def test_backward():

    with pytest.raises(NotImplementedError):
        coefficients.backward(deriv=1, acc=2, dx=0.1)


def test_scattered():
    k = lambda x, y: jnp.exp(-jnp.linalg.norm(x - y) ** 2.0 / 2.0)
    lk = jnp.grad(k, argnums=0)
    llk = jnp.grad(lk, argnums=1)

    x = jnp.array([0.5])
    xs = jnp.array([0.5, 0.0, 1.0])
    a, b = coefficients.scattered(pt=x, neighbours=xs, kernel=(k, lk, llk))
    assert a.shape == (3,)
    assert b.shape == (3,)
    assert jnp.all(b > 0.0)
