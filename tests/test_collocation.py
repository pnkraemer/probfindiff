"""Tests for global collocation."""

import jax
import jax.numpy as jnp
import pytest

from pnfindiff import collocation, kernel


@pytest.fixture
def ks():
    def k(x, y):
        return jnp.exp(-(x - y).dot(x - y)).squeeze()

    def my_grad(fun, argnums=0):
        _assure_scalar_fn = lambda *args, **kwargs: fun(*args, **kwargs).squeeze()
        return jax.grad(_assure_scalar_fn, argnums=argnums)

    lk = my_grad(k, argnums=0)
    llk = my_grad(lk, argnums=1)

    k = kernel.vmap_gram(k)
    lk = kernel.vmap_gram(lk)
    llk = kernel.vmap_gram(llk)

    return k, lk, llk


@pytest.fixture
def Ks(ks, num_xs):
    xs = jnp.arange(10, dtype=float).reshape((10, 1))

    k, lk, llk = ks
    K = k(xs, xs.T).squeeze()
    LK = lk(xs, xs.T).squeeze()
    LLK = llk(xs, xs.T).squeeze()
    return K, LK, LLK


@pytest.mark.parametrize("num_xs", (10,))
def test_unsymmetric(Ks, num_xs):

    K, LK, LLK = Ks
    weights, unc_base = collocation.unsymmetric(K=K, LK=LK, LLK=LLK)

    assert weights.shape == (num_xs, num_xs)
    assert unc_base.shape == (num_xs, num_xs)


@pytest.mark.parametrize("num_xs", (10,))
def test_symmetric(Ks, num_xs):

    K, LK, LLK = Ks
    weights, unc_base = collocation.symmetric(K=K, LK=LK, LLK=LLK)

    assert weights.shape == (num_xs, num_xs)
    assert unc_base.shape == (num_xs, num_xs)
