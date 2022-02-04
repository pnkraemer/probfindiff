"""Tests for properly multidimensional schemes."""


import jax
import jax.numpy as jnp

from probfindiff import collocation, nd
from probfindiff.utils import kernel, kernel_zoo


def test_from_grid():

    diffop = jax.grad
    xs = jnp.asarray(
        [[[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]], [[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]]]
    )
    assert xs.shape == (2, 2, 3)

    fx = jnp.einsum("abc,abc->ac", xs, xs)
    assert fx.shape == (2, 3)

    scheme = nd.from_grid(diffop, xs=xs)
    dfx, _ = nd.differentiate(fx, scheme=scheme)

    assert dfx.shape == (2,)


def test_coeffs():
    k = kernel_zoo.exponentiated_quadratic
    k_batch, lk_batch, llk_batch = kernel.differentiate(k, L=jax.jacfwd)

    d, num_ys = 3, 7
    zeros = jnp.zeros((1, d))
    ys = jnp.arange(1, 1 + d * num_ys, dtype=float).reshape((num_ys, d))

    K = k_batch(ys, ys.T)
    LK = lk_batch(zeros, ys.T)
    LLK = llk_batch(zeros, zeros.T)
    print(K.shape, LK.shape, LLK.shape)
    collocation.unsymmetric(K=K, LK0=LK, LLK=LLK, noise_variance=1.0)
    assert False
