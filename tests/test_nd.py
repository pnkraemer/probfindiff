"""Tests for properly multidimensional schemes."""


import jax
import jax.numpy as jnp

from probfindiff import collocation, nd, stencil
from probfindiff.utils import kernel, kernel_zoo


def test_from_grid():

    diffop = jax.jacfwd
    xs_1d = jnp.array([1.0, 2.0, 3.0])
    xs = stencil.multivariate(xs_1d, shape_input=(2,))
    assert xs.shape == (2, 2, 3)

    fx = jnp.einsum("abc,abc->ac", xs, xs)[:, None, :]
    assert fx.shape == (2, 1, 3)

    scheme = nd.from_grid(diffop, xs=xs)
    dfx, _ = nd.differentiate(fx, scheme=scheme)

    assert dfx.shape == (2,)
