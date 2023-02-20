"""Tests for global collocation."""

import jax
import jax.numpy as jnp
import pytest

from probfindiff import collocation
from probfindiff.utils import autodiff, kernel, kernel_zoo


@pytest.fixture(name="ks")
def fixture_ks():
    L = autodiff.derivative
    k_batch, k = kernel.batch_gram(kernel_zoo.exponentiated_quadratic)
    lk_batch, lk = kernel.batch_gram(L(k, argnums=0))
    llk_batch, _ = kernel.batch_gram(L(lk, argnums=1))

    return k_batch, lk_batch, llk_batch


@pytest.fixture(name="Ks")
def fixture_Ks(ks, num_xs):
    xs = jnp.arange(num_xs, dtype=float).reshape((num_xs, 1))

    k, lk, llk = ks
    K = k(xs, xs.T).reshape((num_xs, num_xs))
    LK = lk(xs, xs.T).reshape((num_xs, num_xs))
    LLK = llk(xs, xs.T).reshape((num_xs, num_xs))
    return K, LK, LLK


@pytest.mark.parametrize("num_xs", (1, 3))
def test_unsymmetric(Ks, num_xs):
    K, LK, LLK = Ks
    weights, unc_base = collocation.unsymmetric(
        K=K, LK0=LK, LLK=LLK, noise_variance=0.1
    )

    assert weights.shape == (num_xs, num_xs)
    assert unc_base.shape == (num_xs, num_xs)


@pytest.mark.parametrize("num_xs", (1, 3))
def test_symmetric(Ks, num_xs):
    K, LK, LLK = Ks
    weights, unc_base = collocation.symmetric(
        K=K, LK1=LK.T, LLK=LLK, noise_variance=0.1
    )

    assert weights.shape == (num_xs, num_xs)
    assert unc_base.shape == (num_xs, num_xs)


def test_batch_shape_correct():
    """Collocation weights and uncertainties provide the correct shape for batch covariances.

    The weights need shape (1,d,1,n), because the FD nodes have shape
    (output_shape, input_shape, input_shape, weight_1d_shape)
    which then gets batch-evaluated to
    (output_shape, input_shape, output_shape, weight_1d_shape).
    For a function that maps (d,) to (1,) (gradient-style Jacobians)
    the nodes have shape (1,d,d,n) and the function evaluations
    have shape (1,d,1,n). The FD scheme then matrix-multiplies them to
    shape (1,d,1,1), and the final two axes are discarded
    (because they have served their purpose) which gives
    the correctly (1,d)-shaped Jacobian with (d,d)-shaped uncertainty.
    """
    k = kernel_zoo.exponentiated_quadratic
    k_batch, lk_batch, llk_batch = kernel.differentiate(k, L=jax.jacfwd)

    d, num_ys = 3, 7
    zeros = jnp.zeros((1, d))
    ys = jnp.arange(1, 1 + d * num_ys, dtype=float).reshape((num_ys, d))

    # Promote ndim of Gram matrices to size of LLK
    K = k_batch(ys, ys.T)[None, None, ...]
    LK = lk_batch(zeros, ys.T)[None, ...]
    LLK = llk_batch(zeros, zeros.T)

    weights, unc = collocation.unsymmetric(K=K, LK0=LK, LLK=LLK, noise_variance=1.0)
    assert weights.shape == (1, d, 1, num_ys)
    assert unc.shape == (d, d, 1, 1)
