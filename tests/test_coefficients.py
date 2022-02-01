"""Tests for FD coefficients."""

import jax.numpy as jnp
import pytest_cases

from pnfindiff import coefficients
from pnfindiff.utils import diffop, kernel


def case_backward():
    return coefficients.backward(x=0.0, deriv=1, acc=2, dx=1.0)


def case_forward():
    return coefficients.forward(x=0.0, deriv=1, acc=2, dx=1.0)


def case_center():
    return coefficients.center(x=0.0, deriv=1, acc=2, dx=1.0)


def case_from_offset():
    return coefficients.from_offset(
        x=0.0, deriv=1, offset=jnp.arange(-2.0, 3.0), dx=1.0
    )


def case_non_uniform_1d():
    ks = kernel.differentiate(kernel.exp_quad()[1], L=diffop.deriv_scalar)
    x = 0.5
    xs = jnp.array([0.5, 0.3, 0.1])
    return coefficients.non_uniform_1d(x=x, xs=xs, ks=ks)


def case_non_uniform_2d():
    L = diffop.laplace
    k_batch, k = kernel.exp_quad()
    lk_batch, lk = kernel.batch_gram(L(k, argnums=0))
    llk_batch, _ = kernel.batch_gram(L(lk, argnums=1))

    x = 0.5 * jnp.ones((1,))
    xs = jnp.array([0.5, 0.3, 0.1, 0.2, 0.4]).reshape((-1, 1))
    return coefficients.non_uniform_nd(x=x, xs=xs, ks=(k_batch, lk_batch, llk_batch))


@pytest_cases.parametrize_with_cases("res", cases=".")
def test_coeff_shapes_and_cov_pos(res):
    a, b = res
    assert a.ndim == 1
    assert b.ndim == 0
    assert b > 0.0


def test_central_coefficients_polynomial():

    x, xs = jnp.array(0.0), jnp.array([-1.0, 0.0, 1.0])

    _, k = kernel.polynomial(order=3)

    L = diffop.compose(diffop.deriv_scalar, diffop.deriv_scalar)
    ks = kernel.differentiate(k=k, L=L)
    coeffs, unc_base = coefficients.non_uniform_1d(x=x, xs=xs, ks=ks)

    assert jnp.allclose(coeffs, jnp.array([1.0, -2.0, 1.0]))
    assert jnp.allclose(unc_base, 0.0)
