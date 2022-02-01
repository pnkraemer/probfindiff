"""Tests for FD coefficients_1d."""

import jax.numpy as jnp
import pytest_cases

from pnfindiff import coefficients_1d
from pnfindiff.utils import autodiff, kernel, kernel_zoo


def case_backward():
    return coefficients_1d.backward(x=0.0, deriv=1, acc=2, dx=1.0)


def case_forward():
    return coefficients_1d.forward(x=0.0, deriv=1, acc=2, dx=1.0)


def case_center():
    return coefficients_1d.center(x=0.0, deriv=1, acc=2, dx=1.0)


def case_from_offset():
    return coefficients_1d.from_offset(
        x=0.0, deriv=1, offset=jnp.arange(-2.0, 3.0), dx=1.0
    )


@pytest_cases.parametrize_with_cases("res", cases=".")
def test_coeff_shapes_and_cov_pos(res):
    a, b = res
    assert a.ndim == 1
    assert b.ndim == 0
    assert b > 0.0


def test_central_coefficients_polynomial():

    x, xs = jnp.array(0.0), jnp.array([-1.0, 0.0, 1.0])

    k = lambda x, y: kernel_zoo.polynomial(x, y, order=3)
    L = autodiff.compose(autodiff.deriv_scalar, autodiff.deriv_scalar)
    ks = kernel.differentiate(k=k, L=L)
    coeffs, unc_base = coefficients_1d.non_uniform_1d(x=x, xs=xs, ks=ks)

    assert jnp.allclose(coeffs, jnp.array([1.0, -2.0, 1.0]))
    assert jnp.allclose(unc_base, 0.0)
