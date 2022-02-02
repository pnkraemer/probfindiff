"""Tests for the top-level API."""
import jax.numpy as jnp
import pytest_cases

import pnfindiff
from pnfindiff import collocation
from pnfindiff.utils import autodiff
from pnfindiff.utils import kernel as kernel_module
from pnfindiff.utils import kernel_zoo


@pytest_cases.parametrize("kernel", [kernel_zoo.exponentiated_quadratic, None])
def case_backward(kernel):
    return pnfindiff.backward(order_derivative=1, order_method=2, dx=1.0, kernel=kernel)


@pytest_cases.parametrize("kernel", [kernel_zoo.exponentiated_quadratic, None])
def case_forward(kernel):
    return pnfindiff.forward(order_derivative=1, order_method=2, dx=1.0, kernel=kernel)


@pytest_cases.parametrize("kernel", [kernel_zoo.exponentiated_quadratic, None])
def case_center(kernel):
    return pnfindiff.central(order_derivative=1, order_method=2, dx=1.0, kernel=kernel)


@pytest_cases.parametrize("kernel", [kernel_zoo.exponentiated_quadratic, None])
@pytest_cases.parametrize("noise_variance", [1.0, 0.0])
def case_from_grid(kernel, noise_variance):
    xs = jnp.arange(-2.0, 3.0)
    scheme = pnfindiff.from_grid(
        order_derivative=1, xs=xs, kernel=kernel, noise_variance=noise_variance
    )
    return scheme, xs


@pytest_cases.parametrize_with_cases("scheme, xs", cases=".")
def test_coeff_type(scheme, xs):

    assert isinstance(scheme, pnfindiff.FiniteDifferenceScheme)
    assert isinstance(xs, jnp.ndarray)


@pytest_cases.parametrize_with_cases("scheme, _", cases=".")
def test_coeff_shapes(scheme, _):
    assert scheme.weights.ndim == 1
    assert scheme.covs_marginal.ndim == 0


@pytest_cases.parametrize_with_cases("scheme, xs", cases=".")
def test_differentiate(scheme, xs):
    fx = jnp.sin(xs)
    dfx_approx, _ = pnfindiff.differentiate(fx, scheme=scheme)
    assert dfx_approx.shape == ()


@pytest_cases.parametrize_with_cases("scheme, xs", cases=".")
def test_differentiate_along_axis(scheme, xs):
    fx = jnp.sin(xs)[:, None] * jnp.cos(xs)[None, :]
    dfx_approx, _ = pnfindiff.differentiate_along_axis(fx, axis=1, scheme=scheme)
    assert dfx_approx.ndim == 1


def test_central_coefficients_polynomial():

    x, xs = jnp.array(0.0), jnp.array([-1.0, 0.0, 1.0])

    k = lambda x, y: kernel_zoo.polynomial(x, y, p=jnp.ones((3,)))
    L = autodiff.compose(autodiff.derivative, autodiff.derivative)
    ks = kernel_module.differentiate(k=k, L=L)
    coeffs, unc_base = collocation.non_uniform_nd(
        x=jnp.array([x]), xs=xs[:, None], ks=ks, noise_variance=0.0
    )

    assert jnp.allclose(coeffs, jnp.array([1.0, -2.0, 1.0]))
    assert jnp.allclose(unc_base, 0.0)
