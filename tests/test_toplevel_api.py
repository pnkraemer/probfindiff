"""Tests for the top-level API."""
import jax
import jax.numpy as jnp
import pytest_cases

import pnfindiff
from pnfindiff import collocation
from pnfindiff.utils import autodiff
from pnfindiff.utils import kernel as kernel_module
from pnfindiff.utils import kernel_zoo


@pytest_cases.parametrize("kernel", [kernel_zoo.exponentiated_quadratic, None])
@pytest_cases.parametrize("noise_variance", [1e-5, 0.0])
def case_backward(kernel, noise_variance):
    return pnfindiff.backward(
        order_derivative=1,
        order_method=3,
        dx=0.05,
        kernel=kernel,
        noise_variance=noise_variance,
    )


@pytest_cases.parametrize("kernel", [kernel_zoo.exponentiated_quadratic, None])
@pytest_cases.parametrize("noise_variance", [1e-5, 0.0])
def case_forward(kernel, noise_variance):
    return pnfindiff.forward(
        order_derivative=1,
        order_method=3,
        dx=0.05,
        kernel=kernel,
        noise_variance=noise_variance,
    )


@pytest_cases.parametrize("kernel", [kernel_zoo.exponentiated_quadratic, None])
@pytest_cases.parametrize("noise_variance", [1e-5, 0.0])
@pytest_cases.case(tags=("central",))
def case_central(kernel, noise_variance):
    return pnfindiff.central(
        order_derivative=1,
        order_method=2,
        dx=0.1,
        kernel=kernel,
        noise_variance=noise_variance,
    )


@pytest_cases.parametrize("kernel", [kernel_zoo.exponentiated_quadratic, None])
@pytest_cases.parametrize("noise_variance", [1e-5, 0.0])
def case_from_grid(kernel, noise_variance):
    xs = 0.02 * jnp.arange(-2.0, 3.0)
    scheme = pnfindiff.from_grid(
        order_derivative=1, xs=xs, kernel=kernel, noise_variance=noise_variance
    )
    return scheme, xs


@pytest_cases.parametrize_with_cases("scheme, xs", cases=".")
def test_scheme_types(scheme, xs):

    assert isinstance(scheme, pnfindiff.FiniteDifferenceScheme)
    assert isinstance(xs, jnp.ndarray)


@pytest_cases.parametrize_with_cases("scheme, _", cases=".")
def test_scheme_shapes(scheme, _):
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


@pytest_cases.parametrize_with_cases("scheme, xs", cases=".", has_tag=("central",))
def test_multivariate(scheme, xs):

    # A simple function R^d -> R
    f, d = lambda z: (z[0] + 1) ** 2 + (z[1] - 1) ** 2 + (z[2] - 1) ** 2, 3

    # Some point x in R^d
    x = jnp.array([1.0, 2.0, 3.0])

    # The gradient takes values in R^d
    df = jax.grad(f)

    scheme, xs = pnfindiff.multivariate(scheme_1d=scheme, xs_1d=xs, shape_input=(3,))
    assert xs.ndim == 3
    assert xs.shape[0] == xs.shape[1] == 3

    xs_shifted = x[None, :, None] + xs
    dfx, _ = pnfindiff.differentiate(f(xs_shifted), scheme=scheme)

    assert dfx.shape == df(x).shape == (d,)
    assert jnp.allclose(dfx, df(x), rtol=1e-2, atol=1e-2)


def test_central_coefficients_polynomial():
    """For polynomials, the usual [1, -2, 1] coefficients should emerge."""
    x, xs = jnp.array(0.0), jnp.array([-1.0, 0.0, 1.0])

    k = lambda x, y: kernel_zoo.polynomial(x, y, p=jnp.ones((3,)))
    L = autodiff.compose(autodiff.derivative, autodiff.derivative)
    ks = kernel_module.differentiate(k=k, L=L)
    coeffs, unc_base = collocation.non_uniform_nd(
        x=jnp.array([x]), xs=xs[:, None], ks=ks, noise_variance=0.0
    )

    assert jnp.allclose(coeffs, jnp.array([1.0, -2.0, 1.0]))
    assert jnp.allclose(unc_base, 0.0)
