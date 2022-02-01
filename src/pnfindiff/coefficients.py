"""Top-level API."""

from functools import partial, reduce

import jax
import jax.numpy as jnp
import scipy.spatial

from pnfindiff import coefficients_1d

from .utils import autodiff, kernel, kernel_zoo


@jax.jit
def apply(f, *, coeffs, indices):
    """Apply a finite difference scheme to a vector of function evaluations."""
    weights, unc_base = coeffs
    dfx = jnp.einsum("nk,nk->n", weights, f[indices])
    return dfx, unc_base


def apply_along_axis(f, *, axis, coeffs, indices):
    """Apply a finite difference scheme along a specified axis."""
    fd = partial(apply, coeffs=coeffs, indices=indices)
    return jnp.apply_along_axis(fd, axis=axis, arr=f)


def derivative(*, xs, num=2):
    """Discretised first-order derivative."""
    return derivative_higher(xs=xs, deriv=1, num=num)


def derivative_higher(*, xs, deriv=1, num=2):
    """Discretised higher-order derivative."""
    neighbours, indices = _neighbours(num=num, xs=xs)

    ks = kernel.differentiate(
        k=kernel_zoo.exponentiated_quadratic,
        L=reduce(autodiff.compose, [autodiff.deriv_scalar] * deriv),
    )
    coeff_fun_batched = jax.jit(
        jax.vmap(partial(coefficients_1d.non_uniform_1d, ks=ks))
    )
    coeffs = coeff_fun_batched(x=xs, xs=neighbours)
    return coeffs, indices


def _neighbours(*, num, xs):
    tree = scipy.spatial.KDTree(data=xs.reshape((-1, 1)))
    _, indices = tree.query(x=xs.reshape((-1, 1)), k=num)
    neighbours = xs[indices]
    return neighbours, indices
