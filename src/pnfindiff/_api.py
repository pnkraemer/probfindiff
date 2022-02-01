"""Top-level API."""

from functools import partial, reduce

import jax
import jax.numpy as jnp
import scipy.spatial

from . import coefficients
from .utils import autodiff, kernel

__all__ = ["findiff_along_axis", "findiff"]


def findiff_along_axis(*, axis, **kwargs):
    """Compute finite-difference-approximations along an axis.

    In other words, approximate partial derivatives.
    """
    fd = findiff(**kwargs)

    def fd_along_axis(fx):
        return jnp.apply_along_axis(fd, axis=axis, arr=fx)

    return fd_along_axis


def findiff(*, xs, deriv=1, num=2):
    """Finite differences."""

    neighbours, indices = _neighbours(num=num, xs=xs)

    ks = kernel.differentiate(
        k=kernel.exp_quad()[1],
        L=reduce(autodiff.compose, [autodiff.deriv_scalar] * deriv),
    )
    coeff_fun_batched = jax.jit(jax.vmap(partial(coefficients.non_uniform_1d, ks=ks)))
    coeffs = coeff_fun_batched(x=xs, xs=neighbours)

    @jax.jit
    def diff(fx):
        weights, unc_base = coeffs
        dfx = jnp.einsum("nk,nk->n", weights, fx[indices])
        return dfx, unc_base

    return diff


def _neighbours(*, num, xs):
    tree = scipy.spatial.KDTree(data=xs.reshape((-1, 1)))
    _, indices = tree.query(x=xs.reshape((-1, 1)), k=num)
    neighbours = xs[indices]
    return neighbours, indices
