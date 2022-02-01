"""Finite difference coefficients."""

from functools import partial, reduce
from typing import Tuple

import jax
import jax.numpy as jnp
import scipy.spatial

from pnfindiff import coefficients_1d
from pnfindiff.typing import ArrayLike

from .utils import autodiff, kernel, kernel_zoo


@jax.jit
def apply(
    f: ArrayLike, *, coeffs: Tuple[ArrayLike, ArrayLike], indices: ArrayLike
) -> ArrayLike:
    """Apply a finite difference scheme to a vector of function evaluations."""
    weights, unc_base = coeffs
    dfx = jnp.einsum("nk,nk->n", weights, f[indices])
    return dfx, unc_base


def apply_along_axis(
    f: ArrayLike, *, axis: int, coeffs: Tuple[ArrayLike, ArrayLike], indices: ArrayLike
) -> ArrayLike:
    """Apply a finite difference scheme along a specified axis."""
    fd = partial(apply, coeffs=coeffs, indices=indices)
    return jnp.apply_along_axis(fd, axis=axis, arr=f)


def derivative(
    *, xs: ArrayLike, num: int = 2
) -> Tuple[Tuple[ArrayLike, ArrayLike], ArrayLike]:
    """Discretised first-order derivative."""
    return derivative_higher(xs=xs, deriv=1, num=num)


def derivative_higher(
    *, xs: ArrayLike, deriv: int = 1, num: int = 2
) -> Tuple[Tuple[ArrayLike, ArrayLike], ArrayLike]:
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


def _neighbours(*, num: int, xs: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    tree = scipy.spatial.KDTree(data=xs.reshape((-1, 1)))
    _, indices = tree.query(x=xs.reshape((-1, 1)), k=num)
    neighbours = xs[indices]
    return neighbours, indices
