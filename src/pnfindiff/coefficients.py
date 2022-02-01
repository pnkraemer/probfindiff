"""Finite difference coefficients."""

from functools import partial, reduce
from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import scipy.spatial

from pnfindiff import collocation
from pnfindiff.typing import ArrayLike

from .utils import autodiff, kernel, kernel_zoo


@jax.jit
def apply(
    f: ArrayLike, *, coeffs: Tuple[ArrayLike, ArrayLike], indices: ArrayLike
) -> ArrayLike:
    """Apply a finite difference scheme to a vector of function evaluations.

    Parameters
    ----------
    f
        Array of function evaluated to be differentiated numerically. Shape ``(n,)``.
    coeffs
        PN finite difference coefficients. Shapes `` (n,k), (n,)``.
    indices
        Indices of the neighbours that shall be used for each derivative. Shape ``(n,k)``.


    Returns
    -------
    :
        Finite difference approximation and the corresponding base-uncertainty. Shapes `` (n,), (n,)``.
    """
    weights, unc_base = coeffs
    dfx = jnp.einsum("nk,nk->n", weights, f[indices])
    return dfx, unc_base


def apply_along_axis(
    f: ArrayLike, *, axis: int, coeffs: Tuple[ArrayLike, ArrayLike], indices: ArrayLike
) -> ArrayLike:
    """Apply a finite difference scheme along a specified axis.

    Parameters
    ----------
    f
        Array of function evaluated to be differentiated numerically. Shape ``(..., n, ...)``.
    coeffs
        PN finite difference coefficients. Shapes `` (n,k), (n,)``.
    indices
        Indices of the neighbours that shall be used for each derivative. Shape ``(n,k)``.


    Returns
    -------
    :
        Finite difference approximation and the corresponding base-uncertainty. Shapes `` (..., n, ...), (n,)``.
    """
    fd = partial(apply, coeffs=coeffs, indices=indices)
    return jnp.apply_along_axis(fd, axis=axis, arr=f)


def derivative(
    *, xs: ArrayLike, num: int = 2
) -> Tuple[Tuple[ArrayLike, ArrayLike], ArrayLike]:
    """Discretised first-order derivative.

    Parameters
    ----------
    xs
        List of grid-points on which the to-be-seen function is evaluated. This is not the list of neighbours to be used.
        Shape ``(n,k)``.
    num
        Number of neighbours to be used for numerical differentiation for each point.

    Returns
    -------
    :
        Tuple ``((a, b), c)`` of a tuple ``(a, b)`` containing the finite difference coefficients ``a`` and the base uncertainty ``b``,
        and the indices ``c`` of the neighbours of each point in the point set.

    """
    return derivative_higher(xs=xs, deriv=1, num=num)


def derivative_higher(
    *, xs: ArrayLike, deriv: int = 1, num: int = 2
) -> Tuple[Tuple[ArrayLike, ArrayLike], ArrayLike]:
    """Discretised higher-order derivative.

    Parameters
    ----------
    xs
        List of grid-points on which the to-be-seen function is evaluated. This is not the list of neighbours to be used.
        Shape ``(n,k)``.
    num
        Number of neighbours to be used for numerical differentiation for each point.

    Returns
    -------
    :
        Tuple ``((a, b), c)`` of a tuple ``(a, b)`` containing the finite difference coefficients ``a`` and the base uncertainty ``b``,
        and the indices ``c`` of the neighbours of each point in the point set.
    """
    neighbours, indices = _neighbours(num=num, xs=xs)

    ks = kernel.differentiate(
        k=kernel_zoo.exponentiated_quadratic,
        L=reduce(autodiff.compose, [autodiff.deriv_scalar] * deriv),
    )
    coeff_fun_batched = jax.jit(
        jax.vmap(partial(collocation.non_uniform_nd, ks=ks))
    )  # type: Callable[..., Tuple[ArrayLike, ArrayLike]]
    coeffs = coeff_fun_batched(x=xs[..., None], xs=neighbours[..., None])
    return coeffs, indices


def _neighbours(*, num: int, xs: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    tree = scipy.spatial.KDTree(data=xs.reshape((-1, 1)))
    _, indices = tree.query(x=xs.reshape((-1, 1)), k=num)
    neighbours = xs[indices]
    return neighbours, indices
