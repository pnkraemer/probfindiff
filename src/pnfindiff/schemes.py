"""Fancy finite difference coefficients."""

from functools import partial, reduce
from typing import Callable, Tuple

import jax
import scipy.spatial

import pnfindiff
from pnfindiff import collocation
from pnfindiff.typing import ArrayLike

from .utils import autodiff, kernel, kernel_zoo


def derivative(*, xs: ArrayLike, num: int = 2) -> pnfindiff.FiniteDifferenceScheme:
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
) -> pnfindiff.FiniteDifferenceScheme:
    """Discretised higher-order derivative.

    Parameters
    ----------
    xs
        List of grid-points on which the to-be-seen function is evaluated. This is not the list of neighbours to be used.
        Shape ``(n,k)``.
    deriv
        Order of the desired derivative.
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
        L=reduce(autodiff.compose, [autodiff.derivative] * deriv),
    )
    coeff_fun_batched = jax.jit(
        jax.vmap(partial(collocation.non_uniform_nd, ks=ks))
    )  # type: Callable[..., Tuple[ArrayLike, ArrayLike]]
    coeffs = coeff_fun_batched(x=xs[..., None], xs=neighbours[..., None])
    return pnfindiff.FiniteDifferenceScheme(
        *coeffs, indices, order_method=xs.shape[0] - deriv, order_derivative=deriv
    )


def _neighbours(*, num: int, xs: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
    tree = scipy.spatial.KDTree(data=xs.reshape((-1, 1)))
    _, indices = tree.query(x=xs.reshape((-1, 1)), k=num)
    neighbours = xs[indices]
    return neighbours, indices
