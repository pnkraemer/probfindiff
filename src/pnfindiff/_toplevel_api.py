"""Top-level API."""

import functools
from collections import namedtuple
from functools import partial
from typing import Any, Optional, Tuple

import jax
import jax.numpy as jnp

from pnfindiff import collocation
from pnfindiff.typing import ArrayLike, KernelFunctionLike
from pnfindiff.utils import autodiff
from pnfindiff.utils import kernel as kernel_module
from pnfindiff.utils import kernel_zoo

FiniteDifferenceScheme = namedtuple(
    "FiniteDifferenceScheme",
    (
        "weights",
        "covs_marginal",
        "order_derivative",
    ),
)
"""Finite difference schemes.

A finite difference scheme consists of a weight-vector, marginal covariances, and offset indices.
"""


@jax.jit
def differentiate(fx: ArrayLike, *, scheme: FiniteDifferenceScheme) -> ArrayLike:
    """Apply a finite difference scheme to a vector of function evaluations.

    Parameters
    ----------
    fx
        Array of function evaluated to be differentiated numerically. Shape ``(n,)``.
    scheme
        PN finite difference schemes.


    Returns
    -------
    :
        Finite difference approximation and the corresponding base-uncertainty. Shapes `` (n,), (n,)``.
    """
    weights, unc_base, *_ = scheme
    dfx = jnp.einsum("...k,...k->...", weights, fx)
    return dfx, unc_base


def differentiate_along_axis(
    fx: ArrayLike, *, axis: int, scheme: FiniteDifferenceScheme
) -> ArrayLike:
    """Apply a finite difference scheme along a specified axis.

    Parameters
    ----------
    fx
        Array of function evaluated to be differentiated numerically. Shape ``(..., n, ...)``.
    axis
        Axis along which the scheme should be applied.
    scheme
        PN finite difference schemes.


    Returns
    -------
    :
        Finite difference approximation and the corresponding base-uncertainty. Shapes `` (..., n, ...), (n,)``.
    """

    fd = partial(differentiate, scheme=scheme)
    return jnp.apply_along_axis(fd, axis=axis, arr=fx)


_DEFAULT_NOISE_VARIANCE = 1e-14
"""We might change the defaults again soon, so we encapsulate it into a variable..."""


@functools.partial(
    jax.jit, static_argnames=("order_derivative", "order_method", "kernel")
)
def backward(
    *,
    dx: float,
    order_derivative: int = 1,
    order_method: int = 2,
    kernel: Optional[KernelFunctionLike] = None,
    noise_variance: float = _DEFAULT_NOISE_VARIANCE,
) -> Any:
    """Backward coefficients in 1d.

    Parameters
    ----------
    dx
        Step-size.
    order_derivative
        Order of the derivative.
    order_method
        Desired accuracy.
    kernel
        Kernel function. Defines the function-model.
    noise_variance
        Variance of the observation noise.

    Returns
    -------
    :
        Finite difference coefficients and base uncertainty.
    """
    offset = -jnp.arange(order_derivative + order_method, step=1)
    grid = offset * dx
    scheme = from_grid(
        xs=grid,
        order_derivative=order_derivative,
        kernel=kernel,
        noise_variance=noise_variance,
    )
    return scheme, grid


@functools.partial(
    jax.jit, static_argnames=("order_derivative", "order_method", "kernel")
)
def forward(
    *,
    dx: float,
    order_derivative: int = 1,
    order_method: int = 2,
    kernel: Optional[KernelFunctionLike] = None,
    noise_variance: float = _DEFAULT_NOISE_VARIANCE,
) -> Any:
    """Forward coefficients in 1d.

    Parameters
    ----------
    dx
        Step-size.
    order_derivative
        Order of the derivative.
    order_method
        Desired accuracy.
    kernel
        Kernel function. Defines the function-model.
    noise_variance
        Variance of the observation noise.

    Returns
    -------
    :
        Finite difference coefficients and base uncertainty.
    """
    offset = jnp.arange(order_derivative + order_method, step=1)
    grid = offset * dx
    scheme = from_grid(
        xs=grid,
        order_derivative=order_derivative,
        kernel=kernel,
        noise_variance=noise_variance,
    )
    return scheme, grid


@functools.partial(
    jax.jit, static_argnames=("order_derivative", "order_method", "kernel")
)
def central(
    *,
    dx: float,
    order_derivative: int = 1,
    order_method: int = 2,
    kernel: Optional[KernelFunctionLike] = None,
    noise_variance: float = _DEFAULT_NOISE_VARIANCE,
) -> Any:
    """Central coefficients in 1d.

    Parameters
    ----------
    dx
        Step-size.
    order_derivative
        Order of the derivative.
    order_method
        Desired accuracy.
    kernel
        Kernel function. Defines the function-model.
    noise_variance
        Variance of the observation noise.

    Returns
    -------
    :
        Finite difference coefficients and base uncertainty.
    """
    num_central = (2 * ((order_derivative + 1.0) / 2.0) // 2) - 1 + order_method
    num_side = num_central // 2
    offset = jnp.arange(-num_side, num_side + 1, step=1)
    grid = offset * dx
    scheme = from_grid(
        xs=grid,
        order_derivative=order_derivative,
        kernel=kernel,
        noise_variance=noise_variance,
    )
    return scheme, grid


@functools.partial(jax.jit, static_argnames=("order_derivative", "kernel"))
def from_grid(
    *,
    xs: ArrayLike,
    order_derivative: int = 1,
    kernel: Optional[KernelFunctionLike] = None,
    noise_variance: float = _DEFAULT_NOISE_VARIANCE,
) -> Any:
    """Finite difference coefficients based on an array of offset indices.

    Parameters
    ----------
    order_derivative
        Order of the derivative.
    xs
        Grid. Shape ``(n,)``.
    kernel
        Kernel function. Defines the function-model.
    noise_variance
        Variance of the observation noise.

    Returns
    -------
    :
        Finite difference coefficients and base uncertainty.
    """
    if kernel is None:
        kernel = _default_kernel(min_order=xs.shape[0])
    L = functools.reduce(autodiff.compose, [autodiff.derivative] * order_derivative)
    ks = kernel_module.differentiate(k=kernel, L=L)

    x = jnp.zeros_like(xs[0])
    weights, cov_marginal = collocation.non_uniform_nd(
        x=x[..., None],
        xs=xs[..., None],
        ks=ks,
        noise_variance=noise_variance,
    )
    scheme = FiniteDifferenceScheme(
        weights,
        cov_marginal,
        order_derivative=order_derivative,
    )
    return scheme


def _default_kernel(*, min_order: int) -> KernelFunctionLike:
    return functools.partial(kernel_zoo.polynomial, p=jnp.ones((min_order,)))


@functools.partial(jax.jit, static_argnames=("shape_input",))
def multivariate(
    scheme_1d: FiniteDifferenceScheme, xs_1d: ArrayLike, shape_input: Tuple[int]
) -> Tuple[FiniteDifferenceScheme, ArrayLike]:
    r"""Turn a univariate finite-difference scheme into a multivariate scheme.

    Parameters
    ----------
    scheme_1d
        Input finite-difference scheme in 1d.
    xs_1d
        Input finite-difference grid/stencil in 1d.
    shape_input
        Input dimension of the to-be-differentiated function as a shape-tuple.
        If the goal is the gradient of an `n`-dimensional function, ``shape_input=(n,)``.

    Returns
    -------
    :
        Tuple of a new scheme and a new grid.
        The new scheme consists of a 1d set of weights.
        The new grid has shape ``(n, n, c)``.
        The first two array-dimensions are determined by the ``shape_input`` (``n``).
        The first one refers to each partial derivative index.
        The second one refers to the *dimension of each nd-stencil-element*.
        In other words, ``output[0, :, :]`` is an ``(n, c)``-shaped array
        containing `c` `(n,)` shaped inputs to the function.
        (The final array-dimension ``c`` describes the length of the provided stencil;
        there are ``c`` elements in ``stencil_1d``.)
        To evaluate your function correctly on the new grid,
        outer-loop over the zeroth index, and inner loop over the third index:
        ``for xs_per_deriv in xs: for fun_input in xs_per_deriv.T: f(fun_input)``
        and stack accordingly, i.e., so that the resulting function evaluation
        has shape ``(n,c)``.

    Examples
    --------
    >>> from pnfindiff import central
    >>> scheme_1d, xs_1d = central(dx=1.)
    >>> print(xs_1d)
    [-1.  0.  1.]

    >>> scheme, xs = multivariate(scheme_1d=scheme_1d, xs_1d=xs_1d, shape_input=(2,))
    >>> print(xs.shape)
    (2, 2, 3)
    >>> print(xs)
    [[[-1.  0.  1.]
      [ 0.  0.  0.]]
     [[ 0.  0.  0.]
      [-1.  0.  1.]]]
    >>> print(scheme.weights)
    [-0.5  0.   0.5]
    """
    assert len(shape_input) == 1

    xs_full = _stencils_for_all_partial_derivatives(
        shape_input=shape_input, stencil_1d=xs_1d
    )

    return scheme_1d, xs_full


@functools.partial(jax.jit, static_argnames=("shape_input",))
def _stencils_for_all_partial_derivatives(
    *, stencil_1d: ArrayLike, shape_input: Tuple[int]
) -> ArrayLike:
    """Compute the stencils for all partial derivatives.

    Parameters
    ----------
    stencil_1d
        Finite difference stencil (i.e., the grid) to be turned into a stencil for higher derivatives.
    shape_input
        The "shape" of the input-domain of the function.
        At the moment, only ``shape=(n,)`` is supported, where ``n`` is the dimension of the input space.
        For a function :math:`f: R^5\rightarrow R`, ``n=5``.

    Returns
    -------
    :
        Batch/Stack of the stencils for each partial derivative. The shape is ``(n, n, c)``.
        The first two array-dimensions are determined by the ``shape_input`` (``n``).
        The first one refers to the partial derivative index.
        The second one refers to the *dimension of each nd-stencil-element*.
        In other words, ``output[0, :, :]`` is an ``(n, c)``-shaped array
        containing `c` `(n,)` shaped inputs to the function.
        (The final array-dimension ``c`` describes the length of the provided stencil;
        there are ``c`` elements in ``stencil_1d``.)

    Examples
    --------
    >>> _stencils_for_all_partial_derivatives(stencil_1d=jnp.array([1, 2, 3]), shape_input=(2,))
    DeviceArray([[[1, 2, 3],
                  [0, 0, 0]],
                 [[0, 0, 0],
                  [1, 2, 3]]], dtype=int32)

    >>> _stencils_for_all_partial_derivatives(stencil_1d=jnp.array([1, 2]), shape_input=(4,))
    DeviceArray([[[1, 2],
                  [0, 0],
                  [0, 0],
                  [0, 0]],
                 [[0, 0],
                  [1, 2],
                  [0, 0],
                  [0, 0]],
                 [[0, 0],
                  [0, 0],
                  [1, 2],
                  [0, 0]],
                 [[0, 0],
                  [0, 0],
                  [0, 0],
                  [1, 2]]], dtype=int32)
    """
    return jnp.stack(
        [
            _stencil_for_ith_partial_derivative(
                stencil_1d_as_row_matrix=stencil_1d[None, ...],
                i=i,
                dimension=shape_input[0],
            )
            for i in range(shape_input[0])
        ]
    )


@functools.partial(
    jax.jit,
    static_argnames=(
        "dimension",
        "i",
    ),
)
def _stencil_for_ith_partial_derivative(
    *, stencil_1d_as_row_matrix: ArrayLike, i: int, dimension: int
) -> ArrayLike:
    """Compute the stencil for the ith partial derivative.

    This is done by padding the 1d stencil into zeros according to the index ``i`` and the spatial dimension.
    It constitutes a stencil for a partial derivative, because such stencils are only affected by
    changes to the input along _a single_ dimension. The others remain unaffected, hence the zeros.

    Examples
    --------
    >>> _stencil_for_ith_partial_derivative(stencil_1d_as_row_matrix=jnp.array([[1, 2, 3]]), i=0, dimension=4)
    DeviceArray([[1, 2, 3],
                 [0, 0, 0],
                 [0, 0, 0],
                 [0, 0, 0]], dtype=int32)
    >>> _stencil_for_ith_partial_derivative(stencil_1d_as_row_matrix=jnp.array([[1, 2, 3]]), i=2, dimension=4)
    DeviceArray([[0, 0, 0],
                 [0, 0, 0],
                 [1, 2, 3],
                 [0, 0, 0]], dtype=int32)
    >>> _stencil_for_ith_partial_derivative(stencil_1d_as_row_matrix=jnp.array([[1, 2, 3, 4, 5]]), i=1, dimension=8)
    DeviceArray([[0, 0, 0, 0, 0],
                 [1, 2, 3, 4, 5],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0]], dtype=int32)
    >>> _stencil_for_ith_partial_derivative(stencil_1d_as_row_matrix=jnp.array([[1, 2, 3, 4, 5]]), i=7, dimension=8)
    DeviceArray([[0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [1, 2, 3, 4, 5]], dtype=int32)
    """
    return jnp.pad(stencil_1d_as_row_matrix, pad_width=((i, dimension - i - 1), (0, 0)))
