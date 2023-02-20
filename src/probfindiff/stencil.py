"""Stencil functionality."""


from typing import Tuple, Union

import jax.numpy as jnp

from probfindiff import defaults
from probfindiff.typing import Array, ArrayLike


def multivariate(
    *,
    xs_1d: ArrayLike,
    shape_input: Tuple[int],
    shape_output: Union[Tuple[()], Tuple[int]] = (),
) -> Array:
    r"""Turn a univariate finite-difference stencil into a multivariate stencil.

    Parameters
    ----------
    xs_1d
        Input finite-difference grid/stencil in 1d.
    shape_input
        Input dimension of the to-be-differentiated function as a shape-tuple.
        If the goal is the gradient of an `n`-dimensional function, ``shape_input=(n,)``.
    shape_output
        Output dimension of the to-be-differentiated function as a shape-tuple.

    Returns
    -------
    :
        New grid with shape ``shape_output + (n, n, c)``.


    Examples
    --------
    >>> from probfindiff import central
    >>> _, xs_1d = central(dx=1.)
    >>> print(xs_1d)
    [-1.  0.  1.]

    >>> xs = multivariate(xs_1d=xs_1d, shape_input=(2,))
    >>> print(xs.shape)
    (2, 2, 3)
    >>> print(xs)
    [[[-1.  0.  1.]
      [ 0.  0.  0.]]
     [[ 0.  0.  0.]
      [-1.  0.  1.]]]
    """
    assert len(shape_input) == 1
    xs_1d = jnp.asarray(xs_1d)

    coeffs = _stencils_for_all_partial_derivatives(
        shape_input=shape_input, stencil_1d=xs_1d
    )
    # Independent copies for each output dimension
    return jnp.broadcast_to(coeffs, shape=shape_output + coeffs.shape)


def _stencils_for_all_partial_derivatives(
    *, stencil_1d: ArrayLike, shape_input: Tuple[int]
) -> Array:
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
    >>> stencils = _stencils_for_all_partial_derivatives(stencil_1d=jnp.array([1, 2, 3]), shape_input=(2,))
    >>> print(stencils)
    [[[1 2 3]
      [0 0 0]]
     [[0 0 0]
      [1 2 3]]]

    >>> stencils = _stencils_for_all_partial_derivatives(stencil_1d=jnp.array([1, 2]), shape_input=(4,))
    >>> print(stencils)
    [[[1 2]
      [0 0]
      [0 0]
      [0 0]]
     [[0 0]
      [1 2]
      [0 0]
      [0 0]]
     [[0 0]
      [0 0]
      [1 2]
      [0 0]]
     [[0 0]
      [0 0]
      [0 0]
      [1 2]]]
    """
    stencil_1d = jnp.asarray(stencil_1d)

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


def _stencil_for_ith_partial_derivative(
    *, stencil_1d_as_row_matrix: ArrayLike, i: int, dimension: int
) -> Array:
    """Compute the stencil for the ith partial derivative.

    This is done by padding the 1d stencil into zeros according to the index ``i`` and the spatial dimension.
    It constitutes a stencil for a partial derivative, because such stencils are only affected by
    changes to the input along _a single_ dimension. The others remain unaffected, hence the zeros.

    Examples
    --------
    >>> stencils = _stencil_for_ith_partial_derivative(stencil_1d_as_row_matrix=jnp.array([[1, 2, 3]]), i=0, dimension=4)
    >>> print(stencils)
    [[1 2 3]
     [0 0 0]
     [0 0 0]
     [0 0 0]]
    >>> stencils = _stencil_for_ith_partial_derivative(stencil_1d_as_row_matrix=jnp.array([[1, 2, 3]]), i=2, dimension=4)
    >>> print(stencils)
    [[0 0 0]
     [0 0 0]
     [1 2 3]
     [0 0 0]]
    >>> stencils = _stencil_for_ith_partial_derivative(stencil_1d_as_row_matrix=jnp.array([[1, 2, 3, 4, 5]]), i=1, dimension=8)
    >>> print(stencils)
    [[0 0 0 0 0]
     [1 2 3 4 5]
     [0 0 0 0 0]
     [0 0 0 0 0]
     [0 0 0 0 0]
     [0 0 0 0 0]
     [0 0 0 0 0]
     [0 0 0 0 0]]
    >>> stencils =  _stencil_for_ith_partial_derivative(stencil_1d_as_row_matrix=jnp.array([[1, 2, 3, 4, 5]]), i=7, dimension=8)
    >>> print(stencils)
    [[0 0 0 0 0]
     [0 0 0 0 0]
     [0 0 0 0 0]
     [0 0 0 0 0]
     [0 0 0 0 0]
     [0 0 0 0 0]
     [0 0 0 0 0]
     [1 2 3 4 5]]
    """
    stencil_1d_as_row_matrix = jnp.asarray(stencil_1d_as_row_matrix)
    return jnp.pad(stencil_1d_as_row_matrix, pad_width=((i, dimension - i - 1), (0, 0)))


def backward(
    *,
    dx: float,
    order_derivative: int = defaults.ORDER_DERIVATIVE,
    order_method: int = defaults.ORDER_METHOD,
) -> Array:
    """Create the stencil for backward finite difference schemes."""
    offset = -jnp.arange(order_derivative + order_method, step=1)
    grid = offset * dx
    return grid


def forward(
    *,
    dx: float,
    order_derivative: int = defaults.ORDER_DERIVATIVE,
    order_method: int = defaults.ORDER_METHOD,
) -> Array:
    """Create the stencil for forward finite difference schemes."""
    offset = jnp.arange(order_derivative + order_method, step=1)
    grid = offset * dx
    return grid


def central(
    *,
    dx: float,
    order_derivative: int = defaults.ORDER_DERIVATIVE,
    order_method: int = defaults.ORDER_METHOD_CENTRAL,
) -> Array:
    """Create the stencil for central finite difference schemes."""
    num_central = (2 * ((order_derivative + 1.0) / 2.0) // 2) - 1 + order_method
    num_side = num_central // 2
    offset = jnp.arange(-num_side, num_side + 1, step=1)
    grid = offset * dx
    return grid
