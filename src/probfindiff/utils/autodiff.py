"""Differential operators that are not part of common autodiff frameworks."""

from typing import Any, Callable

import jax
import jax.numpy as jnp

from probfindiff.typing import DifferentialOperatorLike


def derivative(fun: Callable[[Any], Any], **kwargs: Any) -> Callable[[Any], Any]:
    """Derivative of a scalar function.

    Parameters
    ----------
    fun
        Function to be differentiated.
    **kwargs
        Keyword arguments to be passed down to :func:`jax.grad`.

    Returns
    -------
    :
        Differentiated function.
    """

    grad = jax.grad(fun, **kwargs)
    return jax.jit(lambda *args: grad(*args)[0])


def div(fun: Callable[[Any], Any], **kwargs: Any) -> Callable[[Any], Any]:
    """Divergence of a function as the trace of the Jacobian.

    Parameters
    ----------
    fun
        Function to be differentiated.
    **kwargs
        Keyword arguments to be passed down to :func:`jax.jacrev`.

    Returns
    -------
    :
        Differentiated function.
    """

    jac = jax.jacrev(fun, **kwargs)
    return jax.jit(lambda *args: jnp.trace(jac(*args)))


def laplace(fun: Callable[[Any], Any], **kwargs: Any) -> Callable[[Any], Any]:
    """Laplace operator.

    Parameters
    ----------
    fun
        Function to be differentiated.
    **kwargs
        Keyword arguments to be passed down to :func:`jax.grad` and :func:`jax.jacrev`.

    Returns
    -------
    :
        Differentiated function.
    """
    return compose(div, jax.grad)(fun, **kwargs)


def compose(
    op1: DifferentialOperatorLike, op2: DifferentialOperatorLike
) -> DifferentialOperatorLike:
    """Compose two differential operators.

    Parameters
    ----------
    op1
        Differential operator.
    op2
        Differential operator.

    Returns
    -------
    :
        Composed differential operator.
    """

    def D(fun: Callable[[Any], Any], /, **kwargs: Any) -> Callable[[Any], Any]:
        return op1(op2(fun, **kwargs), **kwargs)

    return D
