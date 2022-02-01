"""Differential operators."""

from typing import Any, Callable

import jax
import jax.numpy as jnp

from pnfindiff.typing import DifferentialOperatorLike


def deriv_scalar(fun: Callable[[Any], Any], /, **kwargs: Any) -> Callable[[Any], Any]:
    """Derivative of a scalar function."""

    grad = jax.grad(fun, **kwargs)
    return jax.jit(lambda *args: grad(*args)[0])


def div(fun: Callable[[Any], Any], /, **kwargs: Any) -> Callable[[Any], Any]:
    """Divergence of a function as the trace of the Jacobian."""

    jac = jax.jacrev(fun, **kwargs)
    return jax.jit(lambda *args: jnp.trace(jac(*args)))


def laplace(fun: Callable[[Any], Any], /, **kwargs: Any) -> Callable[[Any], Any]:
    """Laplace operator."""
    return compose(div, jax.grad)(fun, **kwargs)


def compose(
    op1: DifferentialOperatorLike, op2: DifferentialOperatorLike
) -> DifferentialOperatorLike:
    """Compose two differential operators."""

    def D(fun: Callable[[Any], Any], /, **kwargs: Any) -> Callable[[Any], Any]:
        return op1(op2(fun, **kwargs), **kwargs)

    return D
