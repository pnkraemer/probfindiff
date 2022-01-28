"""Differential operators."""

import jax
import jax.numpy as jnp


def deriv_scalar(fun, /, **kwargs):
    """Derivative of a scalar function."""

    grad = jax.grad(fun, **kwargs)
    return lambda *args: grad(*args).squeeze()


def div(fun, /, **kwargs):
    """Divergence of a function as the trace of the Jacobian."""

    jac = jax.jacrev(fun, **kwargs)
    return lambda *args: jnp.trace(jac(*args))


def laplace(fun, /, **kwargs):
    """Laplace operator."""
    return compose(div, jax.grad)(fun, **kwargs)


def compose(op1, op2):
    """Compose two differential operators."""

    def D(fun, /, **kwargs):
        return op1(op2(fun, **kwargs), **kwargs)

    return D
