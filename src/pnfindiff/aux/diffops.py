"""Differential operators."""

import jax
import jax.numpy as jnp


def deriv_scalar(fun, *, argnums=0):
    """Derivative of a scalar function."""

    grad = jax.grad(fun, argnums=argnums)
    return lambda *args: grad(*args).squeeze()


def div(fun, *, argnums=0):
    """Divergence of a function as the trace of the Jacobian."""

    jac = jax.jacrev(fun, argnums=argnums)
    return lambda *args: jnp.trace(jac(*args))


def laplace(fun, *, argnums=0):
    """Laplace operator."""
    return compose(div, jax.grad)(fun, argnums=argnums)


def compose(op1, op2):
    """Compose two differential operators."""

    def D(fun, *, argnums=0):
        return op1(op2(fun, argnums=argnums), argnums=argnums)

    return D
