"""Differential operators."""

import jax
import jax.numpy as jnp


def div():
    """Divergence of a function as the trace of the Jacobian."""

    def D(fun, *, argnums=0):
        jac = jax.jacrev(fun, argnums=argnums)
        return lambda *args: jnp.trace(jac(*args))

    return D


def deriv_scalar():
    """Derivative of a scalar function."""

    def D(fun, *, argnums=0):
        grad = jax.grad(fun, argnums=argnums)
        return lambda *args: grad(*args).squeeze()

    return D


def laplace():
    """Laplace operator."""
    return compose(div(), jax.grad)


def compose(op1, op2):
    def D(fun, *, argnums=0):
        return op1(op2(fun, argnums=argnums), argnums=argnums)

    return D
