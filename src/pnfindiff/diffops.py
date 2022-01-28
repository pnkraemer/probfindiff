"""Differential operators."""

import jax
import jax.numpy as jnp

def div():
    """Divergence of a function as the trace of the Jacobian."""

    def D(fun, *, argnums=0):
        jac = jax.jacrev(fun, argnums=argnums)
        return lambda *args: jnp.trace(jac(*args))

    return D


def grad():
    """Gradient of a function."""

    def D(fun, *, argnums=0):
        _assure_scalar_fn = lambda *args, **kwargs: fun(*args, **kwargs).squeeze()
        return jax.grad(_assure_scalar_fn, argnums=argnums)

    return D


def laplace():
    """Laplace operator."""
    return _compose(div(), grad())


def _compose(op1, op2):
    def D(fun, *, argnums=0):
        return op1(op2(fun, argnums=argnums), argnums=argnums)

    return D
