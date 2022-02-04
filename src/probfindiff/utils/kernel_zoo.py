"""Kernel zoo."""


from functools import partial

import jax
import jax.numpy as jnp

from probfindiff.typing import ArrayLike


@jax.jit
def exponentiated_quadratic(
    x: ArrayLike, y: ArrayLike, input_scale: float = 1.0, output_scale: float = 1.0
) -> ArrayLike:
    r"""Exponentiated quadratic kernel.

    The kernel is defined as

    .. math:: k(x,y) = \sigma \exp(-\epsilon \|x-y\|^2/2)

    Parameters
    ----------
    x
        Input variable.
    y
        Input variable.
    input_scale
        Input scale :math:`\epsilon` of the kernel.
    output_scale
        Output scale :math:`\sigma` of the kernel.


    Returns
    -------
    :
        Evaluation :math:`k(x,y)`.
    """
    return output_scale * jnp.exp(-input_scale * (x - y).dot(x - y) / 2.0)


@partial(jax.jit, static_argnames=("scale", "order", "bias"))
def polynomial(
    x: ArrayLike,
    y: ArrayLike,
    *,
    p: ArrayLike,
) -> ArrayLike:
    r"""Polynomial kernels.

    The kernel is defined as

    .. math::
        k(x,y) = p[0]*\langle x, y\rangle^(N-1) + p[1]*\langle x, y\rangle^(N-2) + ... + p[N-2]*\langle x, y\rangle + p[N-1]

    Parameters
    ----------
    x
        Input variable.
    y
        Input variable.
    p
        Coeficients of the polynomial

    Returns
    -------
    :
        Evaluation :math:`k(x,y)`.
    """
    return jnp.polyval(p, jnp.dot(x, y)) + 1
