"""Kernel zoo."""


from functools import partial
from typing import Any

import jax
import jax.numpy as jnp


@jax.jit
def exponentiated_quadratic(
    x: Any, y: Any, input_scale: float = 1.0, output_scale: float = 1.0
) -> Any:
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
    x: Any, y: Any, *, order: int = 2, scale: float = 1.0, bias: float = 1.0
) -> Any:
    r"""Polynomial kernels.

    The kernel is defined as

    .. math:: k(x,y) = (a \langle x, y\rangle + b)^c

    Parameters
    ----------
    x
        Input variable.
    y
        Input variable.
    scale
        Scale :math:`a`.
    bias
        Bias :math:`b`.
    order
        Order :math:`c`.

    Returns
    -------
    :
        Evaluation :math:`k(x,y)`.
    """
    return (scale * x.dot(y) + bias) ** order
