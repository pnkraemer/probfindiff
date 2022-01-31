"""Top-level API."""

from functools import partial, reduce

import jax
import jax.numpy as jnp
import scipy.spatial

from . import coefficients
from .aux import diffop, kernel

__all__ = ["findiff"]


def findiff(*, xs, deriv=1, num=2):
    tree = scipy.spatial.KDTree(data=xs.reshape((-1, 1)))
    distances, indices = tree.query(x=xs.reshape((-1, 1)), k=num)
    neighbours = xs[indices]

    _, k = kernel.exp_quad()
    ks = _differentiate_kernel(deriv=deriv, k=k)
    coeffs_batched = jax.jit(jax.vmap(partial(coefficients.scattered_1d, ks=ks)))
    coeffs_full = coeffs_batched(x=xs, xs=neighbours)

    def diff(fx):
        return jnp.einsum("nk,nk->n", coeffs_full[0], fx[indices])

    return diff
    #
    #     df_left = jnp.convolve(a=fx, v=jnp.flip(left_boundary[0]), mode="same")
    #     df_center = jnp.convolve(a=fx, v=center[0], mode="same")
    #     df_right = jnp.convolve(a=fx, v=right_boundary[0], mode="same")
    #
    #     print(df_left)
    #     print(df_center)
    #     print(df_right)
    #     return jnp.concatenate(
    #         (
    #             df_left[0].reshape((1,)),
    #             df_center[1:-1],
    #             df_right[-1].reshape((1,))
    #         )
    #     )
    #
    # return diff


def _differentiate_kernel(*, deriv, k):
    L = reduce(diffop.compose, [diffop.deriv_scalar] * deriv)
    k_batch, _ = kernel.batch_gram(k)
    lk_batch, lk = kernel.batch_gram(L(k, argnums=0))
    llk_batch, _ = kernel.batch_gram(L(lk, argnums=1))
    return k_batch, lk_batch, llk_batch
