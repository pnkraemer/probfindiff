"""Finite difference coefficients. (In 1d.)"""

from pnfindiff import collocation, kernel, diffops
import jax.numpy as jnp


def backward(x, *, deriv, dx, acc=2):

    match_deriv = {
        1: diffops.grad(),
        2: diffops.laplace(),
    }
    L = match_deriv[deriv]
    k_batch, k = kernel.exp_quad()
    lk_batch, lk = kernel.batch_gram(L(k, argnums=0))
    llk_batch, llk = kernel.batch_gram(L(lk, argnums=1))
    xs = x - jnp.arange(deriv + acc) * dx
    return scattered_1d(x=x, xs=xs, ks=(k_batch, lk_batch, llk_batch))


def scattered_1d(*, x, xs, ks):
    k, lk, llk = ks
    n = xs.shape[0]
    x = jnp.array([x])

    K = k(xs[:, None], xs[None, :]).reshape((n, n))
    LK = lk(x[:, None], xs[None, :]).reshape((n,))
    LLK = llk(x[:, None], x[None, :]).reshape(())

    return collocation.unsymmetric(K=K, LK0=LK, LLK=LLK)
