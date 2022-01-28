"""Finite difference coefficients. (In 1d.)"""

from pnfindiff import collocation, kernel, diffops
import jax.numpy as jnp


def backward(x, *, deriv, dx, acc=2):
    k = lambda x, y: jnp.exp(-(x-y).dot(x - y) / 2.0)

    if deriv == 1:
        L = diffops.grad()
    else:
        raise RuntimeError
    lk = L(k, argnums=0)
    llk = L(lk, argnums=1)

    k = kernel.vmap_gram(k)
    lk = kernel.vmap_gram(lk)
    llk = kernel.vmap_gram(llk)

    xs = x - jnp.arange(deriv + acc) * dx
    return scattered_1d(x=x, xs=xs, ks=(k, lk, llk))


def scattered_1d(*, x, xs, ks):
    k, lk, llk = ks
    n = xs.shape[0]
    x = jnp.array([x])

    K = k(xs[:, None], xs[None, :]).reshape((n, n))
    LK = lk(x[:, None], xs[None, :]).reshape((n,))
    LLK = llk(x[:, None], x[None, :]).reshape(())

    return collocation.unsymmetric(K=K, LK0=LK, LLK=LLK)
