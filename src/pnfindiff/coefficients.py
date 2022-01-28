"""Finite difference coefficients. (In 1d.)"""

from pnfindiff import collocation

def backward(*, deriv, dx, acc=2):
    raise NotImplementedError

def scattered1d(*, x, xs, ks):
    k, lk, llk = ks
    n = xs.shape[0]

    K = k(xs[:, None], xs[None, :]).reshape((n,n))
    LK = lk(x[:, None], xs[None, :]).reshape((n,))
    LLK = llk(x[:, None], x[None, :]).reshape(())

    return collocation.unsymmetric(K=K, LK0=LK, LLK=LLK)