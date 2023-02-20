# probfindiff: Probabilistic numerical finite differences

[![PyPi Version](https://img.shields.io/pypi/v/probfindiff.svg?style=flat-square)](https://pypi.org/project/probfindiff/)
[![Docs](https://readthedocs.org/projects/pip/badge/?version=latest&style=flat-square)](https://github.com/pnkraemer/probfindiff)
[![GitHub stars](https://img.shields.io/github/stars/pnkraemer/probfindiff.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/pnkraemer/probfindiff)
[![gh-actions](https://img.shields.io/github/workflow/status/pnkraemer/probfindiff/ci?style=flat-square)](https://github.com/pnkraemer/probfindiff/actions?query=workflow%3Aci)
<a href="https://github.com/pnkraemer/probfindiff/blob/master/LICENSE"><img src="https://img.shields.io/github/license/pnkraemer/probfindiff?style=flat-square&color=2b9348" alt="License Badge"/></a>


Traditional finite difference schemes are absolutely crucial for scientific computing.
If you like uncertainty quantification, transparent algorithm assumptions, and next-level flexibility in your function evaluations, you need probabilistic numerical finite differences.

## Why?
Because when using traditional finite difference coefficients, one implicitly assumes that the function to-be-differentiated is a polynomial and evaluated on an equidistant grid.
Is your function _really_ a polynomial? Are you satisfied with evaluating your function on equidistant grids?
If not, read on.


## In a nutshell
Traditional, non-probabilistic finite difference schemes fit a polynomial to function evaluations and differentiate the polynomial to compute finite difference weights (Fornberg, 1988).
But why use a polynomial? 
If we use a Gaussian process, we get uncertainty quantification, scattered point sets, transparent modelling assumptions, and many more benefits for free!




## How?
Probabilistic numerical finite difference schemes can be applied to function evaluations as follows.
```python
>>> import jax.numpy as jnp
>>> from jax.config import config
>>> import probfindiff
>>>
>>> config.update("jax_platform_name", "cpu")
>>>
>>> scheme, xs = probfindiff.central(dx=0.2)
>>> f = lambda x: (x-1.)**2.
>>> dfx, cov = probfindiff.differentiate(f(xs), scheme=scheme)
>>> print(jnp.round(dfx, 1))
-2.0
>>> print(jnp.round(jnp.log10(cov), 0))
-7.0
>>> print(isinstance(scheme, tuple))
True
>>> print(jnp.round(scheme.weights, 1))
[-2.5  0.   2.5]
```
See [**this page**](https://probfindiff.readthedocs.io/en/latest/notebooks/getting_started/quickstart.html) for more examples.


## Installation

```commandline
pip install probfindiff
```
This assumes that JAX is already installed. If not, use
```commandline
pip install probfindiff[cpu]
```
to combine `probfindiff` with `jax[cpu]`.


With all dev-related setups:
```commandline
pip install probfindiff[ci]
```

## Features
* Forward, backward, central probabilistic numerical finite differences
* Finite differences on arbitrarily scattered point sets and in high dimensions
* Finite difference schemes for observation noise
* Symmetric and unsymmetric collocation ("Kansa's method")
* Polynomial kernels, exponentiated quadratic kernels, and an API for custom kernels
* Partial derivatives, Laplacians, divergences, compositions of differential operators, and an API for custom differential operators
* Tutorials that explain how to use all of the above
* Compilation, vectorisation, automatic differentiation, and everything else that JAX provides.

Check the tutorials on [**this page**](https://probfindiff.readthedocs.io/en/latest/) for examples.



## Background
The technical background is explained in the paper:
```
@article{kramer2022probabilistic,
  title={Probabilistic Numerical Method of Lines for Time-Dependent Partial Differential Equations},
  author={Kr{\"a}mer, Nicholas and Schmidt, Jonathan and Hennig, Philipp},
  journal={AISTATS},
  year={2022}
}
```
Please consider citing it if you use this repository for your research.

## Related work finite difference methods
Finite difference schemes are not new, obviously.

#### Python

* FinDiff (https://findiff.readthedocs.io/en/latest/)
* pystencils (https://pycodegen.pages.i10git.cs.fau.de/pystencils/index.html#)
* FDM (https://github.com/wesselb/fdm)
* RBF(.fd) (https://rbf.readthedocs.io/en/latest/fd.html)

#### Julia

* FiniteDifferences.jl (https://github.com/JuliaDiff/FiniteDifferences.jl)
* FinDiff.jl (https://github.com/JuliaDiff/FiniteDiff.jl)

#### Distinction

`probfindiff` does many things similarly to the above, but differs in more than one points:

* We do _probabilistic numerical_ finite differences.
  Essentially, this encompasses a Gaussian process perspective
  on radial-basis-function-generated finite differences (provided by "RBF" above).
  As such, different to _all_ of the above, we treat uncertainty quantification and modelling
  as first-class-citizens.
* `probfindiff` uses JAX, which brings with it automatic differentiation, JIT-compilation, GPUs, and more.
* `probfindiff` does not evaluate functions. Most of the packages above have an API
  `differentiate(fn, x, scheme)` whereas we use `differentiate(fn(x), scheme.weights)`.
  We choose the latter because implementations simplify (we pass around arrays instead of callables),
  gain efficiency (it becomes obvious which quantities to reuse in multiple applications),
  and because users know best how to evaluate their functions (for example, whether the function is vectorised).

