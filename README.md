# probfindiff: Probabilistic numerical finite differences

[![PyPi Version](https://img.shields.io/pypi/v/probfindiff.svg?style=flat-square)](https://pypi.org/project/probfindiff/)
[![Docs](https://readthedocs.org/projects/pip/badge/?version=latest&style=flat-square)](https://github.com/pnkraemer/probfindiff)
[![GitHub stars](https://img.shields.io/github/stars/pnkraemer/probfindiff.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/pnkraemer/probfindiff)
[![gh-actions](https://img.shields.io/github/workflow/status/pnkraemer/probfindiff/ci?style=flat-square)](https://github.com/pnkraemer/probfindiff/actions?query=workflow%3Aci)
<a href="https://github.com/pnkraemer/probfindiff/blob/master/LICENSE"><img src="https://img.shields.io/github/license/pnkraemer/probfindiff?style=flat-square&color=2b9348" alt="License Badge"/></a>


Traditional finite difference schemes are great, but let's look at the whole picture.
## Why?
Because when using traditional finite difference coefficients, one implicitly assumes that the function to-be-differentiated is a polynomial.
Is your function _really_ a polynomial? If not, read on.



## How?
Probabilistic numerical finite difference schemes can be applied to function evaluations as follows.
```python
>>> import jax.numpy as jnp
>>> import probfindiff
>>> scheme, xs = probfindiff.central(dx=0.2)
WARNING:absl:No GPU/TPU found, falling back to CPU. (Set TF_CPP_MIN_LOG_LEVEL=0 and rerun for more info.)
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
With all dev-related setups:
```commandline
pip install probfindiff[ci]
```
Or from github directly:
```commandline
pip install git+https://github.com/pnkraemer/probfindiff.git
```


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
  as first-class-citizens. In some sense, PN finite differences generalise
  traditional schemes (for specific kernels and grids), which is easily accessible
  in `probfindiff`'s implementation, but a few flavours of methods are different
  because of the probability theory.
* `probfindiff` uses JAX, which brings with it automatic differentiation, JIT-compilation, GPUs, and more.
* `probfindiff` does not evaluate functions. There is also no internal state.
  Finite difference schemes are plain tuples of coefficient vectors,
  and not callables that call the to-be-differentiated function.
  This is more efficient (schemes are reused; functions are easy to JIT-compile;
  we cannot call your function more efficiently than you can),
  more transparent (we do not recompute stencils for new points, because we only provide coefficient vectors),
  and implies fewer corner cases ("Have you provided a vectorised function?";
  "What if my data is hand-gathered or points to some weird black-box-simulator?").

At the time of writing, there has been much more work on the packages above than on `probfindiff`
(which clearly shows -- they're all great and have been big inspiration for this package!), so
interfaces may be more stable with the other packages for now.
Numerical stability may also not be where it could be.
Therefore, choose your package with these thoughts in mind.
