# probfindiff: Probabilistic numerical finite differences.

Traditional finite difference schemes are great, but let's look at the whole picture.
## Why?
Because when using traditional finite difference coefficients, one implicitly assumes that the function to-be-differentiated is a polynomial.
Is your function _really_ a polynomial? If not, read on.


## Getting started

Installation:
```commandline
pip install .
```
With all dev-related setups
```commandline
pip install .[ci]
```

## Background
The technical background is explained in the paper:
```
@article{kramer2021probabilistic,
  title={Probabilistic Numerical Method of Lines for Time-Dependent Partial Differential Equations},
  author={Kr{\"a}mer, Nicholas and Schmidt, Jonathan and Hennig, Philipp},
  journal={AISTATS},
  year={2021}
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

`pnfindiff` does many things similarly to the above, but differs in more than one points:

* We do _probabilistic numerical_ finite differences. 
  Essentially, this encompasses a Gaussian process perspective 
  on radial-basis-function-generated finite differences (provided by "RBF" above).
  As such, different to _all_ of the above, we treat uncertainty quantification and modelling 
  as first-class-citizens. In some sense, PN finite differences generalise
  traditional schemes (for specific kernels and grids), which is easily accessible
  in `pnfindiff`'s implementation, but a few flavours of methods are different 
  because of the probability theory.
* `pnfindiff` uses JAX, which brings with it automatic differentiation, JIT-compilation, GPUs, and more.
* `pnfindiff` does not evaluate functions. There is also no internal state. 
  Finite difference schemes are plain tuples of coefficient vectors,
  and not callables that call the to-be-differentiated function. 
  This is more efficient (schemes are reused; functions are easy to JIT-compile; 
  we cannot call your function more efficiently than you can), 
  more transparent (we do not recompute stencils for new points, because we only provide coefficient vectors), 
  and implies fewer corner cases ("Have you provided a vectorised function?";
  "What if my data is hand-gathered or points to some weird black-box-simulator?").

At the time of writing, there has been much more work on the packages above than on `pnfindiff`
(which clearly shows -- they're all great and have been big inspiration for this package!), so
interfaces may be more stable with the other packages for now. 
Numerical stability may also not be where it could be. 
Therefore, choose your package with these thoughts in mind.