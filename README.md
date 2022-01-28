# pnfindiff: Probabilistic numerical finite differences.

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

## Misc

See also: FinDiff (https://findiff.readthedocs.io/en/latest/). 
