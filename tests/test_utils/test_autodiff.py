"""Tests for differential operators."""

import pytest_cases

from probnum_findiff.utils import autodiff


def case_div():
    return autodiff.div


def case_derivative():
    return autodiff.derivative


def case_laplace():
    return autodiff.laplace


@pytest_cases.parametrize_with_cases("D", cases=".")
def test_diff(D):
    D(lambda x, y: y + x**2, argnums=0)
