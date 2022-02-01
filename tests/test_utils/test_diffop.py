"""Tests for differential operators."""

import pytest_cases

from pnfindiff.utils import diffop


def case_div():
    return diffop.div


def case_deriv_scalar():
    return diffop.deriv_scalar


def case_laplace():
    return diffop.laplace


@pytest_cases.parametrize_with_cases("D", cases=".")
def test_diff(D):
    D(lambda x, y: y + x**2, argnums=0)
