"""Tests for differential operators."""

import pytest_cases

from pnfindiff.base import diffops


def case_div():
    return diffops.div()


def case_deriv_scalar():
    return diffops.deriv_scalar()


def case_laplace():
    return diffops.laplace()


@pytest_cases.parametrize_with_cases("D", cases=".")
def test_diff(D):
    D(lambda x, y: y + x ** 2, argnums=0)
