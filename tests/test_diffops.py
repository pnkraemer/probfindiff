"""Tests for differential operators."""

import pytest_cases
from pnfindiff import diffops


def case_div():
    return diffops.div()


def case_grad():
    return diffops.grad()


def case_laplace():
    return diffops.laplace()


@pytest_cases.parametrize_with_cases("D", cases=".")
def test_diff(D):
    D(lambda x, y: y + x ** 2, argnums=0)
