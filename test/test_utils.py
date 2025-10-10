import pytest
import numpy as np
from qiskit.circuit import Parameter
from qiskit.quantum_info import DensityMatrix
from qadc.utils import Utils


# --------------------------
# normalize_counts
# --------------------------
def test_normalize_counts_valid():
    counts = {"0": 5, "1": 3}
    result = Utils.normalize_counts(counts)
    assert result == {"0": 5, "1": 3}


def test_normalize_counts_missing_key():
    counts = {"0": 7}
    result = Utils.normalize_counts(counts)
    # deve inserire anche '1' con valore 0
    assert result == {"0": 7, "1": 0}


# --------------------------
# cost_function
# --------------------------
def test_cost_function_balanced():
    counts = [{"0": 5, "1": 5}, {"0": 5, "1": 5}]
    result = Utils.cost_function(counts)
    # distribuzione perfetta: costo < 1 ma > 0
    assert 0 <= result <= 1


def test_cost_function_single_outcome():
    counts = [{"0": 10, "1": 0}, {"0": 0, "1": 10}]
    result = Utils.cost_function(counts)
    # perfetta discriminazione => costo = 0
    assert result == pytest.approx(0.0)


# --------------------------
# merge_dict
# --------------------------
def test_merge_dict_valid():
    dicts = [{"a": 1, "b": 2}, {"a": 3, "b": 4}]
    result = Utils.merge_dict(dicts)
    assert result == {"a": [1, 3], "b": [2, 4]}


def test_merge_dict_empty():
    result = Utils.merge_dict([])
    assert result == {}


# --------------------------
# TwoAsinSqrt
# --------------------------
@pytest.mark.parametrize("x", [0, 0.25, 1])
def test_two_asin_sqrt_valid(x):
    theta = Utils.TwoAsinSqrt(x)
    assert 0 <= theta <= np.pi


def test_two_asin_sqrt_parameter():
    p = Parameter("x")
    expr = Utils.TwoAsinSqrt(p)
    assert p in expr.parameters

@pytest.mark.parametrize("x", [-2, -1, 1.5, 2])
def test_two_asin_sqrt_invalid(x):
    with pytest.raises(ValueError):
        Utils.TwoAsinSqrt(x)


# --------------------------
# SquareSinHalf
# --------------------------
@pytest.mark.parametrize("x", [0, np.pi, 2*np.pi])
def test_square_sin_half_valid(x):
    val = Utils.SquareSinHalf(x)
    assert 0 <= val <= 1


def test_square_sin_half_parameter():
    p = Parameter("theta")
    expr = Utils.SquareSinHalf(p)
    assert p in expr.parameters


def test_square_sin_half_invalid():
    with pytest.raises(ValueError):
        Utils.SquareSinHalf(None)


# --------------------------
# reverse_qargs_density_matrix
# --------------------------
def test_reverse_qargs_density_matrix():
    rho = DensityMatrix.from_label("00")
    reversed_rho = Utils.reverse_qargs_density_matrix(rho)
    assert isinstance(reversed_rho, DensityMatrix)
    assert reversed_rho.is_valid()


# --------------------------
# initial_mixed_state_real_density_matrix
# --------------------------
@pytest.mark.parametrize("alpha", [0.0, 0.5, 1.0])
def test_initial_mixed_state_real_density_matrix(alpha):
    rho = Utils.initial_mixed_state_real_density_matrix(alpha)
    assert isinstance(rho, DensityMatrix)
    assert rho.is_valid()
    # stato a 2 qubit => dimensione 4x4
    assert rho.dim == 4


def test_initial_mixed_state_invalid_alpha():
    with pytest.raises(Exception):
        Utils.initial_mixed_state_real_density_matrix(-0.5)
