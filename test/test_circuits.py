import pytest
import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from qadc.circuits import (
    BaseAmplitudeDampingCircuit as BADC,
    AmplitudeDampingConvenzioneClassica,
    AmplitudeDampingConvenzioneQiskit,
    EncodingCircuit,
    DecodingCircuit,
    AncillaInitializationCircuit
)
from qadc.utils import Utils
from qiskit.circuit.exceptions import CircuitError
from IPython.display import display, Latex



# ==============================
# BaseAmplitudeDampingCircuit
# ==============================
def test_base_amplitude_damping_circuit_float():
    eta = 0.3
    circ = BADC(eta)
    assert isinstance(circ,QuantumCircuit)
    assert circ.eta == eta
    # theta deve corrispondere a Utils.TwoAsinSqrt(eta)
    # assert circ.theta == Utils.TwoAsinSqrt(eta)
    assert circ.num_qubits == 2


def test_base_amplitude_damping_circuit_parameter():
    eta = Parameter("η_test")
    circ = BADC(eta)
    assert isinstance(circ,QuantumCircuit)
    assert circ.eta == eta
    # assert circ.theta == Utils.TwoAsinSqrt(eta)
    assert circ.num_qubits == 2


def test_base_amplitude_damping_circuit_none():
    circ = BADC(None)
    assert isinstance(circ,QuantumCircuit)
    assert isinstance(circ.eta, Parameter)
    assert circ.num_qubits == 2



# ==============================
# AmplitudeDampingConvenzioneClassica
# ==============================
def test_amplitude_damping_classica():
    circ = AmplitudeDampingConvenzioneClassica(0.5)
    assert isinstance(circ,QuantumCircuit)
    assert circ.num_qubits == 2
    # deve contenere almeno un gate CX e CRY
    gate_types = [gate[0].name for gate in circ.data]
    assert "cx" in gate_types
    assert "cry" in gate_types

def test_classico_eta_none_assignment_invalid():
    circ = AmplitudeDampingConvenzioneClassica(eta=None)
    assert isinstance(circ,QuantumCircuit)
    # Provo ad assegnare None come parametro → errore
    with pytest.raises(Exception):
        circ.assign_parameters({circ.eta: None})
    # display(circ.assign_parameters({circ.eta: np.nan}).draw("mpl"))
    # with pytest.raises(ValueError):
    #     circ.assign_parameters({circ.eta: np.nan})  # np.nan simula un valore "non valido"

def test_classico_eta_float_outside_range():
    circ = AmplitudeDampingConvenzioneClassica(eta=None)
    assert isinstance(circ,QuantumCircuit)
    # Valore minore di 0
    with pytest.raises((ValueError)):
        circ.assign_parameters({circ.eta: BADC.noise_to_theta(-0.5)})
    
    # Valore maggiore di 1
    with pytest.raises((ValueError)):
        circ.assign_parameters({circ.eta: BADC.noise_to_theta(1.5)})

# ==============================
# AmplitudeDampingConvenzioneQiskit
# ==============================
def test_amplitude_damping_qiskit():
    circ = AmplitudeDampingConvenzioneQiskit(0.5)
    assert isinstance(circ,QuantumCircuit)
    assert circ.num_qubits == 2
    gate_types = [gate[0].name for gate in circ.data]
    assert "cx" in gate_types
    assert "cry" in gate_types


# ==============================
# EncodingCircuit
# ==============================
def test_encoding_circuit_default():
    circ = EncodingCircuit()
    assert circ.num_qubits == 1
    assert len(circ.parameters) == 2


def test_encoding_circuit_float_list():
    circ = EncodingCircuit([0.1, 0.2])
    assert circ.num_qubits == 1


def test_encoding_circuit_wrong_length():
    with pytest.raises(ValueError):
        EncodingCircuit([0.1])  # meno di 2 parametri


# ==============================
# DecodingCircuit
# ==============================
def test_decoding_circuit_default():
    circ = DecodingCircuit()
    assert circ.num_qubits == 1
    assert len(circ.parameters) == 2


def test_decoding_circuit_float_list():
    circ = DecodingCircuit([0.1, 0.2])
    assert circ.num_qubits == 1


def test_decoding_circuit_wrong_length():
    with pytest.raises(ValueError):
        DecodingCircuit([0.1, 0.2, 0.3])  # più di 2 parametri


# ==============================
# AncillaInitializationCircuit
# ==============================
def test_ancilla_initialization_float():
    alpha = 0.25
    circ = AncillaInitializationCircuit(alpha)
    assert isinstance(circ,QuantumCircuit)
    assert circ.alpha == alpha
    # theta deve essere 2 * arcsin(sqrt(alpha))
    # assert np.isclose(circ., 2 * np.arcsin(np.sqrt(alpha)))
    assert circ.num_qubits == 2


def test_ancilla_initialization_parameter():
    alpha = Parameter("α_test")
    circ = AncillaInitializationCircuit(alpha)
    assert isinstance(circ,QuantumCircuit)
    assert circ.alpha == alpha
    assert circ.num_qubits == 2


def test_ancilla_initialization_none():
    circ = AncillaInitializationCircuit(None)
    assert isinstance(circ,QuantumCircuit)
    assert isinstance(circ.alpha, Parameter)
    assert circ.num_qubits == 2
