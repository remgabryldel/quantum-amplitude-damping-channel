import pytest
from qiskit import QuantumCircuit
from qiskit_ibm_runtime.fake_provider import FakeManilaV2
from qadc.simulators import IdealSimulator, NoisySimulator, BackendLikeSimulator


@pytest.fixture
def simple_circuit():
    """Circuito base per i test: H seguito da misura."""
    qc = QuantumCircuit(1, 1)
    qc.h(0)
    qc.measure(0, 0)
    return qc


def test_ideal_simulator(simple_circuit):
    sim = IdealSimulator()
    pubs = [(simple_circuit, {}, 1000)]  # circuito, parametri vuoti, 1000 shots
    result = sim.run(pubs)
    assert result is not None
    counts = result[0].data.c.get_counts()
    assert len(result) >= 1
    assert isinstance(counts, dict)
    # almeno una distribuzione deve esistere



def test_noisy_simulator(simple_circuit):
    backend = FakeManilaV2() # backend fittizio con rumore
    sim = NoisySimulator(backend)
    pubs = [(simple_circuit, {}, 1000)]
    result = sim.run(pubs)
    counts = result[0].data.c.get_counts()
    assert result is not None
    # anche qui deve esserci almeno un risultato
    assert len(result) >= 1
    assert isinstance(counts, dict)


def test_backend_like_simulator(simple_circuit):
    backend = FakeManilaV2()  # backend fittizio con rumore + topologia
    sim = BackendLikeSimulator(backend)
    pubs = [(simple_circuit, {}, 1000)]
    result = sim.run(pubs)
    counts = result[0].data.c.get_counts()
    assert result is not None
    assert len(result) >= 1
    assert isinstance(counts, dict)

