from abc import ABC, abstractmethod
from qiskit import transpile
from qiskit_aer import *
from qiskit_aer.noise import NoiseModel
from qiskit_ibm_runtime import SamplerV2

# === Interfaccia astratta ===
class AbstractSimulator(ABC):
    @abstractmethod
    def run(self, pubs):
        """Esegue il circuito con eventuali parametri e restituisce il risultato."""
        pass


# === Simulatore ideale (senza rumore, senza vincoli hardware) ===
class IdealSimulator(AbstractSimulator):
    def __init__(self):
        self._sampler = SamplerV2(mode= AerSimulator())

    def run(self, pubs):
        return self._sampler.run(pubs).result()


# === Simulatore rumoroso (NoiseModel.from_backend) ===
#   - Usa un AerSimulator generico con rumore estratto dal backend
#   - Non vincola la topologia hardware (ma conviene transpilare lo stesso)
class NoisySimulator(AbstractSimulator):
    def __init__(self, backend):
        _noise_model = NoiseModel.from_backend(backend)
        self.backend = backend
        self._sampler = SamplerV2(mode = AerSimulator(noise_model = _noise_model))

    def run(self, pubs):
        # Transpile sul backend, così il rumore corrisponde ai gate effettivi
        pubs = [(transpile(circuit, backend=self.backend),param,shot) for circuit,param,shot in pubs]
        return self._sampler.run(pubs).result()


# === Simulatore fedele al backend reale (AerSimulator.from_backend) ===
#   - Replica rumore + topologia + basis gates del backend reale
class BackendLikeSimulator(AbstractSimulator):
    def __init__(self, backend):
        from qiskit_aer import AerSimulator
        self.backend = backend
        self._sampler = SamplerV2(mode=AerSimulator.from_backend(backend))

    def run(self, pubs):
        # Transpile sul backend, così il rumore corrisponde ai gate effettivi
        pubs = [(transpile(circuit, backend=self.backend),param,shot) for circuit,param,shot in pubs]
        return self._sampler.run(pubs).result()


# # === Sampler su hardware reale (IBM Runtime) ===
# class RealBackendSimulator(AbstractSimulator):
#     def __init__(self, backend):
#         self.backend = backend
#         self._sampler = SamplerV2(backend=backend)

#     def run(self, circuit, param_values=None, shots=None):
#         transpiled_circ = transpile(circuit, backend=self.backend)
#         pubs = [(transpiled_circ, param_values, shots)]
#         return self._sampler.run(pubs)
