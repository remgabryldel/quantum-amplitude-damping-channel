"""
QADC - Quantum Amplitude Damping Channel
Pacchetto per la simulazione, ottimizzazione e studio dei canali di damping quantistici.
"""

from .circuits import circuits
from .simulators import simulator
#from . import optimizers
from .utils import utils

__all__ = ["circuits",
            "simulator",
             # "optimizers",
                "utils"]
