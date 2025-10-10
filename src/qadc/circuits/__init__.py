# __init__.py

# Import delle classi principali dai moduli interni
from .circuits import (
    BaseAmplitudeDampingCircuit,
    AmplitudeDampingConvenzioneClassica,
    AmplitudeDampingConvenzioneQiskit,
    EncodingCircuit,
    DecodingCircuit,
    AncillaInitializationCircuit
)

# Definisco l'API pubblica del pacchetto
__all__ = [
    "BaseAmplitudeDampingCircuit",
    "AmplitudeDampingConvenzioneClassica",
    "AmplitudeDampingConvenzioneQiskit",
    "EncodingCircuit",
    "DecodingCircuit",
    "AncillaInitializationCircuit"
]
