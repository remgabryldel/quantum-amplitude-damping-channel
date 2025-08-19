from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from typing import Union, List
import numpy as np
import utils
from utils import Utils
# ==============================
#   BASE AMPLITUDE DAMPING
# ==============================
class BaseAmplitudeDampingCircuit(QuantumCircuit):
    """
    Classe base per un circuito che implementa amplitude damping.
    """

    def __init__(self, eta: Union[float, Parameter, None], name="BaseAmpDamp"):
        super().__init__(2, name=name)  # 2 qubit: sistema + ambiente

        # Se eta=None → parametro simbolico η
        if eta is None:
            eta = Parameter("η")

        self.eta = eta
        self.theta = self.noise_to_theta(eta)

    @staticmethod
    def noise_to_theta(noise: Union[float, Parameter]):
        """Converte parametro di rumore η in angolo θ"""
        return Utils.TwoAsinSqrt(noise)

    @staticmethod
    def theta_to_noise(theta):
        """Converte angolo θ in parametro di rumore η"""
        return Utils.SquareSinHalf(theta)


# ==============================
#   CLASSICO
# ==============================
class AmplitudeDampingConvenzioneClassica(BaseAmplitudeDampingCircuit):
    """
    Convenzione "classica": qubit 0 = sistema, qubit 1 = ambiente.
    """
    def __init__(self, eta: Union[float, Parameter, None] = None):
        super().__init__(eta, name="AD_Classico")

        # Costruisco il circuito
        self.cry(self.theta, 1, 0)  # Ry su ambiente
        self.cx(0, 1)               # CX: sistema → ambiente


# ==============================
#   QISKIT
# ==============================
class AmplitudeDampingConvenzioneQiskit(BaseAmplitudeDampingCircuit):
    """
    Convenzione Qiskit: qubit 0 = ambiente, qubit 1 = sistema.
    """
    def __init__(self, eta: Union[float, Parameter, None] = None):
        super().__init__(eta, name="AD_Qiskit")

        # Costruisco il circuito
        self.cry(self.theta, 0, 1)  # Ry su ambiente (qubit 0)
        self.cx(1, 0)               # CX: sistema ← ambiente


# ==============================
#   ENCODING
# ==============================
class EncodingCircuit(QuantumCircuit):
    def __init__(self, num_params: int = 2):
        """
        Circuito di encoding parametrico su un qubit: RY(θ[0]) → RZ(θ[1])
        """
        if num_params != 2:
            raise ValueError("EncodingCircuit richiede esattamente 2 parametri.")

        self.params_vector = ParameterVector("θ_enc", num_params)

        super().__init__(1, name="EncodingCircuit")
        self.ry(self.params_vector[0], 0)
        self.rz(self.params_vector[1], 0)


# ==============================
#   DECODING
# ==============================
class DecodingCircuit(QuantumCircuit):
    def __init__(self, num_params: int = 2):
        """
        Circuito di decodifica parametrico su un qubit: RZ(θ[0]) → RY(θ[1])
        """
        if num_params != 2:
            raise ValueError("DecodingCircuit richiede esattamente 2 parametri.")

        self.params_vector = ParameterVector("θ_dec", num_params)

        super().__init__(1, name="DecodingCircuit")
        self.rz(self.params_vector[0], 0)
        self.ry(self.params_vector[1], 0)

# ==============================
#   ANCILLA INITIALIZATION
# ==============================
class AncillaInitializationCircuit(QuantumCircuit):
    """
    Circuito per preparare lo stato iniziale con un'ancilla.
    Stato target: (1 - α)|0> + α|1> sul qubit di ancilla.
    Poi viene creato entanglement con il sistema.
    """

    def __init__(self, alpha: Union[float, Parameter, None] = None):
        """
        Parametri:
            alpha: probabilità (reale in [0,1]) o Parameter (se None → simbolico 'α')
        """
        super().__init__(2, name="AncillaInit")  # 3 qubit: sistema, ancilla

        # Se alpha=None, crea parametro simbolico
        if alpha is None:
            alpha = Parameter("α")

        self.alpha = alpha

        # Calcola theta corrispondente
        if isinstance(alpha, Parameter):
            theta = 2 * (alpha ** 0.5).arcsin()
        else:
            if not (0 <= alpha <= 1):
                raise ValueError("Parametro α deve essere in [0,1].")
            theta = 2 * np.arcsin(np.sqrt(alpha))

        self.theta = theta

        # --- Costruzione del circuito ---
        # Qubit 2 = ancilla
        self.ry(theta, 1)  # prepara stato diagonale reale
        self.cx(1, 0)      # entanglement ancilla → sistema

        # Nota: ora q0 = sistema, q1 = ambiente, q2 = ancilla
