from qiskit import QuantumCircuit
from qiskit.circuit import Parameter, ParameterVector
from typing import Union, List
import numpy as np
from qadc.utils import Utils
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
        #self.theta = self.noise_to_theta(eta)

    @staticmethod
    def noise_to_theta(noise: Union[float, Parameter]):
        """Converte parametro di rumore η appartenente all'intervallo [0,1] in angolo θ che appartiene ai numeri reali"""
        return Utils.TwoAsinSqrt(noise)

    @staticmethod
    def theta_to_noise(theta: Union[float, Parameter]):
        """Converte angolo θ che appartiene ai numeri reali in parametro di rumore ηappartenente all'intervallo [0,1]"""
        return Utils.SquareSinHalf(theta)


# ==============================
#   CLASSICO
# ==============================
class AmplitudeDampingConvenzioneClassica(BaseAmplitudeDampingCircuit):
    """
    Implementazione del canale di amplitude damping in convenzione classica.

    Descrizione:
        Questo circuito a 2 qubit rappresenta l’implementazione unitaria del canale
        di amplitude damping, dove:
            - qubit 0 = sistema
            - qubit 1 = ambiente

        Il parametro accettato è direttamente un angolo θ ∈ ℝ da usare nella rotazione.
        Per ottenere lo stesso circuito partendo da un parametro fisico η ∈ [0, 1],
        è necessario invocare il metodo `noise_to_theta(eta)` disponibile nella
        classe base, che converte η in θ.

    Parametri:
        eta (float | Parameter | None):
            - Se float o Parameter, viene interpretato come angolo θ.
            - Se None, viene creato automaticamente un parametro simbolico η.

    Restituisce:
        QuantumCircuit:
            Un circuito parametrico che implementa il gate "AD_Classico"
            da applicare a 2 qubit (sistema + ambiente).
    """

    def __init__(self, eta: Union[float, Parameter, None] = None):
        super().__init__(eta, name="AD_Classico")

        # Costruisco il circuito
        self.cry(self.eta, 1, 0)  # Ry su ambiente
        self.cx(0, 1)               # CX: sistema → ambiente


# ==============================
#   QISKIT
# ==============================
class AmplitudeDampingConvenzioneQiskit(BaseAmplitudeDampingCircuit):
    """
    Implementazione del canale di amplitude damping in convenzione Qiskit.

    Descrizione:
        Questo circuito a 2 qubit rappresenta l’implementazione unitaria del canale
        di amplitude damping, seguendo la convenzione di Qiskit:
            - qubit 0 = ambiente
            - qubit 1 = sistema

        Il parametro accettato è direttamente un angolo θ ∈ ℝ da usare nella rotazione.
        Per ottenere lo stesso circuito partendo da un parametro fisico η ∈ [0, 1],
        è necessario invocare il metodo `noise_to_theta(eta)` disponibile nella
        classe base, che converte η in θ.

    Parametri:
        eta (float | Parameter | None):
            - Se float o Parameter, viene interpretato come angolo θ.
            - Se None, viene creato automaticamente un parametro simbolico η.

    Restituisce:
        QuantumCircuit:
            Un circuito parametrico che implementa il gate "AD_Qiskit"
            da applicare a 2 qubit (ambiente + sistema).
    """
    def __init__(self, eta: Union[float, Parameter, None] = None):
        super().__init__(eta, name="AD_Qiskit")

        # Costruisco il circuito
        self.cry(self.eta, 0, 1)  # Ry su ambiente (qubit 0)
        self.cx(1, 0)               # CX: sistema ← ambiente


# ==============================
#   ENCODING
# ==============================
class EncodingCircuit(QuantumCircuit):
    def __init__(self, param: Union[float, Parameter, None] = None):
        """
        Circuito di encoding parametrico su un qubit: RY(θ[0]) → RZ(θ[1])
        """
        if not(param is None) and len(param) != 2:
            raise ValueError("EncodingCircuit richiede esattamente 2 parametri.")

        if param is None:
            param = ParameterVector("θ_de", 2)

        super().__init__(1, name="EncodingCircuit")
        self.ry(param[0], 0)
        self.rz(param[1], 0)


# ==============================
#   DECODING
# ==============================
class DecodingCircuit(QuantumCircuit):
    def __init__(self, param: Union[float, Parameter, None] = None):
        """
        Circuito di decodifica parametrico su un qubit: RZ(θ[0]) → RY(θ[1])
        """
        if not(param is None) and len(param) != 2:
            raise ValueError("DecodingCircuit richiede esattamente 2 parametri.")
        if param is None:
            param = ParameterVector("θ_de", 2)

        super().__init__(1, name="DecodingCircuit")
        self.rz(param[0], 0)
        self.ry(param[1], 0)

# ==============================
#   ANCILLA INITIALIZATION
# ==============================
class AncillaInitializationCircuit(QuantumCircuit):
    """
Circuito per preparare lo stato iniziale con un'ancilla.

Il circuito prepara il qubit di ancilla nello stato target tramite una
rotazione Ry(theta) sull’ancilla, seguita da un entanglement con il
sistema tramite una porta CX.

Parametri:
    theta (float o Parameter, opzionale):
        - Valore reale θ da usare direttamente come angolo di rotazione.
        - Se Parameter, il valore potrà essere assegnato in fase di binding.
        - Se None, viene creato un parametro simbolico 'θ'.

Nota:
    Se si desidera ottenere lo stesso effetto a partire da un
    parametro α ∈ [0,1], è possibile convertire α in θ usando:
        θ = 2 * arcsin(sqrt(α))
    oppure richiamando il metodo `BaseAmplitudeDampingCircuit.noise_to_theta(α)`.
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

        # # Calcola theta corrispondente
        # if isinstance(alpha, Parameter):
        #     theta = 2 * (alpha ** 0.5).arcsin()
        # else:
        #     if not (0 <= alpha <= 1):
        #         raise ValueError("Parametro α deve essere in [0,1].")
        #     theta = 2 * np.arcsin(np.sqrt(alpha))

        # self.theta = theta

        # --- Costruzione del circuito ---
        # Qubit 2 = ancilla
        self.ry(alpha, 1)  # prepara stato diagonale reale
        self.cx(1, 0)      # entanglement ancilla → sistema

        # Nota: ora q0 = sistema, q1 = ambiente, q2 = ancilla

    @staticmethod
    def alpha_to_alpha(alpha: Union[float, Parameter]):
        """Converte parametro di alpha α di miscelanza appartenente all'intervallo [0,1] in angolo θ che appartiene ai numeri reali"""
        return Utils.TwoAsinSqrt(alpha)

    @staticmethod
    def theta_to_alpha(theta: Union[float, Parameter]):
        """Converte angolo θ che appartiene ai numeri reali in parametro di miscelanza ηappartenente all'intervallo [0,1]"""
        return Utils.SquareSinHalf(theta)
    
