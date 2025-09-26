import numpy as np
from qiskit.quantum_info import DensityMatrix
from qiskit.circuit import Parameter
from functools import reduce

class Utils:
    """
    Classe di utilità per funzioni matematiche, conversioni
    e supporto al calcolo della funzione costo.
    """

    # --------------------------
    # Normalizzazione conteggi
    # --------------------------
    @staticmethod
    def normalize_counts(counts: dict) -> dict:
        return {key: counts.get(key, 0) for key in ['0', '1']}

    # --------------------------
    # Funzione costo (error rate)
    # --------------------------
    @staticmethod
    def cost_function(counts: list) -> float:
        """
        counts: lista di dizionari {esito: conteggi}
                es. [{"0": 900, "1": 100}, {"0": 200, "1": 800}, {"2": 50}]
        Ritorna: 1- somma_per_ogni_chiave_k(max_valore_della_chiave_k_su_tutti_gli_elementi(counts)/somma_totale_numero_esperimenti) 
        """
        if not counts:
            return {}

        # Unione di tutte le chiavi presenti
        keys = set().union(*(d.keys() for d in counts))
        #print(keys)
        # Somma totale di tutti i conteggi
        all_totals = sum(sum(d.values()) for d in counts)
        if all_totals == 0:
            return {k: 0.0 for k in keys}
        # print(all_totals)
        # Max della frequenza normalizzata sul totale globale
        max_freq = {
            k: max(d.get(k, 0) / all_totals for d in counts)
            for k in keys
        }
        return 1-sum(max_freq.values())

        

    # --------------------------
    # Merge di dizionari
    # --------------------------
    @staticmethod
    def merge_dict(dicts: list) -> dict:
        return reduce(
            lambda acc, d: {k: acc.get(k, []) + [d[k]] for k in d},
            dicts,
            {}
        )

    # --------------------------
    # Conversione noise <-> theta
    # --------------------------
    @staticmethod
    def TwoAsinSqrt(x):
        """
        Applica la funzione di 2*arcsin(Sqrt(x)) a x.
        Supporta numeri e Qiskit Parameter.
        """
    @staticmethod
    def TwoAsinSqrt(x):
        """
        Applica la funzione di 2*arcsin(sqrt(x)) a x.
        Supporta numeri, Parameter e None (simbolico).
        """
        if x is None:
            return Parameter("θ")   # o direttamente un simbolo generico
    
        if not isinstance(x, Parameter):
            if 0 <= x <= 1:
                return 2 * np.arcsin(np.sqrt(x))
            else:
                raise ValueError("Parametro x fuori dall'intervallo [0,1]")
        else:
            return 2 * (x ** 0.5).arcsin()

    @staticmethod
    def SquareSinHalf(x):
        """
        Applica la funzione di sin((x/2))^2 a x.
        Supporta numeri e Qiskit Parameter.
        """
        if not isinstance(x, Parameter):
            if x is not None:
                return np.square(np.sin(x / 2))
        else:
            return (x / 2).sin() ** 2
        raise ValueError("Parametro theta non valido: deve appartenere all'insieme dei reali o essere un Parameter")

    # --------------------------
    # Reverse Qargs (Density Matrix)
    # --------------------------
    @staticmethod
    def reverse_qargs_density_matrix(rho):
        """
        Inverte l'ordine dei qubit (little-endian vs big-endian) in una matrice densità.

        Questo è utile quando si vuole cambiare l'ordinamento dei qubit nel prodotto tensore,
        ad esempio passando da |q0⟩ ⊗ |q1⟩ a |q1⟩ ⊗ |q0⟩.

        Parametri:
            rho (DensityMatrix): matrice densità di input (multi-qubit)

        Restituisce:
            DensityMatrix: matrice densità con ordine dei qubit invertito
        """
        return DensityMatrix(rho.to_operator().reverse_qargs().to_matrix())
    
    # -----------------------------------------------------------------
    # create a initial state density matrix of 2 qubit (Density Matrix)
    # -----------------------------------------------------------------
    @staticmethod
    def initial_mixed_state_real_density_matrix(_alpha):
        """
        Crea una matrice densità a 2 qubit come stato iniziale, in cui:
        - Il qubit di sistema (qubit 0 S_1) è in uno stato misto reale parametrizzato da alpha:
            ρ_sis = (1 - α)|0⟩⟨0| + α|1⟩⟨1|
        - Il qubit di ambiente (qubit 1 E_1) è inizializzato nello stato puro |0⟩⟨0|
        - L'output è il prodotto tensore ρ_sis ⊗ ρ_env, rappresentato come oggetto DensityMatrix

        Parametri:
            _alpha (float): parametro di mescolanza ∈ [0, 1]; 

        Restituisce:
            DensityMatrix: stato iniziale a 2 qubit (sistema + ambiente)
        """
        # Parametro di mescolanza
        alpha = _alpha
        if not (0 <= _alpha <= 1):
            raise ValueError("Valore non valido di alpha")

        #creo una matrice in forma array[array[,...],...]
        rho_sis_matrix = [[1 - alpha, 0],[0, alpha]]
        
        # Creo lo stato misto da rho_0_matrix per qubit 0
        rho_sis = DensityMatrix(rho_sis_matrix)

        # creo lo stao del qubit 1 (ambiente) in |0>
        rho_env = DensityMatrix.from_label('0')

        # Stato iniziale totale a 2 qubit
        rho_in = rho_sis.tensor(rho_env)

        return rho_in
