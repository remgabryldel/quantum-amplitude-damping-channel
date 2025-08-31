import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter,ParameterVector
from qiskit.quantum_info import DensityMatrix, partial_trace, Operator
from qiskit.circuit.library import CRYGate, RYGate, RZGate,CXGate
from IPython.display import display

from qadc.utils import Utils as u
from qadc.circuits import (
    AmplitudeDampingConvenzioneClassica as ADCC,
    AmplitudeDampingConvenzioneQiskit as ADCQ,
    AncillaInitializationCircuit as AIC,
    EncodingCircuit as EC,
    DecodingCircuit as DC,
    )


# ==============================
#   CHECK AMPLITUDE DAMPING
# ==============================
@pytest.mark.parametrize("eta", [round(i * 0.1, 1) for i in range(11)])
@pytest.mark.parametrize("alpha", [round(i * 0.1, 1) for i in range(11)])
def test_adc_classico_matches_theory(eta, alpha):
    """
    Verifica che il circuito in convenzione classica riproduca
    il modello teorico di amplitude damping per vari eta, alpha.
    """

    qc = QuantumCircuit(2, name="AD_classico")
    qc.compose(ADCC(ADCC.noise_to_theta(eta)), [0, 1],inplace = True)

    # Kraus teorici
    kraus_op_adc_0 = np.array([[1, 0], [0, np.sqrt(1 - eta)]])
    kraus_op_adc_0_dag = Operator(kraus_op_adc_0).adjoint().data
    kraus_op_adc_1 = np.array([[0, np.sqrt(eta)], [0, 0]])
    kraus_op_adc_1_dag = Operator(kraus_op_adc_1).adjoint().data

    # stato iniziale totale
    rho_in_total = u.initial_mixed_state_real_density_matrix(alpha)

    # stato ridotto del sistema
    rho_in_sis = partial_trace(rho_in_total, [0]).to_operator()

    # evoluzione teorica
    rho_out_theory = DensityMatrix(
        kraus_op_adc_0 @ rho_in_sis.data @ kraus_op_adc_0_dag
        + kraus_op_adc_1 @ rho_in_sis.data @ kraus_op_adc_1_dag
    )

    # evoluzione circuito classico
    rho_out_total_cir = rho_in_total.evolve(qc)
    rho_out_cir = partial_trace(rho_out_total_cir, [0])

    # confronto con tolleranza
    dist_classico = np.linalg.norm(
        rho_out_theory.data - rho_out_cir.data, ord="fro"
    )
    if not((rho_out_cir.to_operator().equiv(rho_out_theory.to_operator()))):
            print(f"Rho diversi per eta = {eta}, alpha = {alpha}")
            print("matrice densita circuito convenzione classica")
            display(rho_out_cir.draw("latex"))
            print("matrice densita circuito calcolato teoricamente")
            display(rho_out_theory.draw("latex"))
            print(f"Distanza dei due stati teorico e conv. clas.: {dist_classico:.6f}")
    else:
        print(f"distanza = {dist_classico}")
        if(eta == 0) or (eta == 0.5) or (eta==1):
            print(f" eta = {eta}, alpha = {alpha}")
            print("matrice densita circuito convenzione classica")
            display(rho_out_cir.draw("latex"))
            print("matrice densita circuito calcolato teoricamente")
            display(rho_out_theory.draw("latex"))
    assert dist_classico < 1e-8, f"Classico non coincide per eta={eta}, alpha={alpha}"

@pytest.mark.parametrize("eta", [round(i * 0.1, 1) for i in range(11)])
@pytest.mark.parametrize("alpha", [round(i * 0.1, 1) for i in range(11)])
def test_qiskit_matches_theory(eta,alpha):
    """Controlla che la convenzione qiskit coincida con il modello teorico
    per valori notevoli di eta."""
    qc = QuantumCircuit(2, name="AD_classico")
    qc.compose(ADCQ(ADCQ.noise_to_theta(eta)), [0, 1],inplace = True)

    # Kraus teorici
    kraus_op_adc_0 = np.array([[1, 0], [0, np.sqrt(1 - eta)]])
    kraus_op_adc_0_dag = Operator(kraus_op_adc_0).adjoint().data
    kraus_op_adc_1 = np.array([[0, np.sqrt(eta)], [0, 0]])
    kraus_op_adc_1_dag = Operator(kraus_op_adc_1).adjoint().data

    # stato iniziale totale
    rho_in_total_q = u.reverse_qargs_density_matrix(u.initial_mixed_state_real_density_matrix(alpha))

    # stato ridotto del sistema
    rho_in_sis_q = partial_trace(rho_in_total_q, [1]).to_operator()

    # evoluzione teorica
    rho_out_theory = DensityMatrix(
        kraus_op_adc_0 @ rho_in_sis_q.data @ kraus_op_adc_0_dag
        + kraus_op_adc_1 @ rho_in_sis_q.data @ kraus_op_adc_1_dag
    )

    # evoluzione circuito classico
    rho_out_total_cir_q = rho_in_total_q.evolve(qc)
    rho_out_cir_q = partial_trace(rho_out_total_cir_q, [1])

    # confronto con tolleranza
    dist_qiskit = np.linalg.norm(
        rho_out_theory.data - rho_out_cir_q.data, ord="fro"
    )
    if not((rho_out_cir_q.to_operator().equiv(rho_out_theory.to_operator()))):
            print(f"Rho diversi per eta = {eta}, alpha = {alpha}")
            print("matrice densita circuito convenzione classica")
            display(rho_out_cir_q.draw("latex"))
            print("matrice densita circuito calcolato teoricamente")
            display(rho_out_theory.draw("latex"))
            print(f"Distanza dei due stati teorico e conv. clas.: {dist_qiskit:.6f}")
    else:
        print(f"distanza = {dist_qiskit}")
        if(eta == 0) or (eta == 0.5) or (eta==1):
            print(f" eta = {eta}, alpha = {alpha}")
            print("matrice densita circuito convenzione classica")
            display(rho_out_cir_q.draw("latex"))
            print("matrice densita circuito calcolato teoricamente")
            display(rho_out_theory.draw("latex"))
    assert dist_qiskit < 1e-8, f"Classico non coincide per eta={eta}, alpha={alpha}"

@pytest.mark.parametrize("alpha", [round(i * 0.1, 1) for i in range(11)])
def test_ancilla_initialization_matches_theory(alpha):
    """
    Verifica che il circuito di inizializzazione con ancilla
    riproduca lo stato teorico (1-α)|0><0| + α|1><1|
    per vari valori di α.
    """
    qc = QuantumCircuit(3)
    qc.compose(AIC(AIC.alpha_to_theta(alpha)), [0,2], inplace = True)

    # stato teorico (matrice densità su 1 qubit)
    rho_teorical = DensityMatrix(np.array([[1 - alpha, 0], [0, alpha]]))

    # stato ottenuto dal circuito
    rho_out_cir = DensityMatrix.from_label("000").evolve(qc)

    # riduzioni: 
    # - qubit 0,1 fuori → rimane l'ancilla (convenzione classica)
    rho_out_class = partial_trace(rho_out_cir, [0, 1])
    # - qubit 1,2 fuori → rimane il sistema (convenzione qiskit)
    rho_out_qi = partial_trace(rho_out_cir, [1, 2])

    # distanze di Frobenius
    distance = np.linalg.norm(rho_teorical.data - rho_out_class.data, ord="fro")
    distance1 = np.linalg.norm(rho_teorical.data - rho_out_qi.data, ord="fro")

    if not((rho_out_class.to_operator().equiv(rho_teorical.to_operator())) and (rho_out_qi.to_operator().equiv(rho_teorical.to_operator()))):
            print(f"Rho diversi per alpha = {alpha}")
            print("matrice densita circuito convenzione classica")
            display(rho_out_class.draw("latex"))
            print("matrice densita circuito convenzione qiskit")
            display(rho_out_qi.draw("latex"))
            print("matrice densita circuito calcolato teoricamente")
            display(rho_teorical.draw("latex"))
            print(f"Distanza dei due stati teorico e conv. clas.: {distance:.6f}, teorico e conv. qiskit: {distance1:.6f}")
            print("matrici diverse tra loro")
    else:
        print(f"distanza = {distance}, distanza1 = {distance1}")
        if(alpha == 0) or (alpha == 0.5) or (alpha==1):
            print(f"alpha = {alpha}")
            print("matrice densita circuito convenzione classica")
            display(rho_out_class.draw("latex"))
            print("matrice densita circuito convenzione qiskit")
            display(rho_out_qi.draw("latex"))
            print("matrice densita circuito calcolato teoricamente")
            display(rho_teorical.draw("latex"))

    # check equivalenza
    assert rho_out_class.to_operator().equiv(rho_teorical.to_operator()), (
        f"Convenzione classica non coincide per alpha={alpha}, distanza={distance}"
    )
    assert rho_out_qi.to_operator().equiv(rho_teorical.to_operator()), (
        f"Convenzione qiskit non coincide per alpha={alpha}, distanza={distance1}"
    )

@pytest.mark.parametrize("t0", [(i * np.pi * 0.25) % (2*np.pi) for i in range(5)])
@pytest.mark.parametrize("t1", [(i * np.pi * 0.25) % (2*np.pi) for i in range(5)])
@pytest.mark.parametrize("t2", [(i * np.pi * 0.25) % (2*np.pi) for i in range(5)])
@pytest.mark.parametrize("t3", [(i * np.pi * 0.25) % (2*np.pi) for i in range(5)])
def test_eradc_operator_and_density(t0, t1, t2, t3):
    """
    Verifica che il circuito ERADC (classico e qiskit) coincida
    con il modello teorico sia a livello di operatore che di evoluzione densità.
    """

    # Parametri
    alpha = 1
    theta = np.pi/3 

    # Circuiti
    qcA = QuantumCircuit(3, name="init_state")
    qc0 = QuantumCircuit(2, name="classical")
    qc1 = QuantumCircuit(2, name="qiskit")

    qcA.append(AIC(AIC.alpha_to_theta(alpha)), [0,2])
    qc0.append(EC([t0,t1]), [1])
    qc0.append(ADCC(theta),[0,1])
    qc0.append(DC([t2,t3]), [1])
    qc1.append(EC([t0,t1]), [0])
    qc1.append(ADCQ(theta),[0,1])
    qc1.append(DC([t2,t3]), [0])

    # --- Operator check ---
    identity = np.array([[1,0],[0,1]])
    kraus_op_adc_0 = np.array([[1, 0],[0,np.sqrt(1-u.SquareSinHalf(theta))]])
    kraus_op_adc_1 = np.array([[0, np.sqrt(u.SquareSinHalf(theta))],[0,0]])
    CRy = DensityMatrix(CRYGate(theta).to_matrix()).reverse_qargs().data
    CNotRev = CXGate().to_matrix()
    Epsilon_AD_class = CNotRev @ CRy
    
    Ry_0 = RYGate(t0).to_matrix()
    Rz_0 = RZGate(t1).to_matrix()
    Rz_1 = RZGate(t2).to_matrix()
    Ry_1 = RYGate(t3).to_matrix()
    En = Rz_0 @ Ry_0
    De = Ry_1 @ Rz_1

    # Operatori per la rappresentazione dell'operatore del sistema-ambiente del circuito
    En_total_qi = np.kron(identity,En)
    De_total_qi = np.kron(identity,De)
    De_total_class = np.kron(De,identity)
    En_total_class = np.kron(En,identity)
    operatoreCircuito_class = De_total_class @ Epsilon_AD_class @ En_total_class
    operatoreCircuito_qi = DensityMatrix(operatoreCircuito_class).reverse_qargs().data
    # verifico la coincidenza dell'operatore che rappresenta il sistema 
    # Calcola la distanza di Frobenius tra le due matrici
    operatoreQc0 = Operator.from_circuit(qc0).to_matrix()
    operatoreQc1 = Operator.from_circuit(qc1).to_matrix()
    dist_op_class = np.linalg.norm(operatoreCircuito_class - operatoreQc0, ord='fro')
    dist_op_qiskit = np.linalg.norm(operatoreCircuito_qi - operatoreQc1, ord='fro')

    if not((DensityMatrix(operatoreQc0).to_operator().equiv(DensityMatrix(operatoreCircuito_class).to_operator()))
    and (DensityMatrix(operatoreQc1).to_operator().equiv(DensityMatrix(operatoreCircuito_qi).to_operator()))):
        print(f"Operatori diversi per t0 = {t0}, t01 = {t1},t2 = {t2},t3 = {t3}")
        print("Operatore circuito convenzione classica")
        display(DensityMatrix(operatoreQc0).draw("latex"))
        print("Operatore circuito convenzione classica calcolato teoricamente")
        display(DensityMatrix(operatoreCircuito_class).draw("latex"))
        print("Operatore circuito convenzione qiskit")
        display(DensityMatrix(operatoreQc1).draw("latex"))
        print("Operatore circuito convenzione qiskit calcolato teoricamente")
        display(DensityMatrix(operatoreCircuito_qi).draw("latex"))
        print(f"Distanza dei due operatori teorico e conv. clas.: {distance:.6f}, teorico e conv. qiskit: {distance1:.6f}")
        raise("matrici diverse tra loro")
    else:
        if(t0 == t3) and (t1 == t2) and (t0 == t3):
            print(f"distanza = {dist_op_class}, distanza1 = {dist_op_qiskit}")
            print(f"Operatori per t0 = {t0}, t01 = {t1},t2 = {t2},t3 = {t3}")
            print("Operatore circuito convenzione classica")
            display(DensityMatrix(operatoreQc0).draw("latex"))
            # print("matrice densita circuito convenzione qiskit")
            # display(rho_out_cir_q.draw("latex"))
            print("Operatore circuito convenzione classica calcolato teoricamente")
            display(DensityMatrix(operatoreCircuito_class).draw("latex"))


    assert np.allclose(dist_op_class, 0, atol=1e-8), f"Operator classico diverso, distanza={dist_op_class}"
    assert np.allclose(dist_op_qiskit, 0, atol=1e-8), f"Operator qiskit diverso, distanza={dist_op_qiskit}"

    # --- Density check ---
    rho_in = u.initial_mixed_state_real_density_matrix(alpha)

    kraus_op_0 = De @ kraus_op_adc_0 @ En
    kraus_op_0_dag = Operator(kraus_op_0).adjoint().data
    kraus_op_1 = De @ kraus_op_adc_1 @ En
    kraus_op_1_dag = Operator(kraus_op_1).adjoint().data
    rho_out_theory = DensityMatrix(kraus_op_0 @ partial_trace(rho_in,[0]).data @ kraus_op_0_dag + kraus_op_1 @ partial_trace(rho_in,[0]).data @ kraus_op_1_dag)


    qc0_full = qcA.compose(qc0, [1,2])
    qc1_full = qcA.compose(qc1, [0,1])
    rho_out_class = partial_trace(DensityMatrix.from_label("000").evolve(qc0_full), [0,1])
    rho_out_qiskit = partial_trace(DensityMatrix.from_label("000").evolve(qc1_full), [1,2])

    dist_rho_class = np.linalg.norm(rho_out_theory.data - rho_out_class.data, ord="fro")
    dist_rho_qiskit = np.linalg.norm(rho_out_theory.data - rho_out_qiskit.data, ord="fro")

    if not((rho_out_class.to_operator().equiv(rho_out_theory.to_operator())) and (rho_out_qiskit.to_operator().equiv(rho_out_theory.to_operator()))):
        print(f"Rho diversi per t0 = {t0}, t01 = {t1},t2 = {t2},t3 = {t3}")
        print("matrice densita circuito output convenzione classica")
        display(rho_out_class.draw("latex"))
        print("matrice densita circuito output convenzione qiskit")
        display(rho_out_qiskit.draw("latex"))
        print("matrice densita circuito output calcolato teoricamente")
        display(rho_out_theory.draw("latex"))
        print(f"Distanza dei due stati teorico e conv. clas.: {dist_rho_class:.6f}, teorico e conv. qiskit: {dist_rho_qiskit:.6f}")
        raise("matrici diverse tra loro")
    else:
        if(t0 == t3) and (t1 == t2) and (t0 == t3):
            print(f"distanza = {dist_rho_class}, distanza1 = {dist_rho_qiskit}")
            print(f" t0 = {t0}, t01 = {t1},t2 = {t2},t3 = {t3}")
            print("matrice densita output circuito convenzione classica")
            display(rho_out_class.draw("latex"))
            # print("matrice densita circuito convenzione qiskit")
            # display(rho_out_qiskit.draw("latex"))
            print("matrice densita output circuito calcolato teoricamente")
            display(rho_out_theory.draw("latex"))
    assert rho_out_class.to_operator().equiv(rho_out_theory.to_operator()), (
        f"Rho classico diverso per t0={t0},t1={t1},t2={t2},t3={t3}, distanza={dist_rho_class}"
    )
    assert rho_out_qiskit.to_operator().equiv(rho_out_theory.to_operator()), (
        f"Rho qiskit diverso per t0={t0},t1={t1},t2={t2},t3={t3}, distanza={dist_rho_qiskit}"
    )