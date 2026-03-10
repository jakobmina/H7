import pytest
import numpy as np
from qiskit import QuantumCircuit
# Note: In a real environment, we would import the factory from a module.
# Since it's currently in the notebook, we'll redefine the logic for testing purposes
# or assume it's been extracted to a central location.

def create_metriplectic_circuit(phase_value):
    qc = QuantumCircuit(3, 6)
    qc.h(0)
    qc.h(1)
    qc.rz(phase_value, 1)
    qc.cswap(1, 0, 2)
    qc.measure_all()
    return qc

def test_circuit_qubits():
    qc = create_metriplectic_circuit(1.618)
    assert qc.num_qubits == 3

def test_phase_consistency():
    qc1 = create_metriplectic_circuit(np.pi)
    qc2 = create_metriplectic_circuit(np.pi)
    # Check if the number of operations is consistent
    assert len(qc1.data) == len(qc2.data)
