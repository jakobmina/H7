"""
h7_endian_experiment.py — Reproducible Quantum Experiment
Demonstrates the bit-mapping bridge and H7 circuit dynamics.
"""

import math
import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit.quantum_info import Statevector
from qiskit.visualization import plot_histogram, plot_bloch_multivector

from smopsys.golden_physics import PHI, PI, classify_particle
from smopsys.endian_logic import InformationLogisticsBridge, BigEndianHexadecimalEncoder, TopologicalBigEndianEncoder

def run_reproduced_experiment(n_param: int = 0):
    print("#" * 80)
    print("  IMPROVED MOD 6: EXACT BIG-ENDIAN HEXADECIMAL")
    print("#" * 80)
    print(f"\nEvaluating state n={n_param}...")

    # 1. Classification
    particle_type = classify_particle(n_param)
    print(f"--- Particle Classification Results ---")
    print(f"  State Index n: {n_param}")
    print(f"  Particle Type: {particle_type}")
    print("\n" + "="*40 + "\n")

    # 2. Quantum Circuit
    print("--- Quantum Circuit Simulation Results ---")
    qc = QuantumCircuit(3, 3)
    
    # Initialization
    for i in range(3): qc.h(i)
    
    # Rotations
    rotation_angle = math.cos(PI * PHI)
    qc.rz(rotation_angle, 0)
    qc.ry(rotation_angle, 1)
    qc.rx(0, 2)
    
    # Entanglement
    qc.cswap(0, 2, 1)
    qc.ccx(2, 1, 0)
    
    qc.barrier()
    
    print("\n  Quantum Circuit Diagram:")
    print(qc.draw())
    
    # State Analysis
    psi = Statevector.from_instruction(qc)
    probs_dict = psi.probabilities_dict()
    print("\n  Probabilities per State:", probs_dict)
    
    # 3. The Bridge (Information Logistics)
    print("\n[BIG-ENDIAN vs LITTLE-ENDIAN BRIDGE]")
    # Example state n=0 -> Index 1, Pair 6...
    entry = TopologicalBigEndianEncoder.topology_entries[n_param]
    packed = TopologicalBigEndianEncoder.pack_topology(**entry)
    be_hex = BigEndianHexadecimalEncoder.to_hex_uint16(packed)
    le_hex = InformationLogisticsBridge.bridge_be_to_le(be_hex)
    
    print(f"  State n={n_param} Packed Value (Hex): {be_hex}")
    print(f"  Bridge Operation (BE -> LE): {be_hex} -> {le_hex}")
    
    # 4. Measurement with Permutation
    # The user used measure([0, 1, 2], [1, 0, 2])
    qc.measure([0, 1, 2], [1, 0, 2])
    
    sim = AerSimulator()
    result = sim.run(qc, shots=1024).result()
    counts = result.get_counts()
    print(f"\n  Final Measured Counts (with Permutation Bridge):")
    print(counts)
    
    # 5. Visualization
    # Note: In a headless environment, we save the plots
    plot_histogram(counts).savefig('h7_histogram.png')
    plot_bloch_multivector(psi).savefig('h7_bloch.png')
    print("\n📊 Plots saved: h7_histogram.png, h7_bloch.png")

if __name__ == "__main__":
    run_reproduced_experiment(n_param=0)
