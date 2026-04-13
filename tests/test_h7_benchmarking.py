import pytest
import numpy as np
import os
from smopsys.h7_quantum_oracle import MetriplexOracle, MetriplexConfig
from smopsys.fock_space import FockBasis, FockConfig
from smopsys.endian_logic import TopologicalBigEndianEncoder, BigEndianHexadecimalEncoder
from smopsys.h7_benchmarks import MetriplexEndianBridge, generate_h7_dataset

def test_fock_basis_generation():
    config = FockConfig(n_modes=2, n_max=1)
    basis = FockBasis(config)
    assert len(basis) == 4
    assert (0, 0) in basis
    assert (1, 1) in basis

def test_endian_packing_unpacking():
    val = TopologicalBigEndianEncoder.pack_topology(1, 6, 0, 0, 1, 0)
    unpacked = TopologicalBigEndianEncoder.unpack_topology(val)
    assert unpacked['index'] == 1
    assert unpacked['pair'] == 6
    assert unpacked['winding'] == 0
    assert unpacked['ternary_weight'] == 1

def test_bridge_lagrangian():
    bridge = MetriplexEndianBridge()
    L_symp, L_metr = bridge.compute_lagrangian()
    assert L_symp > 0
    assert L_metr > 0
    # Ratio might vary based on energy profile, but should be finite
    assert L_symp / L_metr > 0

def test_dataset_generation():
    output = "test_dataset.jsonl"
    generate_h7_dataset(num_samples=10, output_file=output)
    assert os.path.exists(output)
    with open(output, 'r') as f:
        lines = f.readlines()
        assert len(lines) == 10
    os.remove(output)

def test_hilbert_oracle():
    oracle = MetriplexOracle()
    fock = FockBasis(FockConfig(n_modes=3, n_max=1))
    q_oracle = oracle.to_hilbert_oracle(fock)
    
    # Test vector (8 states)
    vec = np.ones(8, dtype=complex) / np.sqrt(8)
    transformed = q_oracle(vec)
    
    # Probabilities should be conserved (unitary phase shift)
    np.testing.assert_allclose(np.abs(transformed)**2, np.abs(vec)**2)
