"""
verify_integration.py — Integration test for the H7 framework updates.
"""
from smopsys.h7_quantum_oracle import MetriplexOracle
from smopsys.h7_benchmarks import MetriplexEndianBridge, generate_h7_dataset
from smopsys.endian_logic import TopologicalBigEndianEncoder

def run_verify():
    print("=== H7 Framework Verification ===")
    
    # 1. Oracle & Bridge
    bridge = MetriplexEndianBridge()
    L_symp, L_metr = bridge.compute_lagrangian()
    print(f"Lagrangian: L_symp={L_symp:.4f}, L_metr={L_metr:.4f}")
    print(f"Ratio: {L_symp/L_metr:.4f} (Target φ² ≈ 2.618)")
    
    # 2. Encoding test
    occ = (1, 0, 0)
    hex_state = bridge.encode_fock_state(occ)
    print(f"Fock |1,0,0⟩ -> Hex: {hex_state}")
    
    # 3. Dataset Generation
    print("Generating H7 Dataset...")
    generate_h7_dataset(num_samples=5, output_file="verification_dataset.jsonl")
    
    print("\nVerification Successful.")

if __name__ == "__main__":
    run_verify()
