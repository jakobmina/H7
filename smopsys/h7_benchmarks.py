"""
h7_benchmarks.py — H7 Metriplectic Benchmarking & Dataset Generation

Integrates the MetriplexEndianBridge to evaluate the H7 logic and 
generate synthetic datasets. Uses dual-endianness as the "physical bridge".
"""

import json
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, List, Optional, Any

from smopsys.h7_quantum_oracle import MetriplexOracle, MetriplexConfig, H7Conservation
from smopsys.fock_space import FockBasis, FockConfig
from smopsys.endian_logic import (
    TopologicalBigEndianEncoder, 
    BigEndianHexadecimalEncoder,
    LittleEndianHexadecimalEncoder,
    InformationLogisticsBridge
)
from smopsys.golden_physics import (
    PHI, 
    golden_operator, 
    o_n_to_phase_fragment,
    classify_particle,
    classify_cognitive_layer
)

GOLDEN_RATIO_SQUARED = PHI ** 2

class MetriplexEndianBridge:
    """
    The Information Logistics Bridge.
    Uses Big-Endian for Symplectic (H) and Little-Endian for Metric (S).
    """

    def __init__(
        self,
        oracle: Optional[MetriplexOracle] = None,
        fock: Optional[FockBasis] = None,
    ):
        self.oracle = oracle or MetriplexOracle(MetriplexConfig())
        self.fock = fock or FockBasis(FockConfig(n_modes=3, n_max=2))
        self._encoder = TopologicalBigEndianEncoder
        self._be_hex = BigEndianHexadecimalEncoder
        self._le_hex = LittleEndianHexadecimalEncoder
        self._bridge = InformationLogisticsBridge

    def _get_topo_params(self, group: str) -> Tuple[int, int, int]:
        """Returns (winding, mapping, weight) based on group."""
        if group == 'A':
            return 0, 0, 1
        elif group == 'B':
            return 2, 1, -1
        return 0, 0, 0

    def compute_lagrangian(self) -> Tuple[float, float]:
        """
        Metriplectic Lagrangian calculation (Rule 3.1).
        L_symp: Derived from Big-Endian (Systemic/Conservative)
        L_metr: Derived from Little-Endian (Dissipative/Entropy)
        """
        L_symp = 0.0
        L_metr = 0.0

        for entry in self._encoder.topology_entries:
            p = entry['index']
            if p < self.oracle.p_min or p > self.oracle.p_max:
                continue
            _, _, energy = self.oracle.forward(p)

            # Rule: Winding 0 -> Symplectic, Winding 2 -> Metric
            if entry['winding'] == 0:
                L_symp += energy
            elif entry['winding'] == 2:
                L_metr += energy

        return L_symp, L_metr

    def encode_fock_state(self, occupation: Tuple[int, ...]) -> Dict[str, str]:
        """
        Encodes Fock state into a dual-endian pair (The Bridge).
        """
        p = self.oracle._occupation_to_momentum(occupation)
        group, _, _ = self.oracle.forward(p)

        winding, mapping, weight = self._get_topo_params(group)

        h7_state = (p - 1) % 8
        o_n_val = golden_operator(h7_state)
        phase_frag = o_n_to_phase_fragment(o_n_val)

        packed_val = self._encoder.pack_topology(
            index=p,
            pair=7 - p if 1 <= 7 - p <= 6 else p,
            winding=winding,
            mapping=mapping,
            ternary_weight=weight,
            discrete_phase_fragment=phase_frag
        )

        be_hex = self._be_hex.to_hex_uint16(packed_val)
        le_hex = self._bridge.bridge_be_to_le(be_hex)

        return {
            'be_hex': be_hex,
            'le_hex': le_hex
        }

    def full_state_report(self, occupation: Tuple[int, ...]) -> Dict[str, Any]:
        p = self.oracle._occupation_to_momentum(occupation)
        group, _, energy = self.oracle.forward(p)
        h7_state = (p - 1) % 8
        
        winding, _, weight = self._get_topo_params(group)
        
        L_symp, L_metr = self.compute_lagrangian()
        encoding = self.encode_fock_state(occupation)
        
        # Rule 2.1 Classification
        p_type = classify_particle(h7_state)
        
        # 3-Level Cognitive Classification
        cog_layer = classify_cognitive_layer(weight, winding)
        
        return {
            'occupation': occupation,
            'momentum': p,
            'group': group,
            'energy': energy,
            'h7_state': h7_state,
            'particle_type': p_type,
            'cognitive_layer': cog_layer,
            'be_hex': encoding['be_hex'],
            'le_hex': encoding['le_hex'],
            'L_symp': L_symp,
            'L_metr': L_metr,
            'L_ratio': L_symp / L_metr if L_metr != 0 else 0,
            'ratio_ok': abs((L_symp/L_metr) - GOLDEN_RATIO_SQUARED) < 1.0 if L_metr != 0 else False
        }

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.bool_): return bool(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super(NpEncoder, self).default(obj)

def generate_h7_dataset(num_samples=10000, output_file="h7_logistics_dataset.jsonl"):
    """Generates a massive synthetic dataset using the Information Logistics Bridge."""
    bridge = MetriplexEndianBridge()
    states = list(bridge.fock.basis_states)
    
    print(f"Starting generation of {num_samples} samples...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            occ = tuple(random.choice(states))
            report = bridge.full_state_report(occ)
            
            # Format the "Assistant" response to look like a logical derivation
            prompt = f"Analyze Fock state {occ} through the H7 Endian Bridge and Cognitive Layers."
            answer = (
                f"Topological State: [Index={report['momentum']}, Pair={7-report['momentum']}, "
                f"Winding={0 if report['group'] == 'A' else 2}, Mapping={0 if report['group'] == 'A' else 1}]. "
                f"Cognitive Layer: {report['cognitive_layer'].upper()}. "
                f"Logistics: BE={report['be_hex']} | LE={report['le_hex']}. "
                f"Classification: {report['particle_type'].upper()}."
            )

            record = {
                "id": i,
                "system": "H7 3-Level Cognitive Logistics & Metriplectic Dynamics.",
                "user": prompt,
                "assistant": answer,
                "metadata": {
                    "momentum": report['momentum'],
                    "h7_state": report['h7_state'],
                    "cognitive_layer": report['cognitive_layer'],
                    "particle_type": report['particle_type'],
                    "be_hex": report['be_hex'],
                    "le_hex": report['le_hex'],
                    "ratio": report['L_ratio']
                }
            }
            f.write(json.dumps(record, cls=NpEncoder) + '\n')
            
            if (i + 1) % 2000 == 0:
                print(f"  Progress: {i + 1}/{num_samples} samples generated.")
    
    print(f"✅ Generated {num_samples} samples in {output_file}")

if __name__ == "__main__":
    generate_h7_dataset(10000)
    print("Benchmarking completed.")
