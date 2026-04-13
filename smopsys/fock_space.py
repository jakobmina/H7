"""
fock_space.py — Fock Space Basis and Configuration
Part of the H7 Metriplectic Framework.

Maps discrete occupation numbers |n0, n1, n2...⟩ into Hilbert space.
"""

import numpy as np
from typing import Tuple, List, Iterator, Optional
from dataclasses import dataclass, field
from enum import Enum

class OccupationMode(Enum):
    """Enumeration of occupation number modes in Fock space."""
    BOSONIC = "bosonic"
    FERMIONIC = "fermionic"

@dataclass
class FockConfig:
    """Configuration for Fock space instantiation."""
    n_modes: int = 3
    n_max: int = 1  # max particles per mode (n_max=1 for qubits/fermions)
    mode: OccupationMode = OccupationMode.BOSONIC

class FockBasis:
    """
    Fock space basis generator.
    Creates a basis of states |n1, n2, ..., n_m⟩ where n_i <= n_max.
    """
    def __init__(self, config: FockConfig = None):
        if config is None:
            config = FockConfig()
        self.config = config
        self.basis_states = self._generate_basis()
        self.dim = len(self.basis_states)

    def _generate_basis(self) -> List[Tuple[int, ...]]:
        import itertools
        ranges = [range(self.config.n_max + 1) for _ in range(self.config.n_modes)]
        return list(itertools.product(*ranges))

    def get_index(self, state: Tuple[int, ...]) -> int:
        return self.basis_states.index(state)

    def get_state(self, index: int) -> Tuple[int, ...]:
        return self.basis_states[index]

    def __iter__(self) -> Iterator[Tuple[int, ...]]:
        return iter(self.basis_states)

    def __len__(self) -> int:
        return self.dim
