"""
golden_physics.py — Golden Operator O_n and Phase Fragments
Part of the H7 Metriplectic Framework.

Implements Rule 2.1: Foundation modulated by the Golden Operator.
"""

import math
import numpy as np

# Mandato Metripléxico: Analogía de Arquímedes y Doble Distancia (Mahalanobis-Euclidiana)
# --------------------------------------------------------------------------------
# La distancia de Mahalanobis mapea a la Densidad/Presión (P/rho) - Inercia Estadística.
# La distancia Euclidiana mapea a la Velocidad/Flujo (V/v) - Geometría Cinética.
# El equilibrio de estas distancias define el Empuje de Arquímides en el Espacio de Fase.

PHI = (1 + 5**0.5) / 2
PI  = np.pi

def golden_operator(n: int, phi: float = PHI) -> float:
    """
    Structured Golden Operator (Rule 2.1):
      O_n = cos(π·n) · cos(π·φ·n)

    Range: O_n ∈ [-1, 1]
    - Polarity π  →  Particle tendency / wave collapse
    - Laminar π/φ →  Wave tendency / coherent flow
    """
    parity = math.cos(PI * n)
    quasiperiod = math.cos(PI * phi * n)
    return parity * quasiperiod

def o_n_to_phase_fragment(o_n_value: float) -> int:
    """
    Maps O_n ∈ [-1, 1] → discrete_phase_fragment ∈ [0..7].
    
    Mapping used for topological packing.
    """
    fragment = round((o_n_value + 1.0) * 3.5)
    return int(np.clip(fragment, 0, 7))

def classify_particle(n: int, phi: float = PHI) -> str:
    """
    Classifies a state into 'bosonic' or 'fermionic' based on the
    competition between Parity and Quasiperiodicity.
    """
    parity = math.cos(PI * n)
    quasiperiod = math.cos(PI * phi * n)
    golden = parity * quasiperiod
    
    # Core value logic from Metriplex Oracle
    core_value = golden * quasiperiod
    
    if core_value < 0.1:
        return "fermionic"
    elif core_value > 0.1:
        return "bosonic"
    return "unknown"

def classify_cognitive_layer(ternary_weight: int, winding: int) -> str:
    """
    Classifies a state into one of the 3 cognitive levels:
    - Inconsciente: Sustrato/Crecimiento (Weight +1, Winding 0)
    - Subconciente: Reflexivo/Equilibrio (Weight 0)
    - Conciente: Decisivo/Intuitivo (Weight -1, Winding 2)
    """
    if ternary_weight == 0:
        return "subconciente"
    elif ternary_weight == 1 and winding == 0:
        return "inconsciente"
    elif ternary_weight == -1 and winding == 2:
        return "conciente"
    return "transitorio"

def phase_fragment_to_o_n(fragment: int) -> float:
    """Invierte el mapeo fragmento -> O_n."""
    return (fragment / 3.5) - 1.0
