import numpy as np
import pytest
from notebooks.demo import golden_operator, MetriplecticSystem

def test_golden_operator():
    """Verifica que el Operador Áureo no sea cero ni plano (Regla 2.1)"""
    val = golden_operator(1)
    assert val != 0
    assert not np.isnan(val)

def test_metriplectic_system_rules():
    """Verifica las Reglas 1.1, 1.2 y 1.3"""
    sys = MetriplecticSystem(omega=1.618, gamma=0.1)
    
    # Probar con una amplitud psi
    psi = 1.0
    ls, lm = sys.compute_lagrangian(psi)
    
    # Regla 1.1: d_symp (H) debe ser conservativo (positivo en este modelo simple)
    assert ls > 0
    
    # Regla 1.2: d_metr (S) debe ser disipativo (negativo)
    assert lm < 0
    
    # Regla 1.3: No permitidos sistemas puramente conservativos o disipativos
    assert ls != 0 and lm != 0

def test_lagrangian_nomenclature():
    """Verifica que se use la nomenclatura estándar (Regla 3.2)"""
    sys = MetriplecticSystem()
    # El método debe aceptar psi como argumento conceptual (simulado aquí por la prueba)
    ls, lm = sys.compute_lagrangian(psi_amp=1.0)
    assert ls is not None
    assert lm is not None
