import numpy as np
import pytest
from h7_phi_experiment import H7PhiController

def test_h7_phi_controller_lagrangian():
    """Verifica que el Lagrangiano cumpla la Regla 1.1 y 1.2"""
    h7 = H7PhiController(f0=1.0)
    
    # Evaluar en t=0.25 (pico esperado de seno)
    t = 0.25
    ls, lm = h7.compute_lagrangian(t)
    
    # Regla 1.1 (Simpléctica/Conservativa)
    assert ls != 0
    # Regla 1.2 (Métrica/Disipativa)
    assert lm <= 0 # Debe ser disipativo o neutro

def test_phi_modulation_frequency():
    """Verifica que el componente áureo esté presente en ls"""
    h7 = H7PhiController(f0=1.0)
    ls1, _ = h7.compute_lagrangian(0.1)
    
    # Si quitamos el componente phi, el valor cambiaría
    phi = (1 + np.sqrt(5)) / 2
    pure_sine = np.sin(2 * np.pi * 1.0 * 0.1)
    
    assert ls1 != pure_sine # Indica interferencia de la frecuencia áurea

def test_golden_integrity_operator():
    """Verifica la integridad del operador O_n (Regla 2.1)"""
    h7 = H7PhiController()
    assert h7.O_n_integrity == pytest.approx(np.cos(np.pi * (1+np.sqrt(5))/2))
    assert abs(h7.O_n_integrity) > 0 # El fondo nunca es cero (Regla 2.1)
