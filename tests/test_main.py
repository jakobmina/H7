import os
import sys
import pytest

# Añadir el directorio raíz al path para importar main.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from main import CorticalOrchestrator

def test_orchestrator_metriplectic_components():
    """Verifica que el orquestador respeta las reglas fundamentales del lagrangiano."""
    orch = CorticalOrchestrator()
    L_symp, L_metr = orch.compute_lagrangian()
    
    # Al inicio, sin procesos activos, L_symp es 0 y L_metr es disipativo por falta de procesos
    assert isinstance(L_symp, float)
    assert isinstance(L_metr, float)
    assert L_symp >= 0.0 # Componente conservativa siempre positiva o 0
    assert L_metr <= 0.0 # Componente métrica siempre entropía (<= 0)

def test_golden_operator_bounds():
    """Verifica que el operador áureo (O_n) se mantenga estructurado entre -1 y 1."""
    orch = CorticalOrchestrator()
    # Test a few arbitrary "n" times
    for n in [0, 1.618, 3.1415, 100]:
        o_n = orch.golden_operator(n)
        assert -1.0 <= o_n <= 1.0

def test_orchestrator_simulated_process_death():
    """Verifica cómo responde la física del orquestador ante la 'muerte' de procesos."""
    orch = CorticalOrchestrator()
    
    # Mockeamos como si hubiera procesos activos al principio (trampa para la prueba)
    class MockProcess:
        def poll(self): return None # Vivo
        
    orch.processes = [MockProcess(), MockProcess()] # Dos procesos vivos emulados
    ls_alive, lm_alive = orch.compute_lagrangian()
    
    assert ls_alive > 0.0
    assert lm_alive == 0.0 # Sin entropía si todo funciona
    
    # Emulamos muerte de uno
    class DeadProcess:
        def poll(self): return 1 # Muerto
        
    orch.processes = [MockProcess(), DeadProcess()]
    ls_half, lm_half = orch.compute_lagrangian()
    
    # Debe haber menos orden
    assert ls_half < ls_alive
    # Debe haber surgido entropía (disipación)
    assert lm_half < lm_alive 
    assert lm_half < 0.0
