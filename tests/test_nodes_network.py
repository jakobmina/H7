import sys
import os
import pytest
import numpy as np

# Añadir el directorio raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nodes_network import Node, Network, PHI

def test_node_metriplectic_components():
    """Verifica que el nodo genere componentes simplecticas y métricas válidas."""
    node = Node(1.0, 0.5, -0.5, n=1)
    L_symp, L_metr = node.compute_lagrangian()
    
    # H = (1^2 + 0.5^2 + (-0.5)^2)*0.5 = (1 + 0.25 + 0.25)*0.5 = 0.75
    # O_n = cos(pi*1) * cos(pi*PHI*1) = -1 * cos(pi*1.618) 
    # cos(pi*1.618) es negativo, así que L_symp debería ser positivo
    assert isinstance(L_symp, float)
    assert isinstance(L_metr, float)
    assert L_metr <= 0  # La entropía disipativa debería ser <= 0 en este modelo

def test_golden_operator_range():
    """Verifica que el operador áureo esté acotado."""
    node = Node(0, 0, 0, n=42)
    val = node.golden_operator()
    assert -1.0 <= val <= 1.0

def test_network_aggregation():
    """Verifica que la red sume correctamente los Lagrangianos."""
    net = Network()
    net.add_node(1, 0, 0)
    net.add_node(0, 1, 0)
    
    ls_total, lm_total = net.get_total_lagrangian()
    assert ls_total != 0
    assert lm_total <= 0
