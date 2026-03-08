import sys
import os
import pytest
import numpy as np

# Añadir el directorio raíz al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from nodes_network import Node, Network, HierarchicalNode, Neuron, PHI

def test_node_metriplectic_components():
    """Verifica que el nodo genere componentes simplecticas y métricas válidas."""
    node = Node(1.0, 0.5, -0.5, n=1)
    L_symp, L_metr = node.compute_lagrangian()
    
    assert isinstance(L_symp, float)
    assert isinstance(L_metr, float)
    assert L_metr <= 0  # La entropía disipativa debería ser <= 0 en este modelo

def test_golden_operator_range():
    """Verifica que el operador áureo esté acotado."""
    node = Node(0, 0, 0, n=42)
    val = node.golden_operator()
    assert -1.0 <= val <= 1.0

def test_hierarchical_nesting():
    """Prueba la estructura Box-in-Box."""
    parent = HierarchicalNode(n=0)
    child1 = Node(1.0, 0, 0, n=1)
    child2 = Node(0, 1.0, 0, n=2)
    
    parent.add_child(child1)
    parent.add_child(child2)
    
    ls_total, lm_total = parent.compute_lagrangian()
    
    # Debe ser la suma de hijos + dinámica de frontera
    ls1, lm1 = child1.compute_lagrangian()
    ls2, lm2 = child2.compute_lagrangian()
    
    assert ls_total != (ls1 + ls2) # Debería incluir la frontera
    assert lm_total < 0

def test_neuron_specialization():
    """Verifica el comportamiento de la clase Neuron."""
    neuron = Neuron(membrane_potential=1.2, n=5)
    assert neuron.fire() is True
    
    ls, lm = neuron.compute_lagrangian()
    assert ls != 0
    assert lm <= 0

def test_network_aggregation():
    """Verifica que la red use HierarchicalNode como raíz."""
    net = Network()
    net.add_node(Neuron(0.5, n=1))
    net.add_node(Neuron(1.5, n=2))
    
    ls_total, lm_total = net.get_total_lagrangian()
    assert ls_total != 0
    assert lm_total <= 0
