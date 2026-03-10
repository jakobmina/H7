import pytest
import numpy as np
from smopsys.brain_topology import generate_brain_topology, generate_golden_position

def test_brain_topology_structure():
    n_clusters = 5
    neurons_per_cluster = 10
    brain = generate_brain_topology(n_clusters=n_clusters, neurons_per_cluster=neurons_per_cluster)
    
    assert len(brain.children) == n_clusters
    assert brain.level == 0
    
    total_neurons = 0
    for cluster in brain.children:
        assert cluster.level == 1
        assert len(cluster.children) == neurons_per_cluster
        total_neurons += len(cluster.children)
        
        for neuron in cluster.children:
            assert neuron.level == 2
            # Verificar campos psi (Regla 3.2)
            assert len(neuron.psi) == 3
            
    assert total_neurons == n_clusters * neurons_per_cluster

def test_metriplectic_dynamics():
    brain = generate_brain_topology(n_clusters=2, neurons_per_cluster=5)
    l_symp, l_metr = brain.compute_lagrangian()
    
    # Regla 1.3: No debe ser puramente conservativo ni puramente disipativo
    # (Aunque l_symp podría ser 0 en un cruce, l_metr suele ser negativo)
    assert isinstance(l_symp, float)
    assert isinstance(l_metr, float)
    assert l_metr <= 0 # La disipación (S) siempre resta o es cero

def test_golden_distribution():
    pos1 = generate_golden_position(1, 1, 1, 1)
    pos2 = generate_golden_position(1, 1, 1, 2)
    
    # No deben estar en el mismo sitio
    assert not np.array_equal(pos1, pos2)
    # Deben estar dentro del radio (elipsoide unitario aquí)
    assert np.linalg.norm(pos1) <= np.sqrt(3) # Máximo teórico para r=1 en elipsoide

def test_invalid_parameters():
    with pytest.raises(ValueError):
        generate_brain_topology(n_clusters=0)
    with pytest.raises(ValueError):
        generate_brain_topology(neurons_per_cluster=-1)
