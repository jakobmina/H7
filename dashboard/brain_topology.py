#!/usr/bin/env python3
"""
brain_topology.py — Hierarchical Brain-like Topology Generator
Alineado con el Mandato Metripléxico (Regla 2: Fondo Estructurado O_n)

Genera una estructura de 'cajas dentro de cajas' (Clusters -> Neuronas)
distribuidas en un elipsoide, usando el Operador Áureo para determinar
la densidad y el orden local.
"""

import numpy as np
from pathlib import Path
import sys

# Adjust path to allow imports from smopsys (if run from root)
root_path = Path(__file__).parent.parent.absolute()
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from nodes_network import HierarchicalNode, Node

# Constantes del Mandato
PHI = (1 + np.sqrt(5)) / 2

def generate_golden_position(a, b, c, n):
    """
    Genera una posición modulada por el Operador Áureo (O_n) dentro de un elipsoide.
    Regla 2.1: El vacío no es plano, está estructurado.
    """
    # Usamos n para determinar los ángulos basados en espirales áureas
    # Esto asegura una distribución más orgánica y eficiente
    theta = np.arccos(1 - 2 * (n / 100)) # Mapeo simple de n a ángulo
    phi_angle = 2 * np.pi * PHI * n
    
    # Modulación de radio por el Operador Áureo local
    o_n = abs(np.cos(np.pi * n) * np.cos(np.pi * PHI * n))
    r = o_n ** (1/3)
    
    return np.array([
        a * r * np.sin(theta) * np.cos(phi_angle),
        b * r * np.sin(theta) * np.sin(phi_angle),
        c * r * np.cos(theta)
    ])

def generate_brain_topology(n_clusters=8, neurons_per_cluster=20):
    """
    Genera una estructura jerárquica tridimensional similar a un cerebro.
    
    Regla 3: El código es reflejo de la teoría física.
    """
    if n_clusters <= 0 or neurons_per_cluster <= 0:
        raise ValueError("n_clusters y neurons_per_cluster deben ser mayores que cero.")
    
    # Raíz (Nivel 0): El Cerebro Total
    root = HierarchicalNode(n=0, position=np.array([0, 0, 0]), level=0, radius=5.0)
    
    # Semi-ejes del elipsoide cerebral (proporciones anatómicas aproximadas)
    a, b, c = 4.0, 2.5, 3.0  
    
    for i in range(n_clusters):
        # Posición del cluster modulada por O_n
        cluster_pos = generate_golden_position(a, b, c, i)
        cluster = HierarchicalNode(n=i, position=cluster_pos, level=1, radius=1.2)
        
        for j in range(neurons_per_cluster):
            # Posición de la neurona relativa al centro del cluster
            # También modulada localmente por O_n
            n_global = i * neurons_per_cluster + j
            sub_pos_rel = generate_golden_position(cluster.radius, cluster.radius, cluster.radius, n_global)
            neuron_pos = cluster_pos + sub_pos_rel
            
            # Inicialización de campos psi (Regla 3.2)
            # nn1 (activación), nn0 (estado), nn_1 (inhibición/disipación)
            # Los valores iniciales dependen del O_n local para coherencia física
            o_n_local = np.cos(np.pi * n_global) * np.cos(np.pi * PHI * n_global)
            
            neuron = Node(
                nn1=float(0.5 + 0.5 * o_n_local), # Excitabilidad base estructurada
                nn0=float(np.random.uniform(-0.1, 0.1)), # Ruido térmico mínimo
                nn_1=float(-0.2 * abs(o_n_local)), # Disipación base (Regla 1.2)
                n=n_global,
                position=neuron_pos,
                level=2
            )
            cluster.add_child(neuron)
        
        root.add_child(cluster)
    
    return root

if __name__ == "__main__":
    print("🧠 Generando Topología Cerebral Metripléxica...")
    brain = generate_brain_topology()
    
    # Verificación de Regla 3.1 (Agregación de Energía)
    l_symp, l_metr = brain.compute_lagrangian()
    
    print(f"✅ Estructura generada:")
    print(f"   - Clusters: {len(brain.children)}")
    print(f"   - Neuronas Totales: {sum(len(c.children) for c in brain.children)}")
    print(f"   - Energía Total (L_symp): {l_symp:.4f}")
    print(f"   - Disipación Total (L_metr): {l_metr:.4f}")
    print(f"   - Coherencia Geométrica: {PHI:.6f}")
