import numpy as np
from nodes_network import HierarchicalNode, Node

def generate_brain_topology(n_clusters=8, neurons_per_cluster=20):
    """
    Generates a hierarchical 3D brain-like structure.
    Clusters are distributed in a prolate spheroid (ellipsoid).
    """
    root = HierarchicalNode(n=0, position=np.array([0, 0, 0]), level=0, radius=5.0)
    
    # Brain dimensions (Ellipsoid)
    a, b, c = 4.0, 2.5, 3.0  # x, y, z semi-axes
    
    for i in range(n_clusters):
        # random point in ellipsoid
        phi = np.random.uniform(0, 2 * np.pi)
        theta = np.arccos(np.random.uniform(-1, 1))
        r = np.random.uniform(0, 1) ** (1/3)
        
        pos = np.array([
            a * r * np.sin(theta) * np.cos(phi),
            b * r * np.sin(theta) * np.sin(phi),
            c * r * np.cos(theta)
        ])
        
        cluster = HierarchicalNode(
            n=i, 
            position=pos, 
            level=1, 
            radius=1.2
        )
        
        # Add sub-nodes (neurons) within cluster
        for j in range(neurons_per_cluster):
            # relative random pos within cluster radius
            r_sub = np.random.uniform(0, cluster.radius)
            phi_sub = np.random.uniform(0, 2 * np.pi)
            theta_sub = np.random.uniform(0, np.pi)
            
            sub_pos = pos + np.array([
                r_sub * np.sin(theta_sub) * np.cos(phi_sub),
                r_sub * np.sin(theta_sub) * np.sin(phi_sub),
                r_sub * np.cos(theta_sub)
            ])
            
            # Simple Node (Level 2)
            neuron = Node(
                nn1=np.random.uniform(-1, 1),
                nn0=np.random.uniform(-1, 1),
                nn_1=np.random.uniform(-1, 0),
                n=j,
                position=sub_pos,
                level=2
            )
            cluster.add_child(neuron)
            
        root.add_child(cluster)
        
    return root

if __name__ == "__main__":
    brain = generate_brain_topology()
    print(f"Generated brain with {len(brain.children)} clusters.")
    for cluster in brain.children:
        print(f"  - Cluster at {cluster.position} with {len(cluster.children)} neurons.")
