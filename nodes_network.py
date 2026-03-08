import numpy as np

# Constantes Fundamentales
PHI = (1 + np.sqrt(5)) / 2

class Node:
    """
    Representa un nodo neuronal base bajo el Mandato Metripléctico.
    nn1, nn0, nn_1 actúan como campos de estado (psi).
    """
    def __init__(self, nn1, nn0, nn_1, n=0, position=None, level=0):
        self.psi = np.array([nn1, nn0, nn_1], dtype=float)
        self.n = n # Índice temporal/espacial para O_n
        self.position = position if position is not None else np.zeros(3)
        self.level = level
        
    def golden_operator(self):
        """Regla 2.1: Fondo Estructurado (O_n)"""
        return np.cos(np.pi * self.n) * np.cos(np.pi * PHI * self.n)

    def compute_lagrangian(self):
        """Regla 3.1: Dinámica Metripléctica (H y S)"""
        # H: Energía (Simpléctica). Simulado como rotación de fase coherente.
        H = np.sum(self.psi**2) * 0.5 
        L_symp = H * self.golden_operator()
        
        # S: Entropía (Métrica). Simulado como relajación hacia el equilibrio.
        target = np.mean(self.psi)
        S = np.sum((self.psi - target)**2) * 0.1
        L_metr = -S
        
        return L_symp, L_metr

    def __str__(self):
        L_s, L_m = self.compute_lagrangian()
        pos_str = f"({self.position[0]:.2f}, {self.position[1]:.2f}, {self.position[2]:.2f})"
        return f"{self.__class__.__name__}(n={self.n}, lvl={self.level}) @ {pos_str} | L_symp: {L_s:.4f}, L_metr: {L_m:.4f}"

class HierarchicalNode(Node):
    """
    Nodo contenedor (Caja dentro de Caja). 
    Agrega dinámica de sus hijos y su propia frontera.
    """
    def __init__(self, n=0, position=None, level=0, radius=1.0, children=None):
        super().__init__(0, 0, 0, n=n, position=position, level=level)
        self.radius = radius
        self.children = children if children is not None else []

    def add_child(self, child):
        self.children.append(child)

    def compute_lagrangian(self):
        """Agregación recursiva de Lagrangianos"""
        total_symp = 0.0
        total_metr = 0.0
        
        # 1. Contribución de los hijos
        for child in self.children:
            ls, lm = child.compute_lagrangian()
            total_symp += ls
            total_metr += lm
            
        # 2. Dinámica de Frontera (Membrana del contenedor)
        # La 'frontera' se ve influenciada por el promedio de los hijos
        if self.children:
            representative_psi = np.mean([np.mean(c.psi) if hasattr(c, 'psi') else 0 for c in self.children])
            self.psi = np.array([representative_psi, representative_psi * 0.8, representative_psi * 0.6])
        
        # Aplicar Regla 3.1 a la propia frontera
        L_s_boundary, L_m_boundary = super().compute_lagrangian()
        
        return total_symp + L_s_boundary, total_metr + L_m_boundary

class Neuron(Node):
    """
    Especialización de Node para representar una neurona.
    """
    def __init__(self, membrane_potential, n=0):
        # Mapeamos el potencial de membrana a los tres estados psi
        super().__init__(membrane_potential, membrane_potential * 0.5, 0.0, n=n)
        self.potential = membrane_potential

    def fire(self):
        """Simulación de disparo (spike)"""
        if self.potential > 1.0:
            return True
        return False

class Network:
    def __init__(self):
        self.root = HierarchicalNode(n=0)
        
    def add_node(self, node):
        self.root.add_child(node)
        
    def get_total_lagrangian(self):
        return self.root.compute_lagrangian()

    def __str__(self):
        ts, tm = self.get_total_lagrangian()
        return f"Network(Nodes={len(self.root.children)}) | ΣL_symp: {ts:.4f}, ΣL_metr: {tm:.4f}"