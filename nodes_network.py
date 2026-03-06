import numpy as np

# Constantes Fundamentales
PHI = (1 + np.sqrt(5)) / 2

class Node:
    """
    Representa un nodo neuronal bajo el Mandato Metripléctico.
    nn1, nn0, nn_1 actúan como campos de estado.
    """
    def __init__(self, nn1, nn0, nn_1, n=0):
        self.psi = np.array([nn1, nn0, nn_1], dtype=float)
        self.n = n # Índice temporal/espacial para O_n
        
    def golden_operator(self):
        """Regla 2.1: Fondo Estructurado (O_n)"""
        return np.cos(np.pi * self.n) * np.cos(np.pi * PHI * self.n)

    def compute_lagrangian(self):
        """Regla 3.1: Dinámica Metripléctica"""
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
        return f"Node(n={self.n}) | L_symp: {L_s:.4f}, L_metr: {L_m:.4f}"

class Network:
    def __init__(self):
        self.nodes = []
        
    def add_node(self, nn1, nn0, nn_1):
        idx = len(self.nodes)
        node = Node(nn1, nn0, nn_1, n=idx)
        self.nodes.append(node)
        
    def get_total_lagrangian(self):
        total_symp = 0
        total_metr = 0
        for node in self.nodes:
            ls, lm = node.compute_lagrangian()
            total_symp += ls
            total_metr += lm
        return total_symp, total_metr

    def __str__(self):
        ts, tm = self.get_total_lagrangian()
        return f"Network(Size={len(self.nodes)}) | ΣL_symp: {ts:.4f}, ΣL_metr: {tm:.4f}"