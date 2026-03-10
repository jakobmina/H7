import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import clear_output, display, HTML

# --- DIT System Styling ---
plt.style.use('dark_background') # Estética de terminal de investigación
DIT_GREEN = '#00ff41' # Verde Matrix/Laboratorio
DIT_RED = '#ff3e3e'
DIT_CYAN = '#00f2ff'

# --- DIT Formal Constants ---
PHI = (1 + np.sqrt(5)) / 2
DRIFT_072 = 7.0 - (2 * np.pi)
THRESHOLD = 2.0
TOTAL_CYCLES = 400


class QuoreMindMonitor:
    """
    Monitor de estabilidad de fase para el sistema DIT (Dinámica de Información Turbulenta).
    Implementa el Operador Áureo O_n y el mecanismo de corrección H7.
    
    Componente Simpléctica (d_symp): innovación = cos(π i) * cos(π φ i)
    Componente Métrica (d_metr): corrección por DRIFT_072 cuando |acumulador| > THRESHOLD
    """

    def __init__(self):
        self.accumulator = 0.0
        self.history = []
        self.outliers = []
        self.stability_index = 100.0
        self.corrections = 0

    def compute_lagrangian(self, i):
        """L_symp: término conservativo (Operador Áureo), L_metr: término disipativo."""
        L_symp = np.cos(np.pi * i) * np.cos(np.pi * PHI * i)  # Componente simpléctica
        L_metr = -abs(self.accumulator) * (DRIFT_072 / THRESHOLD)  # Componente métrica
        return L_symp, L_metr

    def simulate_cycle(self, i):
        """Ejecuta un ciclo completo con complejidad metripléctica."""
        L_symp, _ = self.compute_lagrangian(i)
        innovation = L_symp
        self.accumulator += innovation + np.random.normal(0, 0.05)
        is_outlier = abs(self.accumulator) > THRESHOLD

        if is_outlier or (i % 7 == 0):
            self.accumulator *= (DRIFT_072 / THRESHOLD)
            self.corrections += 1
            if is_outlier:
                self.outliers.append((i, self.accumulator))

        self.history.append(self.accumulator)
        self.stability_index = 100 * (1 - (len(self.outliers) / (i + 1)))

    def plot_dashboard(self, i):
        """Visualización diagnóstica: flujo laminar vs. disipación."""
        clear_output(wait=True)
        fig = plt.figure(figsize=(16, 6))
        gs = fig.add_gridspec(1, 3, width_ratios=[2, 1, 0.1])

        # 1. Phase Accumulator Plot (Laminar Flow — d_symp)
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(self.history, color=DIT_CYAN, linewidth=1.5, alpha=0.9, label='Laminar Flow (d_symp)')
        ax1.fill_between(range(len(self.history)), self.history, color=DIT_CYAN, alpha=0.1)
        ax1.axhline(y=THRESHOLD, color=DIT_RED, linestyle='--', alpha=0.5, label='Phase Limit')
        ax1.axhline(y=-THRESHOLD, color=DIT_RED, linestyle='--', alpha=0.5)
        ax1.set_title(
            f"DIT PHASE STABILITY | CYCLE {i}/{TOTAL_CYCLES}",
            loc='left', fontsize=12, fontweight='bold', color=DIT_CYAN
        )
        ax1.set_ylim(-2.8, 2.8)
        ax1.grid(color='gray', linestyle=':', alpha=0.3)
        ax1.legend(loc='upper right', fontsize=8)

        # 2. Stability Meter (Bar — d_metr competition)
        ax2 = fig.add_subplot(gs[1])
        status_color = DIT_GREEN if self.stability_index > 99 else '#f1c40f'
        ax2.bar(["STABILITY"], [self.stability_index], color=status_color, width=0.5)
        ax2.set_ylim(95, 100.2)
        ax2.set_title("SYSTEM INTEGRITY", fontsize=12, fontweight='bold')
        ax2.text(
            0, self.stability_index - 0.5,
            f"{self.stability_index:.2f}%",
            ha='center', va='top', fontsize=20, color='black', fontweight='bold'
        )

        plt.tight_layout()
        display(fig)
        plt.close()


if __name__ == "__main__":
    time.sleep(1.5)

    for i in range(1, TOTAL_CYCLES + 1):
        monitor.simulate_cycle(i)
        if i % 10 == 0:
            monitor.plot_dashboard(i)
        time.sleep(0.01)

    print(f"\n✅ SIMULATION COMPLETE")
    print(f"   Stability Index : {monitor.stability_index:.4f}%")
    print(f"   Total Outliers  : {len(monitor.outliers)}")
    print(f"   Corrections     : {monitor.corrections}")
