#!/usr/bin/env python3
"""
h7_phi_experiment.py — Experiment with Phi-Modulated Stimulus (H7)
Alineado con el Mandato Metripléxico (Regla 1, 2 y 3)

Implementa la emulación de dinámica de Sodio/Potasio mediante 
componentes Simplécticas (H) y Métricas (S).
"""

import numpy as np
import time
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys

# Adjust path to allow imports from dashboard
root_path = Path(__file__).parent.parent.absolute()
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

import cl
from dashboard.cl1_db import CL1Database

# ─── 1. MANDATO METRIPLÉXICO: FISICA CORE ───
PHI = (1 + np.sqrt(5)) / 2

class H7PhiController:
    """
    Controlador H7 para emulación iónica.
    
    Sodium (Na+): Componente de activación rápida (H - Conservativa/Interferencia)
    Potassium (K+): Componente de desactivación lenta (S - Disipativa/Métrica)
    """
    def __init__(self, f0=1.0, base_amp=1.5, k_factor=0.1):
        self.f0 = f0
        self.phi = PHI
        self.base_amp = base_amp
        self.k_factor = k_factor # Factor de disipación del Potasio (Regla 1.2)
        self.O_n_integrity = np.cos(np.pi * PHI) # Regla 2.1
        
    def compute_lagrangian(self, t):
        """Regla 3.1: Lagrangiano Explícito"""
        # Componente Simpléctica (H): Interferencia de frecuencias
        # Na+ emulation: Rapid activation phase
        L_symp = np.sin(2 * np.pi * self.f0 * t) + \
                 np.sin(2 * np.pi * self.f0 * self.phi * t) / self.phi
                 
        # Componente Métrica (S): Relajación / Atractor
        # K+ emulation: Recovery period (dissipation of excitation)
        L_metr = -abs(L_symp) * self.k_factor # Disipación proporcional a la amplitud
        
        return L_symp, L_metr

    def get_stim_design(self, t):
        """Genera el diseño de estímulo para el CL SDK."""
        L_symp, L_metr = self.compute_lagrangian(t)
        
        # Modulación Final (Regla 2: Operador Áureo)
        # Usamos O_n para escalar la amplitud activa
        mod_amp = self.base_amp * abs(self.O_n_integrity) * (L_symp + L_metr)
        
        # Limitamos para seguridad del hardware
        amp = np.clip(mod_amp, 0.1, 2.5)
        dur = 180 # µs (estándar)
        
        return cl.StimDesign(
            int(dur), -float(amp),
            int(dur),  float(amp)
        )

# ─── 2. EJECUCIÓN DEL EXPERIMENTO ───
def run_phi_experiment(duration_s=30, channels=[27], k_factor=0.1, f0=1.0, base_amp=2.0):
    print(f"🚀 Iniciando Experimento H7: Estímulo Modulado por \u03c6")
    print(f"   Modo: Emulación Sodio/Potasio (Metripléxico)")
    print(f"   Parámetros: f0={f0}Hz | Amp={base_amp}uA | K-factor(S)={k_factor}")
    print(f"   Integridad O_n: {np.cos(np.pi * PHI):.6f}\n")
    
    h7 = H7PhiController(f0=f0, base_amp=base_amp, k_factor=k_factor)
    db = CL1Database()
    db.new_session(ticks_per_second=1000, run_for_seconds=duration_s)
    
    start_time = time.time()
    ticks_target = int(duration_s * 1000)
    
    try:
        with cl.open() as neurons:
            for tick in neurons.loop(1000, stop_after_ticks=ticks_target):
                t_rel = tick.iteration / 1000.0 # tiempo relativo en s
                
                # Obtener componentes del Lagrangiano para registro
                ls, lm = h7.compute_lagrangian(t_rel)
                
                # Aplicar estímulo solo si la fase de interferencia es constructiva (activación)
                # Simula el umbral de disparo para Na+
                stim_fired = False
                if ls > 0.5: 
                    stim_design = h7.get_stim_design(t_rel)
                    neurons.stim(cl.ChannelSet(channels), stim_design)
                    stim_fired = True
                
                # Registro en DB (Metriplectic Metadata)
                db.write_tick(
                    ts_ns = time.time_ns(),
                    loop_dur_us = 1000.0, # Target 1ms
                    spike_count = len(tick.analysis.spikes),
                    stim_fired = stim_fired,
                    prob_bottleneck = h7.O_n_integrity, # Usado como marcador de integridad
                    l_symp = ls,
                    l_metr = lm,
                    stim_dur_us = 180,
                    stim_amp_ua = float(h7.base_amp * abs(h7.O_n_integrity) * (ls + lm)) if stim_fired else 0.0
                )
                
                if tick.iteration % 1000 == 0:
                    print(f"   [Tick {tick.iteration}] t={t_rel:.1f}s | Spikes={len(tick.analysis.spikes)} | Fire={stim_fired}")

    except Exception as e:
        print(f"Critical error: {e}")
    finally:
        db.finalize_session(skipped_stims=0, total_spikes=0)
        db.close()
        print("\n✅ Experimento finalizado. Datos guardados en cl1_session.sqlite")
        plot_diagnostic(h7)

def plot_diagnostic(h7, duration=2.0):
    """Regla 3.3: Visualización Diagnóstica de componentes H y S."""
    t = np.linspace(0, duration, 1000)
    l_symps = []
    l_metrs = []
    
    for val in t:
        ls, lm = h7.compute_lagrangian(val)
        l_symps.append(ls)
        l_metrs.append(lm)
        
    plt.figure(figsize=(12, 6))
    plt.style.use('dark_background')
    plt.plot(t, l_symps, label='d_symp (Na+ Activation / H)', color='#00f2ff', linewidth=2)
    plt.plot(t, l_metrs, label='d_metr (K+ Relaxation / S)', color='#f39c12', linewidth=2)
    
    plt.title("Regla 3.3: Competencia Metripléxica (Emulación Na+/K+)", fontweight='bold')
    plt.xlabel("Tiempo (s)")
    plt.ylabel("Amplitud / Energía")
    plt.axhline(0, color='white', linestyle='--', alpha=0.3)
    plt.legend()
    plt.grid(alpha=0.2)
    
    img_path = "h7_ionic_diagnostic.png"
    plt.savefig(img_path, dpi=300)
    print(f"📊 Diagnóstico guardado: {img_path}")
    # plt.show() # Opcional, pero guardamos imagen para el walkthrough

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="H7 Phi-Modulated Experiment")
    parser.add_argument("--duration", type=int, default=30, help="Duration in seconds")
    parser.add_argument("--k_factor", type=float, default=0.1, help="Potassium relaxation factor (S)")
    parser.add_argument("--f0", type=float, default=1.0, help="Base frequency (H)")
    parser.add_argument("--amp", type=float, default=2.0, help="Base amplitude (uA)")
    
    args = parser.parse_args()
    
    run_phi_experiment(
        duration_s=args.duration, 
        k_factor=args.k_factor,
        f0=args.f0,
        base_amp=args.amp
    )
