import subprocess
import sys
import time
import argparse
import signal
import numpy as np

# ─── MANDATO METRIPLÉXICO ───
PHI = (1 + np.sqrt(5)) / 2

class CorticalOrchestrator:
    """
    Orchestrator for the Cortical System.
    Validates physical principles (Metriplectic Mandate) while managing subprocesses.
    """
    def __init__(self, experiment_script="smopsys/h7_phi_experiment.py", dashboard_script="dashboard/unified_dashboard.py"):
        self.experiment_script = experiment_script
        self.dashboard_script = dashboard_script
        self.processes = []
        self.start_time = time.time()
        self.errors = 0
        self.stop_requested = False
        
        # Setup graceful termination
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)

    def signal_handler(self, sig, frame):
        """Sets the stop flag without calling print to avoid reentrancy."""
        self.stop_requested = True

    def golden_operator(self, n):
        """Regla 2.1: Operador Áureo."""
        return np.cos(np.pi * n) * np.cos(np.pi * PHI * n)

    def compute_lagrangian(self):
        """
        Regla 3.1: Lagrangiano Explícito
        L_symp (Conservacional): Estabilidad del sistema (procesos vivos).
        L_metr (Disipativo): Entropía generada por errores o caídas.
        """
        t = time.time() - self.start_time
        
        # H (Symplectic) -> System persistence / order
        active_procs = sum(1 for p in self.processes if p.poll() is None)
        expected_procs = 2 # Experiment + Dashboard
        
        L_symp = (active_procs / expected_procs) * (1.0 + 0.1 * np.sin(t)) if expected_procs > 0 else 0.0
        
        # S (Metric) -> Dissipation / Entropy (Errors)
        L_metr = - (self.errors * 0.5) - (expected_procs - active_procs) * 1.0
        
        # Cap metric term to avoid complete collapse if not meant to
        if L_metr > 0: L_metr = 0.0
            
        return L_symp, L_metr

    def launch(self):
        """Launches the necessary subsystems."""
        print(f"🚀 Iniciando Orchestrator Cortical")
        
        # 1. Launch backend experiment
        try:
            exp_proc = subprocess.Popen([sys.executable, self.experiment_script])
            self.processes.append(exp_proc)
            print(f"✅ Backend iniciado: {self.experiment_script} (PID {exp_proc.pid})")
        except Exception as e:
            print(f"❌ Error iniciando backend: {e}")
            self.errors += 1

        # 2. Launch Streamlit dashboard
        try:
            # streamlit command might not be exactly sys.executable + script
            dash_proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", self.dashboard_script])
            self.processes.append(dash_proc)
            print(f"✅ Dashboard iniciado: {self.dashboard_script} (PID {dash_proc.pid})")
        except Exception as e:
            print(f"❌ Error iniciando dashboard: {e}")
            self.errors += 1

    def monitor(self):
        """Monitors system stability using Metriplectic principles."""
        try:
            while not self.stop_requested:
                # Sleep in small increments to remain responsive to signals
                for _ in range(10):
                    if self.stop_requested:
                        break
                    time.sleep(0.2)
                    
                if self.stop_requested:
                    break
                
                # Check metrics
                l_symp, l_metr = self.compute_lagrangian()
                o_n = self.golden_operator(time.time() - self.start_time)
                
                print(f"📊 Estado Orchestrator | O_n={o_n:.3f} | L_symp(Order)={l_symp:.2f} | L_metr(Entropy)={l_metr:.2f}")
                
                # Check for dead processes
                active = [p for p in self.processes if p.poll() is None]
                if len(active) < len(self.processes):
                    print("⚠️ Alerta: Un subproceso terminó inesperadamente.")
                    self.errors += 1
                    # Break if all dead
                    if len(active) == 0:
                        print("🛑 Todos los subsistemas terminaron. Saliendo...")
                        break
                        
        except KeyboardInterrupt:
            pass
            
        self.shutdown()

    def shutdown(self):
        """Gracefully terminates all subprocesses."""
        print("\n🛑 Señal de interrupción o fin de procesos. Terminando subsistemas (Regla de Reversibilidad - Retorno al Vacío)...")
        for p in self.processes:
            if p.poll() is None:
                p.terminate()
        
        # Wait a bit
        time.sleep(1)
        for p in self.processes:
            if p.poll() is None:
                p.kill() # Force kill if still lingering
                
        print("✅ Orchestrator cerrado limpiamente.")
        sys.exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cortical System Orchestrator")
    parser.add_argument("--experiment", type=str, default="h7", choices=["h7", "adaptive"], help="Backend experiment to run")
    
    args = parser.parse_args()
    
    script = "smopsys/adaptive_cl_loop.py" if args.experiment == "adaptive" else "smopsys/h7_phi_experiment.py"
    
    orch = CorticalOrchestrator(experiment_script=script)
    orch.launch()
    orch.monitor()
