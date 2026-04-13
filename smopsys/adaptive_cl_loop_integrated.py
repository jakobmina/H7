"""
adaptive_cl_loop_integrated.py — Master Bio-Digital Controller.
Integrates Fragmented Gemma-4 Pipeline, H7 Metriplectic Logic, and Quantum Critical Hook.
"""

import time
import numpy as np
import threading
import json
import os
from smopsys.gemma_pipeline_orchestrator import PipelineOrchestrator
from smopsys.quantum_critical_hook import QuantumCriticalHook
from smopsys.h7_quantum_oracle import MetriplexOracle, MetriplexConfig
from smopsys.handover_manager import HandoverManager
from enum import Enum

class EnvironmentProfile(Enum):
    BASELINE    = "baseline"
    SYNERGISTIC = "synergistic"
    DISRUPTIVE  = "disruptive"
    REFLEXIVE   = "reflexive"

class BioDigitalMaster:
    def __init__(self, observation_margin=500):
        self.orchestrator = PipelineOrchestrator()
        self.qc_hook = QuantumCriticalHook(precision_target=0.85)
        self.oracle = MetriplexOracle(MetriplexConfig())
        self.handover = HandoverManager(observation_margin=observation_margin)
        
        self.running = False
        self.critical_boundary = 0.8  # Entropy threshold for Quantum Hook
        
        # Performance Logging
        self.metrics_file = os.getenv("METRICS_FILE", "records/precision_metrics.jsonl")
        os.makedirs("records", exist_ok=True)
        # Clear previous metrics
        with open(self.metrics_file, "w") as f:
            pass

    def log_metric(self, entry: dict):
        with open(self.metrics_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def generate_stimulation(self, tick: int, profile: EnvironmentProfile) -> float:
        """Simulates external stimulation based on the selected profile."""
        if profile == EnvironmentProfile.BASELINE:
            return np.random.random()
        elif profile == EnvironmentProfile.SYNERGISTIC:
            # Coherent oscillation (sinusoidal guiding signal)
            return (np.sin(tick * 0.1) + 1.0) / 2.0
        elif profile == EnvironmentProfile.DISRUPTIVE:
            # Chaotic/High-entropy noise
            return np.random.normal(0.5, 0.4) % 1.0
        elif profile == EnvironmentProfile.REFLEXIVE:
            # Modulation based on previous H7 state (simplified)
            return (np.sin(tick * 0.05) * 0.5 + 0.5) + np.random.normal(0, 0.1)
        return np.random.random()

    def run_loop(self, iterations=5000, profile=EnvironmentProfile.BASELINE):
        self.orchestrator.start_pipeline()
        print("Waiting for pipeline stabilization...")
        time.sleep(3)
        
        print(f"Starting Integrated Loop (1000Hz target)...")
        self.running = True
        
        start_time = time.perf_counter()
        
        for i in range(iterations):
            tick_start = time.perf_counter()
            
            # 1. Simulate DishBrain Input with Profile
            spike_density = self.generate_stimulation(i, profile)
            
            # 2. Pipeline Inference (Asynchronous throughput)
            self.orchestrator.injector.send_json({
                "ts": tick_start,
                "density": spike_density
            })
            
            # 3. Handle Pipeline Output (if available)
            if self.orchestrator.collector.poll(timeout=0):
                result = self.orchestrator.collector.recv_json()
                
                # Assume Gemma-4 output includes simulated entropy for this demo
                entropy = np.random.random() 
                
                # 4. Conscious Layer: Trigger Quantum Critical Hook
                if entropy > self.critical_boundary:
                    # Map to H7 state index
                    h7_state = int(spike_density * 7)
                    qc_res = self.qc_hook.process_critical_event(h7_state, entropy)
                    
                    # 5. Handover Monitoring
                    alpha = self.handover.update(i, qc_res['precision'])
                    
                    # Simulate "Hardware" correction (different noise profile)
                    hw_correction = qc_res['correction'] + np.random.choice([-1, 0, 1]) * (1.0 - qc_res['precision'])
                    
                    # 6. Blended Output (Soft Handover)
                    final_correction = self.handover.blend_output(qc_res['correction'], hw_correction)
                    
                    # Log the critical event for long-run analysis
                    self.log_metric({
                        "tick": i,
                        "h7_state": h7_state,
                        "entropy": entropy,
                        "precision": qc_res['precision'],
                        "sim_correction": qc_res['correction'],
                        "hw_correction": hw_correction,
                        "final_correction": final_correction,
                        "alpha": alpha,
                        "status": self.handover.get_status()
                    })
                    
                    if i % 100 == 0:
                        print(f"  [CONSCIOUS] Tick {i}: Alpha={alpha:.2f}, Status={self.handover.get_status()}")
                
            # 5. Precise Rate Limiting (1ms pulse)
            while (time.perf_counter() - tick_start) < 0.001:
                pass
                
        print(f"Loop completed in {time.perf_counter() - start_time:.2f}s")
        self.orchestrator.cleanup()

if __name__ == "__main__":
    import sys
    iters = 5000
    profile = EnvironmentProfile.BASELINE
    
    if len(sys.argv) > 1:
        iters = int(sys.argv[1])
    if len(sys.argv) > 2:
        profile_str = sys.argv[2].upper()
        if profile_str in EnvironmentProfile.__members__:
            profile = EnvironmentProfile[profile_str]
        
    master = BioDigitalMaster()
    master.run_loop(iterations=iters, profile=profile)
