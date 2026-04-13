"""
handover_manager.py — Graceful transition between Simulation and Hardware.
"""

import numpy as np

class HandoverManager:
    def __init__(self, observation_margin=500, stability_threshold=0.142):
        self.observation_margin = observation_margin
        self.stability_threshold = stability_threshold
        
        self.autonomous_tick = -1
        self.alpha = 0.0  # 0.0 = 100% Sim, 1.0 = 100% HW
        
        self.precision_buffer = []
        self.window_size = 50

    def update(self, current_tick: int, current_precision: float) -> float:
        """Updates the confidence score and returns the current blending alpha."""
        self.precision_buffer.append(current_precision)
        if len(self.precision_buffer) > self.window_size:
            self.precision_buffer.pop(0)

        # 1. Detect Autonomous Point (AP) if not already found
        if self.autonomous_tick == -1 and len(self.precision_buffer) == self.window_size:
            avg_prec = np.mean(self.precision_buffer)
            if avg_prec >= self.stability_threshold:
                self.autonomous_tick = current_tick
                print(f"[HANDOVER] Stability Detected at Tick {current_tick}. Entering Shadow Phase.")

        # 2. Calculate Alpha (Soft Blending)
        if self.autonomous_tick != -1:
            ticks_since_ap = current_tick - self.autonomous_tick
            # Linear ramp-up of hardware influence
            self.alpha = min(1.0, ticks_since_ap / self.observation_margin)
            
        return self.alpha

    def blend_output(self, sim_val: float, hw_val: float) -> float:
        """Returns the weighted output based on the current alpha."""
        return (1.0 - self.alpha) * sim_val + self.alpha * hw_val

    def get_status(self):
        if self.alpha == 0:
            return "SIMULATION_ONLY"
        elif self.alpha < 1.0:
            return f"HYBRID_BLENDING (Alpha={self.alpha:.2f})"
        else:
            return "AUTONOMOUS_HARDWARE"
