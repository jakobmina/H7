"""
parallel_evolution.py — Multiple Parallel Starts Orchestrator.
Monitors the evolution of parallel H7 instances with different environmental seeds.
"""

import multiprocessing
import os
import time
import subprocess
import json

def run_instance(instance_id, profile="REFLEXIVE", iters=5000):
    print(f"[INSTANCE {instance_id}] Starting Evolution (Profile={profile})...")
    # Each instance gets its own record directory? No, let's use suffixes.
    metrics_file = f"records/instance_{instance_id}_metrics.jsonl"
    
    # We'll pass the metrics file via environment variable or modification
    # For now, let's just use the subprocess command
    cmd = ["python3", "smopsys/adaptive_cl_loop_integrated.py", str(iters), profile]
    # We need a way to tell the script to use a different metrics file.
    # I'll update adaptive_cl_loop_integrated.py to accept a metrics file path.
    env = {
        **os.environ, 
        "PYTHONPATH": ".",
        "METRICS_FILE": metrics_file
    }
    subprocess.run(cmd, env=env, check=True)
    print(f"[INSTANCE {instance_id}] Evolution Complete.")

def main():
    os.makedirs("records", exist_ok=True)
    
    # Profiles to test in parallel
    evolutions = [
        (0, "BASELINE"),
        (1, "SYNERGISTIC"),
        (2, "DISRUPTIVE"),
        (3, "REFLEXIVE")
    ]
    
    processes = []
    print("--- Launching Parallel Evolutions ---")
    for inst_id, prof in evolutions:
        p = multiprocessing.Process(target=run_instance, args=(inst_id, prof, 5000))
        p.start()
        processes.append(p)
        
    for p in processes:
        p.join()
        
    print("\n--- All Parallel Evolutions Finished ---")
    print("Metrics saved to records/instance_*_metrics.jsonl")

if __name__ == "__main__":
    main()
