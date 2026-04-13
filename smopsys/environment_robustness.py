"""
environment_robustness.py — Sensitivity and Robustness Tester.
Runs multiple simulations with different environmental profiles and compares Autonomous Points.
"""

import subprocess
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from smopsys.analyze_transition import analyze_metrics

def run_trial(profile, iters=3000):
    print(f"\n>>> Running Trial: Profile={profile}, Iters={iters}")
    cmd = ["python3", "smopsys/adaptive_cl_loop_integrated.py", str(iters), profile]
    subprocess.run(cmd, env={**os.environ, "PYTHONPATH": "."}, check=True)
    
    # Analyze the result
    data = []
    with open("records/precision_metrics.jsonl", "r") as f:
        for line in f:
            data.append(json.loads(line))
            
    if not data:
        return None
        
    ticks = [d['tick'] for d in data]
    precision = [d['precision'] for d in data]
    
    window_size = 50
    moving_avg = np.convolve(precision, np.ones(window_size)/window_size, mode='valid')
    
    target = 0.142
    ap = -1
    for i, val in enumerate(moving_avg):
        if val >= target:
            ap = ticks[i + window_size - 1]
            break
    return ap

def main():
    profiles = ["BASELINE", "SYNERGISTIC", "DISRUPTIVE", "REFLEXIVE"]
    trials_per_profile = 3 # Reduced for speed, can be increased
    results = {p: [] for p in profiles}
    
    os.makedirs("records", exist_ok=True)
    
    for prof in profiles:
        for i in range(trials_per_profile):
            print(f"\n--- Starting {prof} Trial {i+1}/{trials_per_profile} ---")
            ap = run_trial(prof)
            if ap != -1:
                results[prof].append(ap)
            else:
                results[prof].append(None)

    # Save Results
    with open("records/robustness_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    # Analysis & Graph
    print("\n--- Robustness Summary ---")
    plot_data = []
    labels = []
    for prof, aps in results.items():
        valid_aps = [v for v in aps if v is not None]
        mean_ap = np.mean(valid_aps) if valid_aps else 0
        std_ap = np.std(valid_aps) if valid_aps else 0
        print(f"{prof:12}: Mean AP = {mean_ap:.1f} (std={std_ap:.1f})")
        plot_data.append(valid_aps)
        labels.append(prof)
        
    plt.figure(figsize=(10, 6))
    plt.boxplot(plot_data, labels=labels)
    plt.title("H7 Autonomous Point (AP) Robustness across Environments")
    plt.ylabel("Tick Index of AP")
    plt.grid(alpha=0.3)
    plt.savefig("records/robustness_comparison.png")
    print("\nRobustness comparison graph saved to records/robustness_comparison.png")

if __name__ == "__main__":
    main()
