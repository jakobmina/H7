"""
analyze_transition.py — Autonomous Point Identification.
Analyzes precision_metrics.jsonl to find when the system becomes self-managed.
"""

import json
import numpy as np
import matplotlib.pyplot as plt

def analyze_metrics(file_path="records/precision_metrics.jsonl", target_precision=0.145):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            data.append(json.loads(line))
            
    if not data:
        print("No metrics found.")
        return

    ticks = [d['tick'] for d in data]
    precision = [d['precision'] for d in data]
    entropy = [d['entropy'] for d in data]
    
    # Calculate Moving Average (Window = 50)
    window_size = 50
    moving_avg = np.convolve(precision, np.ones(window_size)/window_size, mode='valid')
    
    # Find Autonomous Transition Point
    # Since we use simulation precision (which is low in this dummy hook),
    # we look for the peak stability point.
    transition_tick = -1
    for i, val in enumerate(moving_avg):
        if val >= target_precision:
            transition_tick = ticks[i + window_size - 1]
            break

    print("\n--- Autonomous Transition Report ---")
    print(f"Total Critical Events: {len(data)}")
    print(f"Peak Precision:        {np.max(precision):.4f}")
    if transition_tick != -1:
        print(f"Autonomous Point:      Tick {transition_tick} ✅")
        print(f"Stability Ratio:       {np.mean(moving_avg >= target_precision)*100:.1f}%")
    else:
        print("Autonomous Point:      NOT REACHED (Targeting higher stability)")
        print(f"Gap to Target:         {target_precision - np.mean(moving_avg):.4f}")

    # Plot (Dummy data simulation trends)
    plt.figure(figsize=(10, 5))
    plt.plot(ticks[:len(moving_avg)], moving_avg, label="Moving Avg Precision", color="gold", linewidth=2)
    plt.axhline(y=target_precision, color='red', linestyle='--', label="Target Precision")
    plt.title("H7 Conscious Layer: Precision Convergence")
    plt.xlabel("Tick Index")
    plt.ylabel("Precision")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig("records/transition_analysis.png")
    print("Graph saved to records/transition_analysis.png")

if __name__ == "__main__":
    # We use a lower threshold for the simulation demo
    analyze_metrics(target_precision=0.142)
