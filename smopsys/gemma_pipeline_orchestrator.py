"""
gemma_pipeline_orchestrator.py — Pipeline Manager (Robust Version).
Spawns fragment nodes and manages the data flow.
"""

import subprocess
import zmq
import time
import numpy as np
import os
import signal
import sys
import threading

# Configuration
NUM_FRAGMENTS = 6  # Let's try 6 as a middle ground
BASE_PORT     = 5550
HEARTBEAT_HZ  = 1000  # 1kHz target
TICK_INTERVAL = 1.0 / HEARTBEAT_HZ

class PipelineOrchestrator:
    def __init__(self):
        self.context = zmq.Context()
        self.injector = self.context.socket(zmq.PUSH)
        # We increase LINGER to 0 to prevent hanging on close
        self.injector.setsockopt(zmq.LINGER, 0)
        self.injector.bind(f"tcp://*:{BASE_PORT}")
        
        self.collector = self.context.socket(zmq.PULL)
        self.collector.setsockopt(zmq.LINGER, 0)
        self.collector.connect(f"tcp://localhost:{BASE_PORT + NUM_FRAGMENTS}")
        
        self.processes = []

    def start_pipeline(self):
        print(f"Starting Gemma-4 Pipeline with {NUM_FRAGMENTS} fragments...")
        for i in range(NUM_FRAGMENTS):
            in_port  = BASE_PORT + i
            out_port = BASE_PORT + i + 1
            cmd = [
                sys.executable, 
                "smopsys/gemma_fragment_node.py", 
                str(i), str(in_port), str(out_port)
            ]
            p = subprocess.Popen(cmd)
            self.processes.append(p)
            print(f"  > Spawned Node {i} (Ports: {in_port} -> {out_port})")

    def run_benchmark(self, duration_sec: int = 5):
        print(f"Waiting 3s for ZMQ nodes to connect...")
        time.sleep(3)
        
        print(f"Running benchmark for {duration_sec}s at {HEARTBEAT_HZ}Hz...")
        start_time = time.time()
        latencies = []
        ticks_sent = 0
        ticks_recvd = 0
        
        def collector_thread():
            nonlocal ticks_recvd
            while (time.time() - start_time) < duration_sec + 1:
                if self.collector.poll(timeout=100):
                    result = self.collector.recv_json()
                    latencies.append((time.perf_counter() - result['ts']) * 1000)
                    ticks_recvd += 1
        
        t = threading.Thread(target=collector_thread)
        t.start()
        
        try:
            while (time.time() - start_time) < duration_sec:
                tick_start = time.perf_counter()
                
                # Inject without waiting
                self.injector.send_json({"ts": time.perf_counter()})
                ticks_sent += 1
                
                # Precise rate limiting to 1000Hz (busy wait for sub-millisecond precision)
                while (time.perf_counter() - tick_start) < TICK_INTERVAL:
                    pass
                
        except KeyboardInterrupt:
            pass
            
        t.join()
        self.report(latencies, ticks_sent, ticks_recvd)

    def report(self, latencies, sent, recvd):
        if not latencies:
            print("No data collected.")
            return
            
        mean_lat = np.mean(latencies)
        std_lat  = np.std(latencies)
        throughput = recvd / 5.0  # approximate
        print("\n--- Pipeline Performance Report ---")
        print(f"Ticks Sent:            {sent}")
        print(f"Ticks Received:        {recvd}")
        print(f"Throughput:            {throughput:.1f} Hz")
        print(f"Average Latency (RTT): {mean_lat:.4f} ms")
        print(f"Jitter (StdDev):       {std_lat:.4f} ms")
        print(f"1000Hz Throughput:     {'✅ PASSED' if throughput > 900 else '❌ FAILED'}")

    def cleanup(self):
        print("Cleaning up...")
        for p in self.processes:
            p.terminate()
        self.injector.close()
        self.collector.close()
        self.context.term()

if __name__ == "__main__":
    orch = PipelineOrchestrator()
    try:
        orch.start_pipeline()
        orch.run_benchmark(5)
    finally:
        orch.cleanup()
