"""
gemma_fragment_node.py — A single node in the distributed Gemma-4 pipeline.
Simulates layer-specific processing latency and passes activations via ZMQ.
"""

import sys
import zmq
import time
import json
import numpy as np

def run_fragment(node_id: int, input_port: int, output_port: int):
    context = zmq.Context()
    
    # Entrance: Pull activations from previous node (CONNECT to PUSH-BIND)
    receiver = context.socket(zmq.PULL)
    receiver.connect(f"tcp://localhost:{input_port}")
    
    # Exit: Push activations to next node (BIND for next PULL-CONNECT)
    sender = context.socket(zmq.PUSH)
    sender.bind(f"tcp://*:{output_port}")
    
    # Node-specific simulated latency (Targeting ~80-100us for 1kHz budget)
    LATENCY_US = 80 / 1_000_000.0 

    print(f"Fragment Node {node_id} active. Listening on :{input_port}, Pushing to :{output_port}")
    
    count = 0
    try:
        while True:
            # Receive activation tensor (serialized as bytes)
            message = receiver.recv()
            
            # Simulate processing (Rule: Fragment should perform a simple transform)
            # data = np.frombuffer(message, dtype=np.float32)
            # data = data * 0.99  # Dummy operation
            
            time.sleep(LATENCY_US)
            
            # Forward activation
            sender.send(message)
            
            count += 1
            if count % 1000 == 0:
                print(f"Node {node_id}: Processed {count} activations.")
                
    except KeyboardInterrupt:
        print(f"Node {node_id} shutting down.")
    finally:
        receiver.close()
        sender.close()
        context.term()

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python gemma_fragment_node.py <id> <input_port> <output_port>")
        sys.exit(1)
        
    node_id = int(sys.argv[1])
    in_port = int(sys.argv[2])
    out_port = int(sys.argv[3])
    
    run_fragment(node_id, in_port, out_port)
