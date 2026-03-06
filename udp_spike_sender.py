import socket
import struct
import cl
import time
from cl.util import Benchmark

# Parámetros de Red
PEER = ("127.0.0.1", 12345)

def start_spike_sender():
    """
    Inicia el loop de captura y envío de spikes vía UDP.
    Sigue el Mandato Metripléctico al encapsular el flujo de información.
    """
    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    
    print(f"🚀 SPIKE SENDER INITIALIZED | TRANSMITTING TO {PEER[0]}:{PEER[1]}")
    
    try:
        with cl.open() as neurons:
            for tick in neurons.loop(
                            ticks_per_second=25000,
                            ignore_jitter=True,
                            stop_after_seconds=60): # Aumentado para demostración
                
                spikes = tick.analysis.spikes
                if spikes:
                    # Empaquetado binario: < (little-endian), Q (unsigned long long 8b), B (unsigned char 1b)
                    first_spike = spikes[0]
                    payload = struct.pack('<QB', first_spike.timestamp, first_spike.channel)
                    
                    if len(spikes) > 1:
                        # Spikes múltiples en el mismo timestamp
                        payload = bytearray(payload)
                        payload.extend([spike.channel for spike in spikes[1:]])
                        
                    udp_socket.sendto(payload, PEER)
    except KeyboardInterrupt:
        print("\n🛑 Spike Sender Stopped by User.")
    finally:
        udp_socket.close()

if __name__ == "__main__":
    start_spike_sender()
