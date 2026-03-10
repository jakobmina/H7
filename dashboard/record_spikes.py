import cl
import time
import numpy as np
from pathlib import Path
import sys

# Adjust path to allow imports from smopsys
root_path = Path(__file__).parent.parent.absolute()
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from smopsys.nodes_network import Node

def record_session(duration=3, nn1=0.5, nn0=0.0, nn_1=-0.5):
    """
    Graba una sesión de spikes con control preciso de frames.
    """
    print(f"🚀 INITIALIZING REFINED RECORDING | DURATION: {duration}s")
    
    # Calcular estado físico actual para los metadatos
    node = Node(nn1, nn0, nn_1, n=time.time())
    l_symp, l_metr = node.compute_lagrangian()
    
    with cl.open() as neurons:
        print(f"📊 Physical State: L_symp={l_symp:.4f}, L_metr={l_metr:.4f}")
        
        # Calcular frames totales basados en el framerate real del chip
        fps = neurons.get_frames_per_second()
        total_frames = int(fps * duration)
        
        print(f"📡 Chip FPS: {fps} | Target Frames: {total_frames}")
        
        # Iniciar grabación con parada automática por frames
        recording = neurons.record(stop_after_frames=total_frames)
        
        print("🟢 Recording in progress... (Synchronous wait)")
        recording.wait_until_stopped()
        
    attrs = recording.attributes
    path = recording.file['path']
    
    print("\n✅ REFINED RECORDING COMPLETE")
    print(f"📁 File: {path}")
    print(f"⏱️ Duration: {attrs['duration_seconds']:.2f}s ({attrs['duration_frames']} frames)")
    print(f"🧬 Metriplectic Tag: [LS:{l_symp:.3f}|LM:{l_metr:.3f}]")
    
    return path

if __name__ == "__main__":
    record_session(duration=3)
