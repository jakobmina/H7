#!/usr/bin/env python3
"""
h5_to_json.py — Robust HDF5 to JSON Export Utility (Batch Edition)
Alineado con el Mandato Metripléxico (Regla 1 y 3)

Este script extrae recursivamente datasets y atributos (metadatos)
de archivos .h5 y los convierte a un formato JSON estructurado.
Soporta procesamiento por lotes (batch processing).
"""

import h5py
import json
import numpy as np
import os
import sys
import glob
from datetime import datetime

# ─── Constantes Metripléxicas ──────────────────────────────────────────────────
PHI = (1 + np.sqrt(5)) / 2

class MetriplecticEncoder(json.JSONEncoder):
    """Encoder personalizado para manejar tipos no serializables de HDF5/Numpy."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, bytes):
            return obj.decode('utf-8', errors='replace')
        if isinstance(obj, h5py.Empty):
            return None
        return super().default(obj)

def fix_types(obj):
    """Pre-procesamiento recursivo para asegurar compatibilidad JSON."""
    if isinstance(obj, h5py.Empty):
        return None
    elif isinstance(obj, dict):
        return {k: fix_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [fix_types(i) for i in obj]
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    elif isinstance(obj, (np.ndarray, np.generic)):
        if obj.dtype.kind in ('S', 'V', 'O'): 
            if obj.ndim == 0:
                val = obj.item()
                if isinstance(val, bytes):
                    return val.decode('utf-8', errors='replace')
                return str(val)
            flat = obj.flatten()
            decoded = []
            for i in flat:
                if isinstance(i, bytes):
                    decoded.append(i.decode('utf-8', errors='replace'))
                else:
                    decoded.append(str(i))
            return decoded
        return obj.tolist() if isinstance(obj, np.ndarray) else obj.item()
    return obj

def h5_to_dict(h5_item):
    """Convierte recursivamente un objeto HDF5 a diccionario."""
    result = {}
    attrs = {}
    for key, val in h5_item.attrs.items():
        try:
            attrs[key] = fix_types(val)
        except Exception as e:
            attrs[key] = f"[Error decoding: {e}]"
            
    if attrs:
        result["_metadata"] = attrs

    if isinstance(h5_item, h5py.Dataset):
        try:
            result["data"] = fix_types(h5_item[...])
        except Exception as e:
            result["data"] = f"[Error reading dataset: {e}]"
    elif isinstance(h5_item, (h5py.Group, h5py.File)):
        for key in h5_item.keys():
            result[key] = h5_to_dict(h5_item[key])
            
    return result

def export_h5_to_json(h5_path, json_path=None):
    if not os.path.exists(h5_path):
        print(f"Error: El archivo {h5_path} no existe.")
        return False

    if json_path is None:
        # Generar nombre automático: recording.h5 -> recording.json
        json_path = os.path.splitext(h5_path)[0] + ".json"

    print(f"Processing: {h5_path} -> {json_path}")
    try:
        with h5py.File(h5_path, 'r') as f:
            full_data = h5_to_dict(f)
            full_data["_export_info"] = {
                "timestamp": datetime.now().isoformat(),
                "source": h5_path,
                "O_n_integrity": float(np.cos(np.pi * PHI))
            }

            with open(json_path, 'w', encoding='utf-8') as jf:
                json.dump(full_data, jf, indent=2, ensure_ascii=False, cls=MetriplecticEncoder)
            
            return True
    except Exception as e:
        print(f"  [!] Error en {h5_path}: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso:")
        print("  Manual: python3 h5_to_json.py <archivo.h5> [salida.json]")
        print("  Auto (todos): python3 h5_to_json.py --all")
        sys.exit(1)

    if sys.argv[1] == "--all":
        h5_files = glob.glob("*.h5")
        if not h5_files:
            print("No se encontraron archivos .h5 en el directorio actual.")
        else:
            print(f"Detectados {len(h5_files)} archivos para convertir...")
            for f in h5_files:
                export_h5_to_json(f)
    else:
        path = sys.argv[1]
        out_path = sys.argv[2] if len(sys.argv) > 2 else None
        export_h5_to_json(path, out_path)
