#!/usr/bin/env python3
"""
h5_to_json.py — Robust HDF5 to JSON Export Utility
Alineado con el Mandato Metripléxico (Regla 1 y 3)

Este script extrae recursivamente datasets y atributos (metadatos)
de archivos .h5 y los convierte a un formato JSON estructurado.
"""

import h5py
import json
import numpy as np
import os
import sys
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
        # Si es un array de bytes, decodificar cada elemento
        if obj.dtype.kind in ('S', 'V', 'O'): # S: bytes, V: void, O: object (strings often here)
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
    """
    Convierte recursivamente un objeto HDF5 (File, Group o Dataset) a un diccionario.
    Captura tanto los DATOS como los ATRIBUTOS (Metadatos).
    """
    result = {}
    
    # Extraer Atributos (Metadatos - Regla 3.3 Diagnóstico)
    attrs = {}
    for key, val in h5_item.attrs.items():
        try:
            attrs[key] = fix_types(val)
        except Exception as e:
            attrs[key] = f"[Error decoding: {e}]"
            
    if attrs:
        result["_metadata"] = attrs

    # Procesar contenido
    if isinstance(h5_item, h5py.Dataset):
        try:
            result["data"] = fix_types(h5_item[...])
        except Exception as e:
            result["data"] = f"[Error reading dataset: {e}]"
    elif isinstance(h5_item, (h5py.Group, h5py.File)):
        for key in h5_item.keys():
            result[key] = h5_to_dict(h5_item[key])
            
    return result

def check_metriplectic_integrity(data_dict):
    """
    Verifica si el archivo contiene etiquetas del Mandato Metripléxico.
    (Regla 1: d_symp, d_metr)
    """
    found_tags = []
    
    def search_recursive(d):
        if isinstance(d, dict):
            for k, v in d.items():
                if k in ["L_symp", "L_metr", "d_symp", "d_metr", "H", "S"]:
                    found_tags.append(k)
                search_recursive(v)
    
    search_recursive(data_dict)
    return list(set(found_tags))

def export_h5_to_json(h5_path, json_path=None):
    if not os.path.exists(h5_path):
        print(f"Error: El archivo {h5_path} no existe.")
        return False

    if json_path is None:
        json_path = os.path.splitext(h5_path)[0] + ".json"

    print(f"Reading H5: {h5_path}")
    try:
        with h5py.File(h5_path, 'r') as f:
            full_data = h5_to_dict(f)
            
            # Verificación Metripléxica
            tags = check_metriplectic_integrity(full_data)
            if tags:
                print(f"  [Mandato Metripléxico] Encontradas etiquetas: {tags}")
            
            # Operador Áureo de Integridad
            full_data["_export_info"] = {
                "timestamp": datetime.now().isoformat(),
                "source": h5_path,
                "O_n_integrity": float(np.cos(np.pi * PHI)) # Regla 2.1
            }

            with open(json_path, 'w', encoding='utf-8') as jf:
                json.dump(full_data, jf, indent=2, ensure_ascii=False, cls=MetriplecticEncoder)
            
            print(f"Export successful: {json_path}")
            return True
    except Exception as e:
        print(f"Error durante la exportación: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python3 h5_to_json.py <archivo.h5> [archivo_salida.json]")
    else:
        path = sys.argv[1]
        out_path = sys.argv[2] if len(sys.argv) > 2 else None
        export_h5_to_json(path, out_path)
