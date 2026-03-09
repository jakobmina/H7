import os
import h5py
import json
import numpy as np
import pytest
from h5_to_json import export_h5_to_json

@pytest.fixture
def temp_h5_file(tmp_path):
    h5_path = tmp_path / "test_data.h5"
    with h5py.File(h5_path, "w") as f:
        # Dataset simple
        d1 = f.create_dataset("simple_data", data=np.array([1, 2, 3]))
        d1.attrs["label"] = "test_label"
        
        # Grupo jerárquico
        group = f.create_group("metriplectic")
        group.attrs["Mandato"] = "Regla 1"
        ds = group.create_dataset("L_symp", data=np.array([0.5, 0.6]))
        dm = group.create_dataset("L_metr", data=np.array([-0.1, -0.2]))
        
    return str(h5_path)

def test_export_h5_to_json(temp_h5_file):
    json_path = temp_h5_file.replace(".h5", ".json")
    success = export_h5_to_json(temp_h5_file, json_path)
    
    assert success is True
    assert os.path.exists(json_path)
    
    with open(json_path, "r") as f:
        data = json.load(f)
        
    # Verificar estructura principal
    assert "simple_data" in data
    assert data["simple_data"]["_metadata"]["label"] == "test_label"
    assert data["simple_data"]["data"] == [1, 2, 3]
    
    # Verificar jerarquía y etiquetas metripléxicas
    assert "metriplectic" in data
    assert data["metriplectic"]["_metadata"]["Mandato"] == "Regla 1"
    assert "L_symp" in data["metriplectic"]
    assert data["metriplectic"]["L_symp"]["data"] == [0.5, 0.6]
    assert "L_metr" in data["metriplectic"]
    assert data["metriplectic"]["L_metr"]["data"] == [-0.1, -0.2]
    
    # Verificar info de exportación
    assert "_export_info" in data
    assert "O_n_integrity" in data["_export_info"]

def test_missing_file():
    success = export_h5_to_json("non_existent_file.h5")
    assert success is False
