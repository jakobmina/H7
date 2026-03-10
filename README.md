# Cortical: CL1 — Adaptive Neural Closed-Loop with QuoreMindHP & H7

> **Hardware objetivo:** Cortical Labs CL1 — MEA de 64 electrodos con neuronas biológicas vivas.

Framework de procesamiento adaptativo de lazo cerrado para el CL1. Integra análisis
de alta precisión (QuoreMindHP), lógica bayesiana, el algoritmo H7 y simulación
cuántica (Qiskit) para optimizar el control en tiempo real de estimulación neuronal.

---

## Arquitectura del Sistema

```text
┌─────────────────────────────────────────────────────────────┐
│                    CL1 Hardware (MEA 64ch)                  │
│         ~88B Neuronas virtuales / cultivo biológico    ┌─────────────────────────────────────────────────────────────┐
│              Loop Adaptativo H7 (smopsys/)                  │
│   d_symp = O_n(i) = cos(πi)·cos(πφi)  [conservativo]       │
│   d_metr = [u, S] (Ionic Emulation Na+/K+) [disipativo]    │
└──────────────┬──────────────────────────┬───────────────────┘
               │                          │
               ▼                          ▼
┌──────────────────────┐    ┌─────────────────────────────────┐
│   Red Metripléctica   │    │   MetriplexOracle + H7          │
│   (nodes_network.py)  │    │   (h7_quantum_oracle.py)        │
│   Box-in-Box Nesting  │    │   Qiskit · Simon's Algorithm    │
│   Neuron Layers       │    │   s=7 hidden symmetry           │
└──────────────┬───────┘    └─────────────────────────────────┐
               │
               ▼
┌──────────────────────┐    ┌─────────────────────────────────┐
│   QuoreMindHP        │    │      Lab Analysis (H7)          │
│   BayesLogicHP       │    │   h7_phi_experiment.py          │
│   50-digit precision │    │   h5_to_json.py Utility         │
└──────────────┬───────┘    └────────────────┬────────────────┘
               │                             │
               ▼                             ▼
┌─────────────────────────────────────────────────────────────┐
│        Dashboards (dashboard/streamlit_monitor.py)          │
│   1. Real-Time Monitor: Spikes · Lagrangian · H7 Events      │
│   2. Lab Analysis: Experimento Φ · H5-to-JSON Converter     │
└─────────────────────────────────────────────────────────────┘
```

---

## Estructura del Proyecto (Modular)

El proyecto se divide en dos módulos principales para coherencia física y operativa:

### 🧪 `smopsys/` (Core Physics & Experiments)
- **`h7_phi_experiment.py`**: Experimento de estímulo modulado por $\phi$ con emulación iónica Na+/K+.
- **`brain_topology.py`**: Generador de topología cerebral modular (Box-in-Box) alineado con $O_n$.
- **`nodes_network.py`**: Motor de red metripléptica jerárquica.
- **`h7_quantum_oracle.py`**: Oráculo cuántico Metriplex (Qiskit).
- **`adaptive_cl_loop.py`**: Bucle de control adaptativo CL1.

### 📊 `dashboard/` (Monitoring & Utilities)
- **`streamlit_monitor.py`**: Consola de mando central con visualización de Lagrangianos.
- **`cl1_db.py`**: Interfaz de base de datos para lectura/escritura en tiempo real.
- **`h5_to_json.py`**: Utilidad de exportación de datos preservando metadatos $O_n$.

### 📂 `records/` & `tests/`
- **`records/`**: Grabaciones `.h5`, sesiones `.sqlite` y diagnósticos `.png`.
- **`tests/`**: Suite completa de `pytest` para validación de lógica física y topológica.

---

## Ejecución

```bash
# Lanzar el dashboard central
streamlit run dashboard/streamlit_monitor.py

# Ejecutar tests de integridad
pytest tests/
```
alización volumétrica en tiempo real:
- **Topología de Lóbulos**: Distribución de clusters en un elipsoide prolate.
- **Level of Detail (LOD)**: Navegación jerárquica desde la red global (Nivel 0) hasta grupos funcionales (Nivel 1) y nodos locales (Nivel 2).

---

## Estructura del Repositorio

| Archivo | Descripción |
| :--- | :--- |
| `adaptive_cl_loop.py` | Loop CL adaptativo con BayesLogicHP + H7 |
| `nodes_network.py` | Red jerárquica de nodos metriplécticos (3D + Box-in-Box) |
| `brain_topology.py` | Generador de topología cerebral y distribución de nodos |
| `brain_3d_monitor.py` | Monitor 3D jerárquico con LOD (Plotly) |
| `h7_quantum_oracle.py` | MetriplexOracle, H7Conservation, Qiskit |
| `quoremind_monitor.py` | Simulación DIT de estabilidad de fase |
| `streamlit_monitor.py` | Dashboard 2D en tiempo real (Fix compatible) |
| `quoremindhp.py` | QuoreMindHP — lógica bayesiana de alta precisión |
| `tests/test_nodes_network.py` | Pruebas unitarias para la red de nodos y neuronas |
| `setup_cl1.sh` | Setup completo del entorno CL1 (Fix Qiskit & Pydantic) |
| `requirements_cl1.txt` | Dependencias (Qiskit 1.0+, Aer, Pylatexenc, IPyKernel) |
| `requirements.txt` | Dependencias generales de desarrollo |

---

## Ejecución y Pruebas

```bash
# Ejecutar pruebas de la red metripléctica
pytest tests/test_nodes_network.py

# Simulación de estabilidad de fase
python quoremind_monitor.py
```

---

## Constantes del Sistema

| Constante | Valor | Rol |
| :--- | :--- | :--- |
| `φ` (Golden Ratio) | `1.6180339887` | Operador Áureo $O_n$ |
| `DRIFT_072` | `7 - 2π ≈ 0.7168` | Corrección métrica |
| `s` (H7 symmetry) | `7` (binary 111) | Simetría oculta del oráculo |
| `PRECISION_DPS` | `50` | Dígitos decimales en mpmath |
| `TICKS_PER_SECOND` | `1000` | Frecuencia del loop CL |

---

## Notas de Compatibilidad

> **cl-sdk 0.29.0 + Pydantic ≥ 2.5**: El script `setup_cl1.sh` aplica parches automáticamente
> sobre `_base_result.py` y `model.py` para habilitar `arbitrary_types_allowed=True`
> y reemplazar tipos incompatibles (`StimPulseWidthMicroSeconds`, `Array1DInt`).
> **No modificar manualmente.** El entorno `cl1_env` está aislado del sistema.

---

### Autoría

_Jacobo Tlacaelel Mina Rodriguez · QuoreMind Framework · 2026_
