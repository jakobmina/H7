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
│         ~88B Neuronas virtuales / cultivo biológico         │
└──────────────────────┬──────────────────────────────────────┘
                       │  cl-sdk (1000 Hz loop)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Loop Adaptativo H7 (adaptive_cl_loop.py)       │
│   d_symp = O_n(i) = cos(πi)·cos(πφi)  [conservativo]       │
│   d_metr = DRIFT_072 correction        [disipativo]         │
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
┌──────────────────────┐
│   QuoreMindHP        │
│   BayesLogicHP       │
│   50-digit precision │
│   Shannon Entropy    │
└──────────────┬───────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│        Streamlit Dashboard (streamlit_monitor.py)           │
│   Phase Stability · Lagrangian · Stability Index · Stats    │
└─────────────────────────────────────────────────────────────┘
```

---

## El Mandato Metripléctico: Red de Nodos (Box-in-Box)

El sistema implementa una arquitectura jerárquica de nodos que permite modelar estructuras neuronales complejas mediante capas ("Cajas dentro de Cajas"). Cada nodo, desde una neurona individual hasta un contenedor global, cumple con el Mandato Metripléctico:

1. **Componente Simpléctica ($H$)**: Conservación de energía y rotación de fase coherente.
2. **Componente Métrica ($S$)**: Disipación entrópica y relajación hacia atractores de estabilidad.
3. **Operador Áureo ($O_n$)**: El fondo del espacio-tiempo está modulado por $\phi \approx 1.618$ para evitar singularidades.

### Estructuras de la Red (`nodes_network.py`)

- **`Node`**: Unidad básica metripléctica con estados $\psi = [nn_1, nn_0, nn_{-1}]$.
- **`HierarchicalNode`**: Contenedor que agrega recursivamente los Lagrangianos de sus hijos y mantiene su propia dinámica de frontera (membrana).
- **`Neuron`**: Especialización para mapear potenciales de membrana biológicos al formalismo físico.

---

## Estructura del Repositorio

| Archivo | Descripción |
| :--- | :--- |
| `adaptive_cl_loop.py` | Loop CL adaptativo con BayesLogicHP + H7 |
| `nodes_network.py` | Red jerárquica de nodos metriplécticos (Box-in-Box) |
| `h7_quantum_oracle.py` | MetriplexOracle, H7Conservation, Qiskit |
| `quoremind_monitor.py` | Simulación DIT de estabilidad de fase |
| `streamlit_monitor.py` | Dashboard en tiempo real |
| `quoremindhp.py` | QuoreMindHP — lógica bayesiana de alta precisión |
| `tests/test_nodes_network.py` | Pruebas unitarias para la red de nodos y neuronas |
| `setup_cl1.sh` | Setup completo del entorno CL1 |
| `requirements_cl1.txt` | Dependencias fijadas para CL1 |
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
