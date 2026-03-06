# Cortical: CL1 — Adaptive Neural Closed-Loop with QuoreMindHP & H7

> **Hardware objetivo:** Cortical Labs CL1 — MEA de 64 electrodos con neuronas biológicas vivas.

Framework de procesamiento adaptativo de lazo cerrado para el CL1. Integra análisis
de alta precisión (QuoreMindHP), lógica bayesiana, el algoritmo H7 y simulación
cuántica (Qiskit) para optimizar el control en tiempo real de estimulación neuronal.

---

## Arquitectura del Sistema

```
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
│   QuoreMindHP        │    │   MetriplexOracle + H7          │
│   BayesLogicHP       │    │   (h7_quantum_oracle.py)        │
│   50-digit precision │    │   Qiskit · Simon's Algorithm    │
│   Shannon Entropy    │    │   s=7 hidden symmetry           │
└──────────────┬───────┘    └─────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────┐
│        Streamlit Dashboard (streamlit_monitor.py)           │
│   Phase Stability · Lagrangian · Stability Index · Stats    │
└─────────────────────────────────────────────────────────────┘
```

---

## Objetivos del Proyecto

1. **Análisis de performance CL** — Medir latencia de loop (`loop_timing_test.py`), cuantificar "Skipped stims" y percentiles de jitter a 1000 Hz.
2. **Integración QuoreMindHP** — `BayesLogicHP.calculate_posterior_probability()` + `StatisticalAnalysisHP.shannon_entropy()` para análisis de alta precisión sobre spikes y timing.
3. **Algoritmo H7** — Marco de decisión basado en la simetría topológica H7 (`s=7`, pares `x↔(7⊕x)`) para adaptar parámetros de estimulación en tiempo real.
4. **CL Adaptativo** — Ajuste dinámico de amplitud/duración de pulso basado en `P(bottleneck)` bayesiana.
5. **Validación** — Re-ejecución de tests de timing y comparación de skipped stims antes/después.
6. **Resumen final** — Bottlenecks identificados, soluciones implementadas, mejoras cuantificadas.

---

## Setup para Hardware CL1

### Requisitos del sistema

- **Python 3.12+**
- **CL1 conectado físicamente** (USB o red local)
- **cl-sdk 0.29.0** (con parche de compatibilidad Pydantic incluido en el script)

### Instalación (una sola vez)

```bash
git clone <este-repositorio>
cd cortical

# Crear entorno aislado y aplicar parches automáticamente
chmod +x setup_cl1.sh
./setup_cl1.sh

# Activar entorno
source cl1_env/bin/activate
```

### Ejecución en CL1

```bash
# 1. Activar entorno
source cl1_env/bin/activate

# 2. Test de timing base (30s, 1000 Hz)
python loop_timing_test.py

# 3. Loop adaptativo H7 + QuoreMindHP
python adaptive_cl_loop.py

# 4. Monitor en tiempo real
streamlit run streamlit_monitor.py

# 5. Simulación cuántica H7 (sin hardware)
python h7_quantum_oracle.py
```

---

## Estructura del Repositorio

| Archivo | Descripción |
|---------|-------------|
| `adaptive_cl_loop.py` | Loop CL adaptativo con BayesLogicHP + H7 |
| `loop_timing_test.py` | Benchmark de latencia de loop (baseline) |
| `h7_quantum_oracle.py` | MetriplexOracle, H7Conservation, Qiskit |
| `quoremind_monitor.py` | Simulación DIT de estabilidad de fase |
| `streamlit_monitor.py` | Dashboard en tiempo real |
| `quoremindhp.py` | QuoreMindHP — lógica bayesiana de alta precisión |
| `setup_cl1.sh` | Setup completo del entorno CL1 |
| `requirements_cl1.txt` | Dependencias fijadas para CL1 |
| `requirements.txt` | Dependencias generales de desarrollo |

---

## Constantes del Sistema

| Constante | Valor | Rol |
|-----------|-------|-----|
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

*Jacobo Tlacaelel Mina Rodriguez · QuoreMind Framework · 2026*
