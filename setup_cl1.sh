#!/usr/bin/env bash
# =============================================================================
# setup_cl1.sh — Entorno Aislado para CL1 Hardware (Cortical Labs)
#
# Crea un venv limpio con versiones fijadas de cl-sdk, pydantic y QuoreMindHP.
# Diseñado para ejecutarse en el sistema donde el CL1 está conectado físicamente.
#
# Uso:
#   chmod +x setup_cl1.sh
#   ./setup_cl1.sh
#   source cl1_env/bin/activate
#   python adaptive_cl_loop.py
#
# Autor: Jacobo Tlacaelel Mina Rodriguez
# =============================================================================

set -e  # Salir ante cualquier error

VENV_NAME="cl1_env"
PYTHON_BIN="python3"          # Cambiar a 'python3.11' si se requiere versión específica
QUOREMIND_SRC="$HOME/quoreMind/QuoreMind-Metriplectic"

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║         CORTICAL CL1 — Setup de Entorno Hardware        ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# ── 1. Crear virtualenv aislado ────────────────────────────────────────────────
if [ -d "$VENV_NAME" ]; then
    echo "⚠  El entorno '$VENV_NAME' ya existe. Eliminando para reinstalar limpio..."
    rm -rf "$VENV_NAME"
fi

echo "📦 Creando entorno virtual '$VENV_NAME'..."
$PYTHON_BIN -m venv "$VENV_NAME"
source "$VENV_NAME/bin/activate"

# Actualizar pip dentro del venv
pip install --upgrade pip setuptools wheel -q

# ── 2. Instalar cl-sdk con pydantic compatible ─────────────────────────────────
echo ""
echo "📡 Instalando cl-sdk (hardware CL1)..."
# cl-sdk 0.29.0 usa la sintaxis `type X = Annotated[...]` de Python 3.12+
# pero sus modelos necesitan arbitrary_types_allowed — se instala aquí con el patch aplicado.
pip install "cl-sdk==0.29.0" -q

echo "🔧 Aplicando parche de compatibilidad Pydantic → cl-sdk..."

# Patch: agregar arbitrary_types_allowed=True al base model de resultados
CL_BASE="$VENV_NAME/lib/python*/site-packages/cl/analysis/_results/_base_result.py"
for f in $CL_BASE; do
    if [ -f "$f" ]; then
        # Insertar ConfigDict import después de __future__
        sed -i '/from __future__ import annotations/a from pydantic import ConfigDict' "$f"
        sed -i 's/class AnalysisResult(BaseModel):/class AnalysisResult(BaseModel):\n    model_config = ConfigDict(arbitrary_types_allowed=True)/' "$f"
        echo "   ✅ Parcheado: $f"
    fi
done

# Patch: StimPulseWidthMicroSeconds → Any (incompatible con pydantic 2.x)
CL_MODEL="$VENV_NAME/lib/python*/site-packages/cl/app/model/model.py"
for f in $CL_MODEL; do
    if [ -f "$f" ]; then
        # Añadir Any import al inicio del archivo
        sed -i '1s/^/from typing import Any\n/' "$f"
        # Reemplazar definición problemática
        python3 - "$f" <<'PYEOF'
import sys
path = sys.argv[1]
with open(path, 'r') as fh:
    content = fh.read()

import re
content = re.sub(
    r'type StimPulseWidthMicroSeconds = Annotated\[[\s\S]*?\]\n"""',
    'type StimPulseWidthMicroSeconds = Any\n"""',
    content
)
with open(path, 'w') as fh:
    fh.write(content)
print(f"   ✅ Parcheado: {path}")
PYEOF
    fi
done

# ── 3. Instalar dependencias científicas (versiones CL1-compatibles) ────────────
echo ""
echo "🔬 Instalando dependencias científicas..."
pip install -r requirements_cl1.txt -q

# ── 4. Instalar QuoreMindHP desde fuente local ─────────────────────────────────
echo ""
if [ -d "$QUOREMIND_SRC" ]; then
    echo "🧠 Instalando QuoreMindHP desde fuente local: $QUOREMIND_SRC"
    pip install -e "$QUOREMIND_SRC" -q
else
    echo "⚠  No se encontró QuoreMindHP en $QUOREMIND_SRC"
    echo "   Copiando quoremindhp.py local al entorno..."
    cp quoremindhp.py "$VENV_NAME/lib/python*/site-packages/" 2>/dev/null || \
    echo "   Usa PYTHONPATH=. para importar quoremindhp.py localmente"
fi

# ── 5. Verificación funcional ──────────────────────────────────────────────────
echo ""
echo "🧪 Verificando instalación..."

python - <<'PYEOF'
import sys

errors = []

# Verificar cl
try:
    import cl
    print("   ✅ cl-sdk importado correctamente")
except Exception as e:
    errors.append(f"   ❌ cl-sdk: {e}")

# Verificar QuoreMindHP
try:
    from quoremindhp import BayesLogicHP, StatisticalAnalysisHP
    import mpmath
    bayes = BayesLogicHP()
    result = bayes.calculate_posterior_probability(
        mpmath.mpf("0.1"), mpmath.mpf("0.3"), mpmath.mpf("0.9")
    )
    print(f"   ✅ QuoreMindHP BayesLogicHP OK → P(bottleneck)={float(result):.4f}")
except Exception as e:
    errors.append(f"   ❌ QuoreMindHP: {e}")

    # Verificar Qiskit
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        qc = QuantumCircuit(1)
        qc.h(0)
        # Verificar que draw('mpl') no explote (requiere pylatexenc)
        try:
            import pylatexenc
            print("   ✅ Qiskit + AerSimulator + pylatexenc OK")
        except ImportError:
            print("   ⚠  Qiskit OK, pero falta 'pylatexenc' para visualizaciones MPL")
    except Exception as e:
        errors.append(f"   ❌ Qiskit: {e}")

    # Verificar Streamlit e IPyKernel
    try:
        import streamlit
        import ipykernel
        print(f"   ✅ Streamlit {streamlit.__version__} + IPyKernel OK")
    except Exception as e:
        errors.append(f"   ❌ Tooling (Streamlit/IPyKernel): {e}")

if errors:
    print("\n⚠  Errores encontrados:")
    for err in errors:
        print(err)
    sys.exit(1)
else:
    print("\n✅ Entorno CL1 listo para hardware físico.")
PYEOF

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║  Setup completo. Para activar el entorno:               ║"
echo "║    source cl1_env/bin/activate                          ║"
echo "║                                                          ║"
echo "║  Para ejecutar el loop adaptativo en CL1:              ║"
echo "║    python adaptive_cl_loop.py                           ║"
echo "║                                                          ║"
echo "║  Para el monitor Streamlit:                             ║"
echo "║    streamlit run streamlit_monitor.py                   ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
