"""
streamlit_monitor.py — CL1 Real-Time Monitor

Lee datos reales del hardware CL1 desde cl1_session.sqlite (WAL).
Muestra: spikes/tick, latencia de loop, P(bottleneck), L_symp vs L_metr,
eventos H7, y todas las estadísticas de sesión en tiempo real.

Modo simulación: si no hay sesión CL1 activa, corre la simulación DIT local.
"""

import streamlit as st
import numpy as np
import time
import pandas as pd
import sys
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Adjust path to allow imports from smopsys
root_path = Path(__file__).parent.parent.absolute()
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from dashboard.cl1_db import CL1Reader
from smopsys.nodes_network import Node

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CL1 Neural Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── DIT Constants ─────────────────────────────────────────────────────────────
PHI        = (1 + np.sqrt(5)) / 2
DRIFT_072  = 7.0 - (2 * np.pi)
DIT_CYAN   = "#00f2ff"
DIT_GREEN  = "#00ff41"
DIT_RED    = "#ff3e3e"
DIT_ORANGE = "#f39c12"

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0d0d0d; }
    .stMetric label { color: #00f2ff !important; font-weight: bold; }
    .stMetric div[data-testid="stMetricValue"] { color: #00ff41 !important; font-size: 1.6rem !important; }
    h1, h2 { color: #00f2ff !important; text-shadow: 0 0 8px #00f2ff44; }
    .stSidebar { background-color: #111827; }
    .block-container { padding-top: 1rem; }
    .stAlert { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 CL1 Neural Monitor")
    st.markdown("**QuoreMindHP + H7 Adaptive Loop**")
    st.divider()

    mode = st.radio("Modo", ["🔴 CL1 Hardware (live)", "🟡 Simulación DIT"], index=0)
    refresh_s = st.slider("Auto-refresh (s)", 1, 10, 2)
    n_ticks   = st.slider("Ticks visibles", 100, 2000, 500, step=100)
 
    st.divider()
    st.markdown("### 🎛️ Control de Nodos (Metriplectic)")
    nn1 = st.slider("NN1 (Fase/Inercia)", -1.0, 1.0, 0.5, step=0.01)
    nn0 = st.slider("NN0 (Estado/Base)",  -1.0, 1.0, 0.0, step=0.01)
    nn_1 = st.slider("NN-1 (Disipación)", -1.0, 1.0, -0.5, step=0.01)
    
    # Instance Node and calculate Lagrangian
    # Usamos n basado en el tiempo para ver fluctuaciones del Operador Áureo
    n_val = (time.time() * 10) % 1000
    node = Node(nn1, nn0, nn_1, n=n_val)
    l_s, l_m = node.compute_lagrangian()
    
    c1, c2 = st.columns(2)
    c1.metric("L_symp (H)", f"{l_s:.4f}")
    c2.metric("L_metr (S)", f"{l_m:.4f}")

    st.divider()
    st.caption("Loop: `adaptive_cl_loop.py`")
    st.caption("Bridge: `cl1_session.sqlite` (WAL)")

# ─── Tabs ──────────────────────────────────────────────────────────────────────
tabs = st.tabs(["📊 Real-Time Monitor", "🧪 Lab Analysis"])

# ════════════════════════════════════════════════════════════════════
#  TAB 1: MONITOR (Hardware + Simulation)
# ════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown("## 🧠 CL1 Neural Monitor — Spike & Loop Analysis")

    # ─── Helper: Simulation DIT ────────────────────────────────────────────────────
    def sim_dit_ticks(n: int) -> pd.DataFrame:
        """Genera ticks simulados con el Operador Áureo para demo sin hardware."""
        data = []
        acc = 0.0
        for i in range(1, n + 1):
            innovation = np.cos(np.pi * i) * np.cos(np.pi * PHI * i)
            acc += innovation + np.random.normal(0, 0.05)
            is_out = abs(acc) > 2.0
            if is_out or i % 7 == 0:
                acc *= DRIFT_072 / 2.0
            data.append({
                "loop_dur_us":     1000 + np.random.normal(0, 40),
                "spike_count":     int(max(0, np.random.poisson(3))),
                "stim_fired":      1 if np.random.random() < 0.04 else 0,
                "prob_bottleneck": max(0, min(1, 0.1 + abs(acc) * 0.05)),
                "l_symp":          innovation,
                "l_metr":          -abs(acc) * (DRIFT_072 / 2.0),
            })
        return pd.DataFrame(data)

    if "hardware" in mode:
        reader = CL1Reader()
        status = reader.session_status()

        # Status banner
        if status == "running":
            st.success("🔴 **CL1 hardware ACTIVO** — Leyendo datos en tiempo real")
        elif status == "done":
            st.info("✅ Sesión completada — mostrando datos históricos")
        else:
            st.warning("⏳ Esperando sesión CL1... Ejecuta `python adaptive_cl_loop.py`")

        # Metric row
        summary = reader.get_summary()
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("Total Ticks",    summary.get("total_ticks", "—"))
        c2.metric("Total Spikes",   summary.get("total_spikes", "—"))
        c3.metric("Stims Fired",    summary.get("stim_count", "—"))
        c4.metric("Skipped Stims",  summary.get("skipped_stims", "—"))
        c5.metric("Mean Latency",   f"{summary.get('mean_dur_us', 0):.1f} µs"  if summary.get("mean_dur_us") else "—")
        c6.metric("Mean P(bottleneck)", f"{summary.get('mean_prob_bottleneck', 0):.3f}" if summary.get("mean_prob_bottleneck") else "—")

        st.divider()

        # Fetch ticks
        ticks = reader.get_recent_ticks(n_ticks)
        if not ticks:
            st.info("Sin datos aún. Inicia el loop CL1.")
        else:
            df = pd.DataFrame(ticks)

            # ── Gráficas principales ─────────────────────────────────────────────
            chart_col, event_col = st.columns([3, 1])
            with chart_col:
                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.40, 0.30, 0.30],
                    subplot_titles=(
                        "Spikes por Tick + Estimulaciones",
                        "Latencia del Loop (µs) + P(bottleneck)",
                        "Lagrangiano: L_symp (conservativo) vs L_metr (disipativo)"
                    ),
                )
                x = list(range(len(df)))

                # Panel 1: Spikes
                fig.add_trace(go.Bar(
                    x=x, y=df["spike_count"],
                    name="Spikes/tick", marker_color=DIT_CYAN, opacity=0.8
                ), row=1, col=1)

                # Stims fired as markers
                stim_idx = df[df["stim_fired"] == 1].index.tolist()
                if stim_idx:
                    fig.add_trace(go.Scatter(
                        x=stim_idx, y=df.loc[stim_idx, "spike_count"],
                        mode="markers", name="Stim fired",
                        marker=dict(color=DIT_GREEN, size=8, symbol="triangle-up")
                    ), row=1, col=1)

                # Panel 2: Latencia + P(bottleneck)
                fig.add_trace(go.Scatter(
                    x=x, y=df["loop_dur_us"],
                    name="Loop dur (µs)", line=dict(color=DIT_ORANGE, width=1.2)
                ), row=2, col=1)
                fig.add_hline(y=1000, line_dash="dot", line_color="#ffffff", opacity=0.3, row=2, col=1)
                fig.add_hline(y=1500, line_dash="dash", line_color=DIT_RED, opacity=0.5, row=2, col=1)

                if "prob_bottleneck" in df.columns:
                    fig.add_trace(go.Scatter(
                        x=x, y=df["prob_bottleneck"] * 1000,
                        name="P(bottleneck)×1000", line=dict(color=DIT_RED, width=1.2, dash="dot"),
                    ), row=2, col=1)

                # Panel 3: Lagrangiano
                if "l_symp" in df.columns:
                    fig.add_trace(go.Scatter(
                        x=x, y=df["l_symp"],
                        name="L_symp (conservative)", line=dict(color=DIT_GREEN, width=1.2)
                    ), row=3, col=1)
                if "l_metr" in df.columns:
                    fig.add_trace(go.Scatter(
                        x=x, y=df["l_metr"],
                        name="L_metr (dissipative)", line=dict(color=DIT_RED, width=1.2)
                    ), row=3, col=1)

                fig.update_layout(
                    height=600, paper_bgcolor="#0d0d0d", plot_bgcolor="#111827",
                    font=dict(color="#cce6ff"),
                    legend=dict(orientation="h", y=-0.06, font=dict(size=10)),
                    margin=dict(l=30, r=10, t=40, b=30),
                )
                fig.update_xaxes(showgrid=True, gridcolor="#1e293b")
                fig.update_yaxes(showgrid=True, gridcolor="#1e293b")
                st.plotly_chart(fig, use_container_width=True)

            # ── H7 Events ────────────────────────────────────────────────────────
            with event_col:
                st.markdown("#### ⚡ Eventos H7")
                events = reader.get_h7_events(30)
                if events:
                    for ev in reversed(events[-15:]):
                        color = DIT_RED if ev["event_type"] == "reduce_complexity" else DIT_GREEN
                        st.markdown(
                            f"<span style='color:{color}'>●</span> "
                            f"**{ev['event_type']}** `P={ev['value']:.3f}`<br>"
                            f"<small>{ev['description']}</small>",
                            unsafe_allow_html=True
                        )
                else:
                    st.caption("Sin eventos H7 aún")

                st.divider()
                st.markdown("#### 📊 Stats")
                if summary:
                    st.metric("Max Latency",  f"{summary.get('max_dur_us', 0):.0f} µs")
                    st.metric("Min Latency",  f"{summary.get('min_dur_us', 0):.0f} µs")

        if status == "running":
            time.sleep(refresh_s)
            st.rerun()
            
    else:
        st.info("🟡 Modo Simulación DIT — sin hardware CL1 conectado")

        with st.sidebar:
            st.markdown("### ⚙️ Parámetros DIT")
            total_cycles = st.slider("Ciclos", 100, 2000, 400, step=50)
            threshold    = st.slider("Threshold", 0.5, 5.0, 2.0, step=0.1)
            run_btn      = st.button("▶ Simular", type="primary", use_container_width=True)

        if run_btn:
            df = sim_dit_ticks(total_cycles)
            x  = list(range(len(df)))

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Spikes",  int(df["spike_count"].sum()))
            c2.metric("Stims Fired",   int(df["stim_fired"].sum()))
            c3.metric("Mean Latency",  f"{df['loop_dur_us'].mean():.1f} µs")

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                subplot_titles=("Spikes/tick (DIT simulated)", "L_symp vs L_metr"))
            fig.add_trace(go.Bar(x=x, y=df["spike_count"], name="Spikes", marker_color=DIT_CYAN, opacity=0.8), row=1, col=1)
            fig.add_trace(go.Scatter(x=x, y=df["l_symp"],  name="L_symp", line=dict(color=DIT_GREEN, width=1.3)), row=2, col=1)
            fig.add_trace(go.Scatter(x=x, y=df["l_metr"],  name="L_metr", line=dict(color=DIT_RED,   width=1.3)), row=2, col=1)
            fig.update_layout(height=500, paper_bgcolor="#0d0d0d", plot_bgcolor="#111827",
                              font=dict(color="#cce6ff"), margin=dict(l=30, r=10, t=40, b=30))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("👈 Configura los parámetros y haz clic en **▶ Simular**")

# ════════════════════════════════════════════════════════════════════
#  TAB 2: LAB ANALYSIS
# ════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## 🧪 H7 Lab Utilities")
    
    col_exp, col_h5 = st.columns(2)
    
    with col_exp:
        st.subheader("🚀 H7 Experiment Launcher")
        st.write("Lanza experimentos pre-configurados con el Mandato Metripléxico.")
        
        exp_k = st.slider("Disipación K+ (S-Factor)", 0.0, 1.0, 0.1, step=0.01)
        exp_f = st.slider("Frecuencia Base (H-Base)", 0.1, 10.0, 1.0, step=0.1)
        
        if st.button("Lanzar Φ-Modulated Experiment (30s)", help="Ejecuta h7_phi_experiment.py"):
            with st.spinner("Ejecutando experimento..."):
                try:
                    # Script path relative to root
                    script_path = root_path / "smopsys" / "h7_phi_experiment.py"
                    res = subprocess.run([
                        sys.executable, str(script_path), 
                        "--duration", "30",
                        "--k_factor", str(exp_k),
                        "--f0", str(exp_f)
                    ], capture_output=True, text=True, timeout=60, cwd=str(root_path))
                    if res.returncode == 0:
                        st.success("✅ Experimento completado con éxito.")
                        st.code(res.stdout[-500:], language="text")
                        # Forzar recarga de la lista de archivos para el convertidor
                        st.rerun()
                    else:
                        st.error(f"❌ Error en el experimento: {res.stderr}")
                except Exception as e:
                    st.error(f"Error fatal: {e}")
        
        diag_img = Path("h7_ionic_diagnostic.png")
        if diag_img.exists():
            st.image(str(diag_img), caption="Regla 3.3: Diagnóstico de última corrida")

    with col_h5:
        st.subheader("📂 H5 to JSON Converter")
        st.write("Convierte grabaciones binarias H5 a JSON preservando la integridad metripléxica.")
        
        # Search for .h5 in root and dashboard
        h5_files = glob.glob(str(root_path / "*.h5")) + glob.glob("*.h5")
        h5_files = list(set([str(Path(f).absolute()) for f in h5_files]))
        
        # Ordenar por fecha de modificación (más nuevos primero)
        h5_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
        if not h5_files:
            st.warning("No se encontraron archivos .h5 en el directorio.")
        else:
            selected_h5 = st.selectbox("Selecciona archivo H5", h5_files)
            if st.button("Convertir a JSON"):
                with st.spinner(f"Convirtiendo {selected_h5}..."):
                    try:
                        # h5_to_json.py is in the same folder as this script
                        script_path = Path(__file__).parent / "h5_to_json.py"
                        res = subprocess.run([sys.executable, str(script_path), selected_h5], capture_output=True, text=True)
                        if res.returncode == 0:
                            st.success(f"✅ {selected_h5} convertido exitosamente.")
                            json_file = selected_h5.replace(".h5", ".json")
                            st.info(f"Archivo generado: `{json_file}`")
                        else:
                            st.error(f"❌ Error en conversión: {res.stderr}")
                    except Exception as e:
                        st.error(f"Error: {e}")

    st.divider()
    st.markdown("### 📜 Metriplectic Rules Compliance")
    st.info("""
    **Regla 1**: Conservación (H) vs Disipación (S) | **Regla 2**: Fondo Estructurado ($O_n$) | **Regla 3**: Código como Teoría
    """)
