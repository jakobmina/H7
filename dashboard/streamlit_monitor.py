"""
streamlit_monitor.py — CL1 Real-Time Monitor

Reads real data from CL1 hardware from cl1_session.sqlite (WAL).
Shows: spikes/tick, loop latency, P(bottleneck), L_symp vs L_metr,
H7 events, and all session statistics in real time.

Simulation mode: if no active CL1 session, runs local DIT simulation.
"""

import streamlit as st
import numpy as np
import time
import pandas as pd
import sys
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Adjust path to allow imports from smopsys
root_path = Path(__file__).parent.parent.absolute()
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from dashboard.cl1_db import CL1Reader
# nodes_network.py is now in dashboard/
from dashboard.nodes_network import Node

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

    mode = st.radio("Mode", ["🔴 CL1 Hardware (live)", "🟡 DIT Simulation"], index=0)
    refresh_s = st.slider("Auto-refresh (s)", 1, 10, 2)
    n_ticks   = st.slider("Visible Ticks", 100, 2000, 500, step=100)
 
    st.divider()
    st.markdown("### 🎛️ Node Control (Metriplectic)")
    nn1 = st.slider("NN1 (Phase/Inertia)", -1.0, 1.0, 0.5, step=0.01)
    nn0 = st.slider("NN0 (State/Base)",  -1.0, 1.0, 0.0, step=0.01)
    nn_1 = st.slider("NN-1 (Dissipation)", -1.0, 1.0, -0.5, step=0.01)
    
    # Instance Node and calculate Lagrangian
    # Use n based on time to see Golden Operator fluctuations
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
        """Generates simulated ticks with the Golden Operator for demo without hardware."""
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
            st.success("🔴 **CL1 hardware ACTIVE** — Reading real-time data")
        elif status == "done":
            st.info("✅ Session completed — showing historical data")
        else:
            st.warning("⏳ Waiting for CL1 session... Run `python adaptive_cl_loop.py`")

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
            st.info("No data yet. Start the CL1 loop.")
        else:
            df = pd.DataFrame(ticks)

            # ── Main Charts ─────────────────────────────────────────────
            chart_col, event_col = st.columns([3, 1])
            with chart_col:
                fig = make_subplots(
                    rows=3, cols=1, shared_xaxes=True,
                    row_heights=[0.40, 0.30, 0.30],
                    subplot_titles=(
                        "Spikes per Tick + Stimulations",
                        "Loop Latency (µs) + P(bottleneck)",
                        "Lagrangian: L_symp (conservative) vs L_metr (dissipative)"
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
                st.markdown("#### ⚡ H7 Events")
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
                    st.caption("No H7 events yet")

                st.divider()
                st.markdown("#### 📊 Stats")
                if summary:
                    st.metric("Max Latency",  f"{summary.get('max_dur_us', 0):.0f} µs")
                    st.metric("Min Latency",  f"{summary.get('min_dur_us', 0):.0f} µs")

        if status == "running":
            time.sleep(refresh_s)
            st.rerun()
            
    else:
        st.info("🟡 DIT Simulation Mode — no CL1 hardware connected")

        with st.sidebar:
            st.markdown("### ⚙️ DIT Parameters")
            total_cycles = st.slider("Cycles", 100, 2000, 400, step=50)
            threshold    = st.slider("Threshold", 0.5, 5.0, 2.0, step=0.1)
            run_btn      = st.button("▶ Simulate", type="primary", use_container_width=True)

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
            st.info("👈 Configure the parameters and click on **▶ Simulate**")

# ════════════════════════════════════════════════════════════════════
#  TAB 2: LAB ANALYSIS
# ════════════════════════════════════════════════════════════════════
with tabs[1]:
    st.markdown("## 🧪 H7 Lab Utilities")
    
    col_exp, col_h5 = st.columns(2)
    
    with col_exp:
        st.subheader("🚀 H7 Experiment Launcher")
        st.write("Launch pre-configured experiments with the Metriplectic Mandate.")
        
        exp_k = st.slider("Dissipation K+ (S-Factor)", 0.0, 1.0, 0.1, step=0.01)
        exp_f = st.slider("Base Frequency (H-Base)", 0.1, 10.0, 1.0, step=0.1)
        
        if st.button("Launch Φ-Modulated Experiment (30s)", help="Executes h7_phi_experiment.py"):
            with st.spinner("Running experiment..."):
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
                        st.success("✅ Experiment successfully completed.")
                        st.code(res.stdout[-500:], language="text")
                        # Force file list reload for the converter
                        st.rerun()
                    else:
                        st.error(f"❌ Experiment error: {res.stderr}")
                except Exception as e:
                    st.error(f"Fatal error: {e}")
        
        diag_img = Path("h7_ionic_diagnostic.png")
        if diag_img.exists():
            st.image(str(diag_img), caption="Rule 3.3: Last run diagnostic")

    with col_h5:
        st.subheader("📂 H5 to JSON Converter")
        st.write("Convert H5 binary recordings to JSON preserving metriplectic integrity.")
        
        # Search for .h5 in root and dashboard
        h5_files = glob(str(root_path / "*.h5")) + glob.glob("*.h5")
        h5_files = list(set([str(Path(f).absolute()) for f in h5_files]))
        
        # Ordenar por fecha de modificación (más nuevos primero)
        h5_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
        if not h5_files:
            st.warning("No .h5 files found in the directory.")
        else:
            selected_h5 = st.selectbox("Select H5 file", h5_files)
            if st.button("Convert to JSON"):
                with st.spinner(f"Converting {selected_h5}..."):
                    try:
                        # h5_to_json.py is in the same folder as this script
                        script_path = Path(__file__).parent / "h5_to_json.py"
                        res = subprocess.run([sys.executable, str(script_path), selected_h5], capture_output=True, text=True)
                        if res.returncode == 0:
                            st.success(f"✅ {selected_h5} successfully converted.")
                            json_file = selected_h5.replace(".h5", ".json")
                            st.info(f"Generated file: `{json_file}`")

                            # Provide download button for the newly created file
                            if Path(json_file).exists():
                                with open(json_file, "rb") as f:
                                    st.download_button(
                                        label="📥 Download JSON",
                                        data=f,
                                        file_name=json_file,
                                        mime="application/json"
                                    )
                        else:
                            st.error(f"❌ Conversion error: {res.stderr}")
                    except Exception as e:
                        st.error(f"Error: {e}")

    st.divider()

    st.subheader("📂 Manage JSON Recordings")
    json_files = glob.glob("*.json")
    json_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)

    if not json_files:
        st.info("No JSON files found. Convert some H5 files first.")
    else:
        for jf in json_files:
            col1, col2 = st.columns([3, 1])
            col1.write(f"📄 `{jf}`")
            with open(jf, "rb") as f:
                col2.download_button(
                    label="Download",
                    data=f,
                    file_name=jf,
                    mime="application/json",
                    key=f"dl_{jf}"
                )
    st.markdown("### 📜 Metriplectic Rules Compliance")
    st.info("""
    **Rule 1**: Conservation (H) vs Dissipation (S) | **Rule 2**: Structured Background ($O_n$) | **Rule 3**: Code as Theory
    """)
