import streamlit as st
import numpy as np
import time
import pandas as pd
import sys
import glob
import subprocess
import os
import json
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import Counter

# Adjust path to allow imports from smopsys and dashboard
root_path = Path(__file__).parent.parent.absolute()
if str(root_path) not in sys.path:
    sys.path.append(str(root_path))

from dashboard.cl1_db import CL1Reader
from dashboard.nodes_network import Node, HierarchicalNode
from dashboard.brain_topology import generate_brain_topology
from dashboard.quoremind_monitor import QuoreMindMonitor

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="H7 Metriplectic Unified Dashboard",
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
st.markdown(f"""
<style>
    .main {{ background-color: #0d0d0d; color: #e0e0e0; }}
    .stMetric label {{ color: {DIT_CYAN} !important; font-weight: bold; }}
    .stMetric div[data-testid="stMetricValue"] {{ color: {DIT_GREEN} !important; font-size: 1.6rem !important; }}
    h1, h2, h3 {{ color: {DIT_CYAN} !important; text-shadow: 0 0 8px {DIT_CYAN}44; }}
    .stSidebar {{ background-color: #111827; }}
</style>
""", unsafe_allow_html=True)

# ─── Global State ──────────────────────────────────────────────────────────────
if 'brain' not in st.session_state:
    st.session_state.brain = generate_brain_topology(n_clusters=12, neurons_per_cluster=30)

if 'engine' not in st.session_state:
    st.session_state.engine = QuoreMindMonitor()
    st.session_state.engine_cycle = 0

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧠 Unified Control")
    st.markdown("**QuoreMindHP + H7 Loop**")
    st.divider()
    
    selected_tab = st.radio("Navigate to:", ["📊 Neural Monitor", "🕸️ 3D Topology", "⚡ Engine Status", "🧪 Lab Utilities"])
    
    st.divider()
    st.header("🎛️ Node Control")
    nn1 = st.slider("NN1 (Phase/Inertia)", -1.0, 1.0, 0.5, step=0.01)
    nn0 = st.slider("NN0 (State/Base)",  -1.0, 1.0, 0.0, step=0.01)
    nn_1 = st.slider("NN-1 (Dissipation)", -1.0, 1.0, -0.5, step=0.01)
    
    n_val = (time.time() * 10) % 1000
    temp_node = Node(nn1, nn0, nn_1, n=n_val)
    l_s, l_m = temp_node.compute_lagrangian()
    
    c1, c2 = st.columns(2)
    c1.metric("L_symp (H)", f"{l_s:.4f}")
    c2.metric("L_metr (S)", f"{l_m:.4f}")

    st.divider()
    lod_level = st.slider("LOD View Layer", 0, 2, 1)
    st.caption("Lower LOD = Higher abstraction")

# ─── Main Content ─────────────────────────────────────────────────────────────

if selected_tab == "📊 Neural Monitor":
    st.header("📊 Real-Time Neural Activity")
    
    mode = st.radio("Source", ["Hardware Live", "DIT Simulation"], horizontal=True)
    
    def sim_dit_ticks(n: int) -> pd.DataFrame:
        data = []
        acc = 0.0
        for i in range(1, n + 1):
            innovation = np.cos(np.pi * i) * np.cos(np.pi * PHI * i)
            acc += innovation + np.random.normal(0, 0.05)
            if abs(acc) > 2.0 or i % 7 == 0:
                acc *= DRIFT_072 / 2.0
            data.append({
                "loop_dur_us": 1000 + np.random.normal(0, 40),
                "spike_count": int(max(0, np.random.poisson(3))),
                "stim_fired": 1 if np.random.random() < 0.04 else 0,
                "l_symp": innovation,
                "l_metr": -abs(acc) * (DRIFT_072 / 2.0),
            })
        return pd.DataFrame(data)

    if mode == "DIT Simulation":
        total_cycles = st.slider("Samples", 100, 1000, 500)
        if st.button("Run Simulation Burst"):
            df = sim_dit_ticks(total_cycles)
            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("Spike Density", "Lagrangian Competition"))
            x_range = list(range(len(df)))
            fig.add_trace(go.Bar(x=x_range, y=df["spike_count"], name="Spikes", marker_color=DIT_CYAN), row=1, col=1)
            fig.add_trace(go.Scatter(x=x_range, y=df["l_symp"], name="L_symp (H)", line=dict(color=DIT_GREEN)), row=2, col=1)
            fig.add_trace(go.Scatter(x=x_range, y=df["l_metr"], name="L_metr (S)", line=dict(color=DIT_RED)), row=2, col=1)
            fig.update_layout(height=500, template="plotly_dark", paper_bgcolor="#0d0d0d")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Searching for `cl1_session.sqlite`...")
        reader = CL1Reader()
        ticks = reader.get_recent_ticks(500)
        if ticks:
            df = pd.DataFrame(ticks)
            st.write(df.tail())
        else:
            st.warning("No hardware session found. Try Simulation mode.")

elif selected_tab == "🕸️ 3D Topology":
    st.header("🕸️ 3D Hierarchical Brain Topology")
    
    def get_nodes_at_level(root, target_level):
        nodes = []
        def traverse(node):
            if node.level == target_level: nodes.append(node)
            elif node.level < target_level and isinstance(node, HierarchicalNode):
                for child in node.children: traverse(child)
        traverse(root)
        return nodes

    visible_nodes = get_nodes_at_level(st.session_state.brain, lod_level)
    
    # Prepare Data for Plotly
    x, y, z, stability, labels = [], [], [], [], []
    for node in visible_nodes:
        pos = node.position
        # Sanity check for NaNs
        if np.any(np.isnan(pos)): continue
        
        ls, lm = node.compute_lagrangian()
        stab = 100 * (1.0 - abs(lm / (abs(ls) + 1e-6)))
        
        x.append(pos[0])
        y.append(pos[1])
        z.append(pos[2])
        stability.append(stab)
        labels.append(f"Lvl: {node.level}<br>Stab: {stab:.2f}%<br>L_symp: {ls:.4f}<br>L_metr: {lm:.4f}")

    # --- 3D Plotly Figure ---
    fig = go.Figure(data=[go.Scatter3d(
        x=x, y=y, z=z, mode='markers',
        marker=dict(
            size=12 if lod_level == 0 else (8 if lod_level == 1 else 4),
            color=stability, 
            colorscale='Viridis', 
            opacity=0.8,
            colorbar=dict(title="Stability Index"),
            line=dict(width=0.5, color='white')
        ),
        text=labels, hoverinfo='text'
    )])
    
    fig.update_layout(
        scene=dict(
            xaxis=dict(backgroundcolor="rgb(0, 0, 0)", gridcolor="gray", showbackground=True, zerolinecolor="gray"),
            yaxis=dict(backgroundcolor="rgb(0, 0, 0)", gridcolor="gray", showbackground=True, zerolinecolor="gray"),
            zaxis=dict(backgroundcolor="rgb(0, 0, 0)", gridcolor="gray", showbackground=True, zerolinecolor="gray"),
        ),
        paper_bgcolor="#0d0d0d", 
        height=700, 
        margin=dict(l=0,r=0,b=0,t=0)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    if st.button("♻️ Regenerate Topology"):
        st.session_state.brain = generate_brain_topology(n_clusters=12, neurons_per_cluster=30)
        st.rerun()

elif selected_tab == "⚡ Engine Status":
    st.header("⚡ QuoreMind Engine Integrity")
    
    engine = st.session_state.engine
    
    # Simulate single cycle to show movement
    st.session_state.engine_cycle += 1
    engine.simulate_cycle(st.session_state.engine_cycle)
    
    c1, c2, c3 = st.columns(3)
    c1.metric("Accumulator", f"{engine.accumulator:.4f}")
    c2.metric("Integrity Index", f"{engine.stability_index:.2f}%")
    c3.metric("Corrections", engine.corrections)
    
    # History plot
    hist_df = pd.DataFrame({"Phase Accumulator": engine.history})
    st.line_chart(hist_df, color=DIT_CYAN)
    
    st.divider()
    st.info("Engine is running in dynamic background mode. Every refresh simulates a new cycle.")
    if st.button("Reset Engine"):
        st.session_state.engine = QuoreMindMonitor()
        st.session_state.engine_cycle = 0
        st.rerun()

elif selected_tab == "🧪 Lab Utilities":
    st.header("🧪 Lab Analysis Utilities")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("H5 Converter")
        h5_files = glob.glob(str(root_path / "*.h5"))
        if h5_files:
            target = st.selectbox("Recordings", h5_files)
            if st.button("Convert to JSON"):
                st.write(f"Converting {target}...")
                # Mock call to converter logic
                time.sleep(1)
                st.success("Conversion complete (Simulated)")
        else:
            st.warning("No recordings found.")
            
    with col2:
        st.subheader("Metriplectic Mandate")
        st.markdown("""
        - **Rule 1**: Orthogonal Brackets {u,H} and [u,S]
        - **Rule 2**: Golden Operator $O_n$ modulation
        - **Rule 3**: Code implements Theory
        """)
        image_path = root_path / "h7_ionic_diagnostic.png"
        if image_path.exists():
            st.image(str(image_path), caption="Last Physical Diagnostic")
        else:
            st.info("No physical diagnostic image found yet.")

st.divider()
st.caption(f"H7 Unified Hub | Local Time: {time.strftime('%H:%M:%S')} | PHI Coherence: {PHI:.4f}")
