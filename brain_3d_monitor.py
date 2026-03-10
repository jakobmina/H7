import streamlit as st
import numpy as np
import plotly.graph_objects as go
from brain_topology import generate_brain_topology
from nodes_network import HierarchicalNode, Node

# --- Page Config ---
st.set_page_config(page_title="3D Brain Node Mapper", page_icon="🧠", layout="wide")

# --- Styling ---
st.markdown("""
<style>
    .main { background-color: #0d0d0d; color: #00f2ff; }
    h1, h2, h3 { color: #00f2ff !important; text-shadow: 0 0 10px #00f2ff44; }
    .stSidebar { background-color: #111827; }
</style>
""", unsafe_allow_html=True)

st.title("🧠 3D Hierarchical Brain Node Mapper")
st.markdown("### Metriplectic Mandate: Control Mapping in 3D Topology")

# --- State Management ---
if 'brain' not in st.session_state:
    st.session_state.brain = generate_brain_topology(n_clusters=12, neurons_per_cluster=30)

# --- Sidebar Controls ---
with st.sidebar:
    st.header("🎛️ Visual Control")
    lod_level = st.slider("Visual Depth (LOD)", 0, 2, 1)
    st.divider()
    
    st.header("🔬 Network Stats")
    total_l_symp, total_l_metr = st.session_state.brain.compute_lagrangian()
    st.metric("Total L_symp (H)", f"{total_l_symp:.2f}")
    st.metric("Total L_metr (S)", f"{total_l_metr:.2f}")
    st.divider()
    
    if st.button("♻️ Regenerate Topology"):
        st.session_state.brain = generate_brain_topology(n_clusters=12, neurons_per_cluster=30)
        st.rerun()

# --- 3D Visualization Logic ---
def get_nodes_at_level(root, target_level):
    nodes = []
    
    def traverse(node):
        if node.level == target_level:
            nodes.append(node)
        elif node.level < target_level:
            if isinstance(node, HierarchicalNode):
                for child in node.children:
                    traverse(child)
    
    traverse(root)
    return nodes

visible_nodes = get_nodes_at_level(st.session_state.brain, lod_level)

# Prepare Data for Plotly
x, y, z = [], [], []
stability = []
text_labels = []

for node in visible_nodes:
    pos = node.position
    x.append(pos[0])
    y.append(pos[1])
    z.append(pos[2])
    
    ls, lm = node.compute_lagrangian()
    stab = 100 * (1.0 - abs(lm / (abs(ls) + 1e-6)))
    stability.append(stab)
    
    text_labels.append(
        f"Level: {node.level}<br>"
        f"L_symp: {ls:.4f}<br>"
        f"L_metr: {lm:.4f}<br>"
        f"Stability: {stab:.2f}%"
    )

# --- 3D Plotly Figure ---
fig = go.Figure(data=[go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    marker=dict(
        size=12 if lod_level == 0 else (8 if lod_level == 1 else 4),
        color=stability,
        colorscale='Viridis',
        colorbar=dict(title="Stability Index"),
        opacity=0.8,
        line=dict(width=0.5, color='white')
    ),
    text=text_labels,
    hoverinfo='text'
)])

fig.update_layout(
    scene=dict(
        xaxis=dict(backgroundcolor="rgb(0, 0, 0)", gridcolor="gray", showbackground=True, zerolinecolor="gray"),
        yaxis=dict(backgroundcolor="rgb(0, 0, 0)", gridcolor="gray", showbackground=True, zerolinecolor="gray"),
        zaxis=dict(backgroundcolor="rgb(0, 0, 0)", gridcolor="gray", showbackground=True, zerolinecolor="gray"),
    ),
    paper_bgcolor="#0d0d0d",
    margin=dict(r=10, l=10, b=10, t=10),
    height=700
)

st.plotly_chart(fig, width='stretch')

# --- Details Panel ---
st.divider()
c1, c2 = st.columns(2)

with c1:
    st.markdown("#### Topology Details")
    st.write(f"Showing **{len(visible_nodes)}** nodes at Hierarchy Level **{lod_level}**.")
    if lod_level == 0:
        st.info("System root representing the global network state.")
    elif lod_level == 1:
        st.info("Functional clusters distributed across the brain volume.")
    else:
        st.info("Local neuron-nodes providing high-granularity control.")

with c2:
    st.markdown("#### Metriplectic Competition")
    # Show histogram of L_symp vs L_metr
    ls_vals = [n.compute_lagrangian()[0] for n in visible_nodes]
    lm_vals = [n.compute_lagrangian()[1] for n in visible_nodes]
    
    fig_hist = go.Figure()
    fig_hist.add_trace(go.Histogram(x=ls_vals, name='L_symp (H)', marker_color='#00ff41', opacity=0.75))
    fig_hist.add_trace(go.Histogram(x=lm_vals, name='L_metr (S)', marker_color='#ff3e3e', opacity=0.75))
    
    fig_hist.update_layout(
        barmode='overlay', 
        template='plotly_dark', 
        height=300,
        margin=dict(l=0, r=0, t=30, b=0),
        title="Lagrangian Distribution"
    )
    st.plotly_chart(fig_hist, width='stretch')
