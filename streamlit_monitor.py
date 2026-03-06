import streamlit as st
import numpy as np
import time
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuoreMind DIT Monitor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── DIT Constants ─────────────────────────────────────────────────────────────
PHI = (1 + np.sqrt(5)) / 2
DRIFT_072 = 7.0 - (2 * np.pi)

# ─── Sidebar Controls ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 QuoreMind DIT Monitor")
    st.markdown("**Metriplectic Engine v1.0**")
    st.divider()

    st.markdown("### ⚙️ Simulation Parameters")
    total_cycles  = st.slider("Total Cycles",      min_value=100, max_value=2000, value=400, step=50)
    threshold     = st.slider("Phase Threshold",   min_value=0.5, max_value=5.0,  value=2.0, step=0.1)
    noise_sigma   = st.slider("Noise σ",           min_value=0.0, max_value=0.5,  value=0.05, step=0.01)
    sim_speed     = st.slider("Update Interval (cycles)", min_value=5, max_value=50, value=10, step=5)
    st.divider()

    st.markdown("### ℹ️ DIT Constants")
    st.metric("φ (Golden Ratio)", f"{PHI:.6f}")
    st.metric("DRIFT_072", f"{DRIFT_072:.6f}")
    st.metric("THRESHOLD", f"{threshold:.2f}")

    run_sim = st.button("▶ Run Simulation", type="primary", use_container_width=True)
    stop_sim = st.button("⏹ Stop", use_container_width=True)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0d0d0d; }
    .stMetric label { color: #00f2ff !important; font-weight: bold; }
    .stMetric div[data-testid="stMetricValue"] { color: #00ff41 !important; font-size: 1.8rem !important; }
    div[data-testid="stMetricDelta"] { font-size: 0.9rem; }
    .block-container { padding-top: 1rem; }
    h1 { color: #00f2ff !important; text-shadow: 0 0 10px #00f2ff44; }
    .stSidebar { background-color: #111827; }
</style>
""", unsafe_allow_html=True)

# ─── Header ────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 QuoreMind DIT Phase Stability Monitor")
st.markdown("*Metriplectic Closed-Loop | $d_{symp} = O_n$ (Áureo) | $d_{metr}$ = DRIFT_072 correction*")
st.divider()

# ─── Metric Row ────────────────────────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)
metric_cycle       = col1.metric("Cycle",            "—")
metric_stability   = col2.metric("Stability Index",  "—")
metric_outliers    = col3.metric("Outliers",         "—")
metric_corrections = col4.metric("Corrections",      "—")
metric_accumulator = col5.metric("Accumulator",      "—")

st.divider()

# ─── Charts ────────────────────────────────────────────────────────────────────
chart_col, stats_col = st.columns([3, 1])
with chart_col:
    phase_chart = st.empty()
with stats_col:
    st.markdown("#### 📊 Live Stats")
    stats_table = st.empty()
    lagrangian_card = st.empty()

# ─── Simulation Core ───────────────────────────────────────────────────────────
def compute_lagrangian(i, accumulator, threshold):
    L_symp = np.cos(np.pi * i) * np.cos(np.pi * PHI * i)
    L_metr = -abs(accumulator) * (DRIFT_072 / threshold)
    return L_symp, L_metr

def run_simulation(total_cycles, threshold, noise_sigma, sim_speed):
    accumulator = 0.0
    history     = []
    outliers    = []
    corrections = 0
    L_symp_hist = []
    L_metr_hist = []

    for i in range(1, total_cycles + 1):
        L_symp, L_metr = compute_lagrangian(i, accumulator, threshold)
        L_symp_hist.append(L_symp)
        L_metr_hist.append(L_metr)

        accumulator += L_symp + np.random.normal(0, noise_sigma)
        is_outlier = abs(accumulator) > threshold

        if is_outlier or (i % 7 == 0):
            accumulator *= (DRIFT_072 / threshold)
            corrections += 1
            if is_outlier:
                outliers.append((i, accumulator))

        history.append(accumulator)
        stability_index = 100 * (1 - (len(outliers) / i))

        # ── Update UI every sim_speed cycles ──
        if i % sim_speed == 0 or i == total_cycles:
            # ── Plotly Chart ──
            fig = make_subplots(
                rows=2, cols=1, shared_xaxes=True,
                row_heights=[0.65, 0.35],
                subplot_titles=("Phase Accumulator (Laminar Flow)", "Lagrangian Components L_symp vs L_metr")
            )

            x = list(range(len(history)))

            # Phase line
            fig.add_trace(go.Scatter(
                x=x, y=history,
                name="Accumulator",
                line=dict(color="#00f2ff", width=1.5),
                fill='tozeroy', fillcolor='rgba(0,242,255,0.07)'
            ), row=1, col=1)

            # Thresholds
            fig.add_hline(y=threshold,  line_dash="dash", line_color="#ff3e3e", opacity=0.5, row=1, col=1)
            fig.add_hline(y=-threshold, line_dash="dash", line_color="#ff3e3e", opacity=0.5, row=1, col=1)

            # Outlier markers
            if outliers:
                ox, oy = zip(*outliers)
                fig.add_trace(go.Scatter(
                    x=list(ox), y=list(oy), mode='markers',
                    name="Outliers",
                    marker=dict(color="#ff3e3e", size=7, symbol="x")
                ), row=1, col=1)

            # Lagrangian
            fig.add_trace(go.Scatter(
                x=x, y=L_symp_hist, name="L_symp (conservative)",
                line=dict(color="#00ff41", width=1.2)
            ), row=2, col=1)
            fig.add_trace(go.Scatter(
                x=x, y=L_metr_hist, name="L_metr (dissipative)",
                line=dict(color="#f39c12", width=1.2)
            ), row=2, col=1)

            fig.update_layout(
                height=520,
                paper_bgcolor="#0d0d0d",
                plot_bgcolor="#111827",
                font=dict(color="#cce6ff"),
                legend=dict(orientation="h", y=-0.08, font=dict(size=10)),
                margin=dict(l=30, r=10, t=40, b=30),
            )
            fig.update_xaxes(showgrid=True, gridcolor="#1e293b")
            fig.update_yaxes(showgrid=True, gridcolor="#1e293b")

            phase_chart.plotly_chart(fig, use_container_width=True)

            # ── Metrics ──
            si_color = "normal" if stability_index >= 99 else "inverse"
            metric_cycle.metric(       "Cycle",           f"{i} / {total_cycles}")
            metric_stability.metric(   "Stability Index", f"{stability_index:.2f}%",
                                       delta=f"{stability_index - 100:.2f}%")
            metric_outliers.metric(    "Outliers",        len(outliers))
            metric_corrections.metric( "Corrections",     corrections)
            metric_accumulator.metric( "Accumulator",     f"{accumulator:.4f}")

            # ── Stats Table ──
            arr = np.array(history)
            df_stats = pd.DataFrame({
                "Metric": ["Mean", "Std Dev", "Min", "Max", "Last"],
                "Value":  [
                    f"{arr.mean():.4f}",
                    f"{arr.std():.4f}",
                    f"{arr.min():.4f}",
                    f"{arr.max():.4f}",
                    f"{arr[-1]:.4f}",
                ]
            })
            stats_table.dataframe(df_stats, hide_index=True, use_container_width=True)

            # ── Lagrangian Card ──
            lagrangian_card.markdown(f"""
**Lagrangian (cycle {i})**
| Component | Value |
|-----------|-------|
| `L_symp` (conservative) | `{L_symp:.6f}` |
| `L_metr` (dissipative)  | `{L_metr:.6f}` |
| **Ratio** |  `{abs(L_symp / L_metr) if L_metr != 0 else '∞':.4f}` |
""")

            time.sleep(0.02)

    st.success(f"✅ Simulation complete! Final stability: {stability_index:.4f}% | Outliers: {len(outliers)} | Corrections: {corrections}")

# ─── Run ───────────────────────────────────────────────────────────────────────
if run_sim:
    run_simulation(total_cycles, threshold, noise_sigma, sim_speed)
else:
    phase_chart.info("👈 Configure parameters in the sidebar and click **▶ Run Simulation** to start.")
