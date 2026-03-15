"""
h7_api.py — FastAPI Backend for H7 Live Monitor
Exposes MetriplexOracle and H7Conservation data as REST endpoints.
CORS enabled for React dev server on localhost:5173
"""
import sys, math, time
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import hashlib, numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from smopsys.h7_quantum_oracle import MetriplexOracle, MetriplexConfig, H7Conservation

app = FastAPI(title="H7 Live Monitor API", version="1.0")

# ── CORS: allow React dev server ────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ── Shared oracle instances ─────────────────────────────────────────────────────
PHI = (1 + math.sqrt(5)) / 2
PI  = math.pi
oracle = MetriplexOracle(MetriplexConfig())

STATES = ["G[1-6]", "G[2-5]", "G[3-4]", "G[0-7]"]
VIOLATIONS = [
    "uud ↑↑↓ (1,6)",     # par protónico — producto ternario = +1
    "ddu ↓↓↑ (6,1)",     # inverso — producto ternario = +1
    "udu ↑↓↑ (2,5)",     # box-in-box interno
    "dud ↓↑↓ (5,2)",     # box-in-box externo
    "∅  ↓∅   (3,4)·(4,3)", # producto = 0 — estado anulado
]
N_VALS     = [1, 2, 3, 4, 5, 6]
BIN_A      = ["001","010","011","100","101","110"]
BIN_B      = ["110","101","100","011","010","001"]


def _md5(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()

def _hex_mod6(hex_str: str) -> int:
    r = 0
    for ch in hex_str:
        r = (r * 16 + int(ch, 16)) % 6
    return r

def _psi(n: int) -> float:
    return math.cos(PI * PHI * n)

def _ternary(v: float, eps: float = 0.25) -> int:
    if v > eps:  return  1
    if v < -eps: return -1
    return 0

def _h7_classify(key: str) -> dict:
    n = N_VALS[_hex_mod6(_md5(key))]
    p = _psi(n)
    t = _ternary(p)
    label = {1: "constructive", 0: "equilibrium", -1: "destructive"}[t]
    return {
        "key":    key,
        "n":      n,
        "sv":     f"({n},{7-n})",
        "psi":    round(p, 6),
        "t":      t,
        "fw":     BIN_A[n - 1],
        "bw":     BIN_B[n - 1],
        "label":  label,
    }


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/api/h7/matrix")
def get_matrix():
    """Full 4×5 H7 classification matrix, computed server-side."""
    rows = []
    for st in STATES:
        for viol in VIOLATIONS:
            entry = _h7_classify(f"{st}::{viol}")
            entry["state"]     = st
            entry["violation"] = viol
            rows.append(entry)
    summary = {lbl: sum(1 for r in rows if r["label"] == lbl)
               for lbl in ("constructive", "equilibrium", "destructive")}
    return {"rows": rows, "summary": summary, "source": "python-backend"}


@app.get("/api/h7/oracle")
def get_oracle():
    """MetriplexOracle info + Lagrangian per momentum state."""
    info = oracle.get_oracle_info()
    lagrangians = {}
    for p in range(1, 7):
        L_symp, L_metr = oracle.compute_lagrangian(p)
        group, vec, energy = oracle.forward(p)
        lagrangians[p] = {
            "L_symp":  round(L_symp, 6),
            "L_metr":  round(L_metr, 6),
            "energy":  round(energy, 6),
            "group":   group,
            "psi_n":   round(_psi(p), 6),
        }
    pairing = H7Conservation.pairing_table()
    return {
        "momentum_range":   info["momentum_range"],
        "hidden_symmetry":  info["symmetry_string"],
        "collision_groups": info["collision_groups"],
        "energy_profile":   info["energy_profile"],
        "lagrangians":      lagrangians,
        "h7_pairing":       {str(k): v for k, v in pairing.items()},
    }


@app.get("/api/h7/status")
def get_status():
    """Real-time Lagrangian tick evaluated at current time."""
    t = time.time() % (2 * PI)   # wrap to [0, 2π]
    L_symp = math.sin(2 * PI * t) + math.sin(2 * PI * PHI * t) / PHI
    L_metr = -abs(L_symp) * 0.1
    O_n    = math.cos(PI * PHI)
    stability = max(0.0, 1.0 - abs(L_metr))
    return {
        "tick":       round(t, 4),
        "L_symp":     round(L_symp, 6),
        "L_metr":     round(L_metr, 6),
        "O_n":        round(O_n, 6),
        "stability":  round(stability, 4),
        "phi":        round(PHI, 6),
    }

@app.get("/api/h7/network")
def get_network(n_neurons: int = 88, seed: int = 42):
    """
    Generate a seeded H7 neuron network.
    Each neuron is classified by H7. Edges link collision-group partners.
    n_neurons: number of neurons (default 88, reference to ~88B brain neurons).
    """
    import random
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    # ── Build nodes ──────────────────────────────────────────────────────────
    nodes = []
    for i in range(n_neurons):
        key    = f"neuron::{i}::{seed}"
        n_val  = N_VALS[_hex_mod6(_md5(key))]
        p      = _psi(n_val)
        t      = _ternary(p)
        label  = {1: "constructive", 0: "equilibrium", -1: "destructive"}[t]
        L_symp, L_metr = oracle.compute_lagrangian(n_val)
        group, _, energy = oracle.forward(n_val)

        # 3-D position: sphere shell with golden angle spacing
        phi_angle  = i * 2.399963  # golden angle in radians
        theta_angle = math.acos(1 - 2 * (i + 0.5) / n_neurons)
        r = 8.0
        nodes.append({
            "id":     i,
            "key":    key,
            "n":      n_val,
            "label":  label,
            "group":  group,
            "L_symp": round(L_symp, 4),
            "L_metr": round(L_metr, 4),
            "energy": round(energy, 4),
            "psi":    round(p, 4),
            "x": round(r * math.sin(theta_angle) * math.cos(phi_angle), 3),
            "y": round(r * math.sin(theta_angle) * math.sin(phi_angle), 3),
            "z": round(r * math.cos(theta_angle), 3),
        })

    # ── Build edges: connect collision-group partners ────────────────────────
    group_map: dict[str, list[int]] = {}
    for node in nodes:
        group_map.setdefault(node["group"], []).append(node["id"])

    edges = []
    seen  = set()
    for group_nodes in group_map.values():
        # Connect each node to up to 3 partners in the same group
        for nid in group_nodes:
            candidates = [x for x in group_nodes if x != nid]
            partners   = rng.sample(candidates, min(3, len(candidates)))
            for pid in partners:
                key_e = (min(nid, pid), max(nid, pid))
                if key_e not in seen:
                    seen.add(key_e)
                    src = nodes[nid]; dst = nodes[pid]
                    w   = round(abs(src["L_symp"] + dst["L_symp"]) / 2, 4)
                    edges.append({"src": nid, "dst": pid, "weight": w,
                                  "type": "excitatory" if w > 0.1 else "inhibitory"})

    # ── Summary ──────────────────────────────────────────────────────────────
    counts = {"constructive": 0, "equilibrium": 0, "destructive": 0}
    for nd in nodes:
        counts[nd["label"]] += 1

    return {
        "n_neurons": len(nodes),
        "n_edges":   len(edges),
        "nodes":     nodes,
        "edges":     edges,
        "summary":   counts,
    }


    import uvicorn
    uvicorn.run("h7_api:app", host="0.0.0.0", port=8000, reload=True)
