/**
 * NeuronNet.jsx — 3D H7 Neuron Network Visualizer
 * Each neuron is a sphere colored by its H7 state (constructive/equilibrium/destructive).
 * Synaptic edges link neurons within the same MetriplexOracle collision group.
 * Pulls data from /api/h7/network every 15 seconds.
 */
import { useRef, useState, useEffect, useMemo } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import { OrbitControls, Html } from "@react-three/drei";
import * as THREE from "three";

// ── Colors by H7 state ────────────────────────────────────────────────────────
const STATE_COLOR = {
  constructive: "#4ade80",
  equilibrium:  "#fde047",
  destructive:  "#f87171",
};
const STATE_EMISSIVE = {
  constructive: "#166534",
  equilibrium:  "#713f12",
  destructive:  "#7f1d1d",
};

// ── Individual neuron sphere ──────────────────────────────────────────────────
function Neuron({ node, onClick, selected }) {
  const meshRef = useRef();
  const [hovered, setHovered] = useState(false);
  const color   = STATE_COLOR[node.label]   || "#ffffff";
  const emissive = STATE_EMISSIVE[node.label] || "#000000";

  // Pulse animation for constructive neurons
  useFrame((_, delta) => {
    if (!meshRef.current) return;
    if (node.label === "constructive") {
      const t = Date.now() * 0.002 + node.id * 0.4;
      const scale = 1.0 + 0.12 * Math.sin(t);
      meshRef.current.scale.setScalar(scale);
    }
    if (node.label === "destructive") {
      const t = Date.now() * 0.0008 + node.id * 0.2;
      const scale = 1.0 - 0.08 * Math.abs(Math.sin(t));
      meshRef.current.scale.setScalar(scale);
    }
  });

  const radius = selected ? 0.28 : hovered ? 0.22 : 0.16;

  return (
    <mesh
      ref={meshRef}
      position={[node.x, node.y, node.z]}
      onClick={(e) => { e.stopPropagation(); onClick(node); }}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
    >
      <sphereGeometry args={[radius, 12, 12]} />
      <meshStandardMaterial
        color={color}
        emissive={emissive}
        emissiveIntensity={selected ? 1.2 : hovered ? 0.7 : 0.3}
        roughness={0.4}
        metalness={0.3}
      />
    </mesh>
  );
}

// ── Synaptic edge (line between two neurons) ──────────────────────────────────
function SynapticEdge({ src, dst, type }) {
  const points = useMemo(() => [
    new THREE.Vector3(src.x, src.y, src.z),
    new THREE.Vector3(dst.x, dst.y, dst.z),
  ], [src, dst]);

  const geometry = useMemo(() => {
    const g = new THREE.BufferGeometry().setFromPoints(points);
    return g;
  }, [points]);

  const color   = type === "excitatory" ? "#34d39940" : "#f8717130";
  const opacity = type === "excitatory" ? 0.35 : 0.15;

  return (
    <line geometry={geometry}>
      <lineBasicMaterial color={color} transparent opacity={opacity} />
    </line>
  );
}

// ── Info tooltip for selected neuron ──────────────────────────────────────────
function NeuronTooltip({ node }) {
  if (!node) return null;
  const color = STATE_COLOR[node.label];
  return (
    <Html position={[node.x + 0.4, node.y + 0.4, node.z]} style={{pointerEvents:"none"}}>
      <div style={{
        background:"#0d1117", border:`1px solid ${color}`, borderRadius:"8px",
        padding:"8px 12px", fontSize:"11px", color:"#e6edf3",
        whiteSpace:"nowrap", fontFamily:"'Courier New',monospace",
        boxShadow:`0 0 12px ${color}44`
      }}>
        <div style={{color, fontWeight:"bold", marginBottom:"4px"}}>
          Neuron #{node.id}  n={node.n}
        </div>
        <div>Group: <span style={{color:"#58a6ff"}}>{node.group}</span></div>
        <div>Ψₙ: <span style={{color}}>{node.psi}</span></div>
        <div>L_symp: <span style={{color:"#4ade80"}}>{node.L_symp}</span></div>
        <div>L_metr: <span style={{color:"#f59e0b"}}>{node.L_metr}</span></div>
        <div style={{marginTop:"4px",fontSize:"10px",color:"#484f58"}}>{node.label}</div>
      </div>
    </Html>
  );
}

// ── Scene ─────────────────────────────────────────────────────────────────────
function Scene({ data, selected, onSelect }) {
  const nodeMap = useMemo(() => {
    if (!data) return {};
    return Object.fromEntries(data.nodes.map(n => [n.id, n]));
  }, [data]);

  if (!data) return null;

  return (
    <>
      <ambientLight intensity={0.4} />
      <pointLight position={[10, 10, 10]} intensity={1.2} />
      <pointLight position={[-10, -10, -10]} intensity={0.4} color="#4ade80" />

      {/* Edges first (drawn behind nodes) */}
      {data.edges.map((e, i) => {
        const src = nodeMap[e.src]; const dst = nodeMap[e.dst];
        if (!src || !dst) return null;
        return <SynapticEdge key={i} src={src} dst={dst} type={e.type} />;
      })}

      {/* Neurons */}
      {data.nodes.map(node => (
        <Neuron
          key={node.id}
          node={node}
          selected={selected?.id === node.id}
          onClick={onSelect}
        />
      ))}

      {/* Tooltip */}
      <NeuronTooltip node={selected} />

      <OrbitControls enablePan enableZoom autoRotate autoRotateSpeed={0.4} />
    </>
  );
}

// ── Main exported component ───────────────────────────────────────────────────
export default function NeuronNet() {
  const [data,     setData]     = useState(null);
  const [selected, setSelected] = useState(null);
  const [loading,  setLoading]  = useState(true);
  const [nCount,   setNCount]   = useState(88);
  const [live,     setLive]     = useState(false);

  async function fetchNetwork(n = nCount) {
    setLoading(true);
    try {
      const r = await fetch(`http://localhost:8000/api/h7/network?n_neurons=${n}`,
        {signal: AbortSignal.timeout(4000)});
      if (!r.ok) throw new Error();
      setData(await r.json());
      setLive(true);
    } catch { setLive(false); }
    setLoading(false);
  }

  useEffect(() => {
    fetchNetwork(nCount);
    const id = setInterval(() => fetchNetwork(nCount), 15000);
    return () => clearInterval(id);
  }, [nCount]);

  const counts = data?.summary || {};

  return (
    <div style={{fontFamily:"'Courier New',monospace"}}>
      {/* Controls bar */}
      <div style={{display:"flex",gap:"8px",alignItems:"center",
        marginBottom:"12px",flexWrap:"wrap"}}>
        {[44, 88, 176].map(n => (
          <button key={n} onClick={() => { setNCount(n); fetchNetwork(n); }} style={{
            padding:"4px 12px",borderRadius:"6px",border:"1px solid",fontSize:"11px",cursor:"pointer",
            background: nCount===n?"#1f6feb":"#161b22",
            borderColor: nCount===n?"#388bfd":"#30363d",
            color: nCount===n?"#fff":"#8b949e"
          }}>{n} neurons</button>
        ))}
        <button onClick={() => fetchNetwork(nCount)} style={{
          padding:"4px 12px",borderRadius:"6px",border:"1px solid #30363d",
          fontSize:"11px",cursor:"pointer",background:"#161b22",color:"#8b949e"
        }}>↻ Refresh</button>

        <span style={{marginLeft:"auto",display:"flex",gap:"6px"}}>
          {Object.entries(counts).map(([lbl, cnt]) => (
            <span key={lbl} style={{
              background:"#161b22",border:`1px solid ${STATE_COLOR[lbl]||"#30363d"}`,
              borderRadius:"4px",padding:"2px 8px",fontSize:"10px",
              color: STATE_COLOR[lbl]||"#8b949e"}}>
              {cnt} {lbl}
            </span>
          ))}
          {data && <span style={{color:"#484f58",fontSize:"10px",alignSelf:"center"}}>
            {data.n_edges} synapses
          </span>}
        </span>
      </div>

      {/* 3D Canvas */}
      <div style={{
        height:"520px", borderRadius:"12px", overflow:"hidden",
        border:"1px solid #21262d", background:"#050810",
        position:"relative"
      }}>
        {loading && (
          <div style={{position:"absolute",inset:0,display:"flex",
            alignItems:"center",justifyContent:"center",zIndex:10,
            color:"#58a6ff",fontSize:"13px",background:"#050810"}}>
            Loading network…
          </div>
        )}
        <Canvas camera={{ position: [0, 0, 18], fov: 60 }} gl={{ antialias: true }}>
          <Scene data={data} selected={selected} onSelect={setSelected} />
        </Canvas>

        {/* Legend overlay */}
        <div style={{position:"absolute",bottom:"12px",left:"12px",
          display:"flex",gap:"8px",pointerEvents:"none"}}>
          {Object.entries(STATE_COLOR).map(([lbl,col]) => (
            <div key={lbl} style={{display:"flex",alignItems:"center",gap:"4px",
              background:"#0d111799",borderRadius:"4px",padding:"2px 7px"}}>
              <div style={{width:8,height:8,borderRadius:"50%",background:col,
                boxShadow:`0 0 6px ${col}`}}/>
              <span style={{fontSize:"10px",color:col}}>{lbl}</span>
            </div>
          ))}
        </div>

        {/* Connection status */}
        <div style={{position:"absolute",top:"10px",right:"12px",fontSize:"10px",
          color:live?"#4ade80":"#f87171",pointerEvents:"none"}}>
          {live ? "● LIVE" : "● OFFLINE"}
        </div>
      </div>

      {selected && (
        <div style={{marginTop:"12px",padding:"10px 14px",
          background:"#0d1117",border:`1px solid ${STATE_COLOR[selected.label]}`,
          borderRadius:"8px",fontSize:"11px",display:"flex",gap:"20px",flexWrap:"wrap"}}>
          <span style={{color:"#8b949e"}}>Selected:</span>
          <span>Neuron <b style={{color:"#58a6ff"}}>#{selected.id}</b></span>
          <span>n=<b style={{color:"#a78bfa"}}>{selected.n}</b></span>
          <span>Group <b style={{color:"#60a5fa"}}>{selected.group}</b></span>
          <span>Ψₙ=<b style={{color:STATE_COLOR[selected.label]}}>{selected.psi}</b></span>
          <span>L_symp=<b style={{color:"#4ade80"}}>{selected.L_symp}</b></span>
          <span>L_metr=<b style={{color:"#f59e0b"}}>{selected.L_metr}</b></span>
          <span style={{color:STATE_COLOR[selected.label],fontWeight:"bold"}}>
            {selected.label}
          </span>
        </div>
      )}

      <div style={{marginTop:"8px",color:"#484f58",fontSize:"10px"}}>
        Drag to orbit · Scroll to zoom · Click a neuron to inspect · Auto-rotates
        · Positions: golden angle spiral · Edges: MetriplexOracle collision groups
      </div>
    </div>
  );
}
