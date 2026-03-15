import { useState, useEffect, useCallback } from "react";
import "./App.css";

// ── Constants ──────────────────────────────────────────────────────────────────
const API = "http://localhost:8000/api/h7";
const PHI = (1 + Math.sqrt(5)) / 2;
const PI  = Math.PI;

// ── Fallback client-side H7 (used when API is offline) ────────────────────────
function md5(s) {
  function rh(n){var j,s="";for(j=0;j<=3;j++)s+="0123456789abcdef".charAt((n>>(j*8+4))&0xF)+"0123456789abcdef".charAt((n>>(j*8))&0xF);return s;}
  function ad(x,y){var l=(x&0xFFFF)+(y&0xFFFF);var m=(x>>16)+(y>>16)+(l>>16);return(m<<16)|(l&0xFFFF);}
  function rl(n,c){return(n<<c)|(n>>>(32-c));}
  function cm(q,a,b,x,s,t){return ad(rl(ad(ad(a,q),ad(x,t)),s),b);}
  function ff(a,b,c,d,x,s,t){return cm((b&c)|((~b)&d),a,b,x,s,t);}
  function gg(a,b,c,d,x,s,t){return cm((b&d)|(c&(~d)),a,b,x,s,t);}
  function hh(a,b,c,d,x,s,t){return cm(b^c^d,a,b,x,s,t);}
  function ii(a,b,c,d,x,s,t){return cm(c^(b|(~d)),a,b,x,s,t);}
  function sb(x){var i;var nb=((x.length+8)>>6)+1;var bl=new Array(nb*16);for(i=0;i<nb*16;i++)bl[i]=0;for(i=0;i<x.length;i++)bl[i>>2]|=x.charCodeAt(i)<<((i%4)*8);bl[i>>2]|=0x80<<((i%4)*8);bl[nb*16-2]=x.length*8;return bl;}
  var i,x=sb(s),a=1732584193,b=-271733879,c=-1732584194,d=271733878,oa,ob,oc,od;
  for(i=0;i<x.length;i+=16){oa=a;ob=b;oc=c;od=d;
    a=ff(a,b,c,d,x[i],7,-680876936);d=ff(d,a,b,c,x[i+1],12,-389564586);c=ff(c,d,a,b,x[i+2],17,606105819);b=ff(b,c,d,a,x[i+3],22,-1044525330);
    a=ff(a,b,c,d,x[i+4],7,-176418897);d=ff(d,a,b,c,x[i+5],12,1200080426);c=ff(c,d,a,b,x[i+6],17,-1473231341);b=ff(b,c,d,a,x[i+7],22,-45705983);
    a=ff(a,b,c,d,x[i+8],7,1770035416);d=ff(d,a,b,c,x[i+9],12,-1958414417);c=ff(c,d,a,b,x[i+10],17,-42063);b=ff(b,c,d,a,x[i+11],22,-1990404162);
    a=ff(a,b,c,d,x[i+12],7,1804603682);d=ff(d,a,b,c,x[i+13],12,-40341101);c=ff(c,d,a,b,x[i+14],17,-1502002290);b=ff(b,c,d,a,x[i+15],22,1236535329);
    a=gg(a,b,c,d,x[i+1],5,-165796510);d=gg(d,a,b,c,x[i+6],9,-1069501632);c=gg(c,d,a,b,x[i+11],14,643717713);b=gg(b,c,d,a,x[i],20,-373897302);
    a=gg(a,b,c,d,x[i+5],5,-701558691);d=gg(d,a,b,c,x[i+10],9,38016083);c=gg(c,d,a,b,x[i+15],14,-660478335);b=gg(b,c,d,a,x[i+4],20,-405537848);
    a=gg(a,b,c,d,x[i+9],5,568446438);d=gg(d,a,b,c,x[i+14],9,-1019803690);c=gg(c,d,a,b,x[i+3],14,-187363961);b=gg(b,c,d,a,x[i+8],20,1163531501);
    a=gg(a,b,c,d,x[i+13],5,-1444681467);d=gg(d,a,b,c,x[i+2],9,-51403784);c=gg(c,d,a,b,x[i+7],14,1735328473);b=gg(b,c,d,a,x[i+12],20,-1926607734);
    a=hh(a,b,c,d,x[i+5],4,-378558);d=hh(d,a,b,c,x[i+8],11,-2022574463);c=hh(c,d,a,b,x[i+11],16,1839030562);b=hh(b,c,d,a,x[i+14],23,-35309556);
    a=hh(a,b,c,d,x[i+1],4,-1530992060);d=hh(d,a,b,c,x[i+4],11,1272893353);c=hh(c,d,a,b,x[i+7],16,-155497632);b=hh(b,c,d,a,x[i+10],23,-1094730640);
    a=hh(a,b,c,d,x[i+13],4,681279174);d=hh(d,a,b,c,x[i],11,-358537222);c=hh(c,d,a,b,x[i+3],16,-722521979);b=hh(b,c,d,a,x[i+6],23,76029189);
    a=hh(a,b,c,d,x[i+9],4,-640364487);d=hh(d,a,b,c,x[i+12],11,-421815835);c=hh(c,d,a,b,x[i+15],16,530742520);b=hh(b,c,d,a,x[i+2],23,-995338651);
    a=ii(a,b,c,d,x[i],6,-198630844);d=ii(d,a,b,c,x[i+7],10,1126891415);c=ii(c,d,a,b,x[i+14],15,-1416354905);b=ii(b,c,d,a,x[i+5],21,-57434055);
    a=ii(a,b,c,d,x[i+12],6,1700485571);d=ii(d,a,b,c,x[i+3],10,-1894986606);c=ii(c,d,a,b,x[i+10],15,-1051523);b=ii(b,c,d,a,x[i+1],21,-2054922799);
    a=ii(a,b,c,d,x[i+8],6,1873313359);d=ii(d,a,b,c,x[i+15],10,-30611744);c=ii(c,d,a,b,x[i+6],15,-1560198380);b=ii(b,c,d,a,x[i+13],21,1309151649);
    a=ii(a,b,c,d,x[i+4],6,-145523070);d=ii(d,a,b,c,x[i+11],10,-1120210379);c=ii(c,d,a,b,x[i+2],15,718787259);b=ii(b,c,d,a,x[i+9],21,-343485551);
    a=ad(a,oa);b=ad(b,ob);c=ad(c,oc);d=ad(d,od);}
  return rh(a)+rh(b)+rh(c)+rh(d);
}
const N_VALS = [1,2,3,4,5,6];
const BIN_A  = ['001','010','011','100','101','110'];
const BIN_B  = ['110','101','100','011','010','001'];
function hexMod6(h){let r=BigInt(0),m=BigInt(6);for(const c of h)r=(r*16n+BigInt(parseInt(c,16)))%m;return Number(r);}
function psi(n){return Math.cos(PI*PHI*n);}
function ternary(v,e=0.25){return v>e?1:v<-e?-1:0;}
function h7Local(key){const n=N_VALS[hexMod6(md5(key))];const p=psi(n);const t=ternary(p);return{n,sv:`(${n},${7-n})`,psi:p.toFixed(6),t,fw:BIN_A[n-1],bw:BIN_B[n-1],label:t===1?"constructive":t===0?"equilibrium":"destructive"};}

const STATES = ["G[1-6]", "G[2-5]", "G[3-4]", "G[0-7]"];
const VIOLATIONS = [
  "uud ↑↑↓ (1,6)",      // par protónico — producto ternario = +1
  "ddu ↓↓↑ (6,1)",      // inverso — producto ternario = +1
  "udu ↑↓↑ (2,5)",      // box-in-box interno
  "dud ↓↑↓ (5,2)",      // box-in-box externo
  "∅ ↓∅ (3,4)·(4,3)", // producto = 0 — estado anulado
];
const N_LIST     = [1,2,3,4,5,6];
const COLORS = {
  constructive:{ bg:"#0d2b1a",border:"#22c55e",text:"#4ade80",badge:"#166534"},
  equilibrium :{ bg:"#1a1a0d",border:"#eab308",text:"#fde047",badge:"#713f12"},
  destructive :{ bg:"#2b0d0d",border:"#ef4444",text:"#f87171",badge:"#7f1d1d"},
};
const SYMBOL = {1:"+1 ▲",0:" 0 ◆","-1":"-1 ▼"};

// ── Local fallback matrix ──────────────────────────────────────────────────────
function buildLocalMatrix() {
  return STATES.flatMap(st => VIOLATIONS.map(viol => ({
    state:st, violation:viol, ...h7Local(`${st}::${viol}`)
  })));
}

// ── Components ─────────────────────────────────────────────────────────────────
function LiveBadge({ live }) {
  return (
    <span style={{display:"inline-flex",alignItems:"center",gap:"5px",
      padding:"2px 10px",borderRadius:"20px",fontSize:"10px",letterSpacing:"1px",
      background: live?"#0d2b1a":"#2b0d0d",
      border:`1px solid ${live?"#22c55e":"#ef4444"}`,
      color: live?"#4ade80":"#f87171"}}>
      <span style={{width:7,height:7,borderRadius:"50%",
        background:live?"#4ade80":"#f87171",
        boxShadow:live?"0 0 6px #4ade80":"none",
        animation:live?"pulse 1.5s infinite":"none"}}/>
      {live ? "LIVE" : "OFFLINE"}
    </span>
  );
}

function StatusBar({ status }) {
  if (!status) return null;
  const ok = status.stability > 0.85;
  return (
    <div style={{display:"flex",flexWrap:"wrap",gap:"8px",padding:"8px 0",marginBottom:"12px",
      borderBottom:"1px solid #21262d"}}>
      {[
        ["L_symp", status.L_symp, status.L_symp >= 0 ? "#4ade80" : "#f87171"],
        ["L_metr", status.L_metr, "#f59e0b"],
        ["O_n",    status.O_n,    "#60a5fa"],
        ["Stab",   (status.stability*100).toFixed(1)+"%", ok?"#4ade80":"#f87171"],
        ["φ",      status.phi,    "#a78bfa"],
      ].map(([k,v,col])=>(
        <div key={k} style={{background:"#0d1117",border:"1px solid #21262d",borderRadius:"6px",
          padding:"4px 10px",fontSize:"11px"}}>
          <span style={{color:"#8b949e"}}>{k} </span>
          <span style={{color:col,fontWeight:"bold"}}>{typeof v==="number"?v.toFixed(4):v}</span>
        </div>
      ))}
    </div>
  );
}

function OraclePanel({ oracle }) {
  if (!oracle) return (
    <div style={{color:"#8b949e",padding:"20px",textAlign:"center"}}>
      Start the Python backend to see Oracle data
    </div>
  );
  return (
    <div>
      <div style={{display:"flex",gap:"16px",flexWrap:"wrap",marginBottom:"16px"}}>
        <Stat label="Hidden Symmetry s" value={oracle.hidden_symmetry} color="#a78bfa"/>
        <Stat label="Energy Profile"    value={oracle.energy_profile}  color="#60a5fa"/>
        <Stat label="Momentum Range"    value={`[${oracle.momentum_range.join(", ")}]`} color="#fde047"/>
      </div>

      <div style={{marginBottom:"16px"}}>
        <div style={{color:"#8b949e",fontSize:"11px",marginBottom:"6px"}}>
          H7 State Pairing (|n⟩ ↔ |7⊕n⟩)
        </div>
        <div style={{display:"flex",gap:"6px",flexWrap:"wrap"}}>
          {Object.entries(oracle.h7_pairing).slice(0,8).map(([a,b])=>(
            <div key={a} style={{background:"#161b22",border:"1px solid #30363d",borderRadius:"6px",
              padding:"4px 10px",fontSize:"11px",color:"#e6edf3"}}>
              |{a}⟩↔|{b}⟩
            </div>
          ))}
        </div>
      </div>

      <div style={{overflowX:"auto"}}>
        <table style={{width:"100%",borderCollapse:"collapse",fontSize:"12px"}}>
          <thead>
            <tr style={{background:"#161b22",borderBottom:"1px solid #30363d"}}>
              {["p","Group","E(p)","L_symp","L_metr","Ψₙ"].map(h=>(
                <th key={h} style={{padding:"8px 10px",textAlign:"left",color:"#58a6ff",
                  fontWeight:"normal",letterSpacing:"1px"}}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {Object.entries(oracle.lagrangians).map(([p,d],i)=>(
              <tr key={p} style={{background:i%2===0?"#0d1117":"#0a0e14",
                borderBottom:"1px solid #21262d"}}>
                <td style={{padding:"7px 10px",color:"#58a6ff",fontWeight:"bold"}}>{p}</td>
                <td style={{padding:"7px 10px",color:"#c9d1d9"}}>{d.group}</td>
                <td style={{padding:"7px 10px",color:"#fde047"}}>{d.energy.toFixed(4)}</td>
                <td style={{padding:"7px 10px",color:"#4ade80"}}>{d.L_symp.toFixed(4)}</td>
                <td style={{padding:"7px 10px",color:"#f59e0b"}}>{d.L_metr.toFixed(4)}</td>
                <td style={{padding:"7px 10px",color:d.psi_n>0?"#4ade80":d.psi_n<0?"#f87171":"#fde047"}}>
                  {d.psi_n.toFixed(4)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Stat({ label, value, color }) {
  return (
    <div style={{background:"#0d1117",border:"1px solid #21262d",borderRadius:"8px",
      padding:"8px 14px",minWidth:"120px"}}>
      <div style={{color:"#8b949e",fontSize:"10px",letterSpacing:"1px"}}>{label}</div>
      <div style={{color:color||"#e6edf3",fontWeight:"bold",fontSize:"14px",marginTop:"2px"}}>{value}</div>
    </div>
  );
}

// ── Main App ───────────────────────────────────────────────────────────────────
export default function App() {
  const [tab,      setTab]      = useState("matrix");
  const [mode,     setMode]     = useState("matrix");
  const [selected, setSelected] = useState(null);
  const [live,     setLive]     = useState(false);
  const [rows,     setRows]     = useState(buildLocalMatrix());
  const [status,   setStatus]   = useState(null);
  const [oracleData, setOracleData] = useState(null);
  const [lastUpdate, setLastUpdate] = useState(null);

  // ── Poll /status every 2 s ─────────────────────────────────────────────────
  useEffect(() => {
    async function fetchStatus() {
      try {
        const r = await fetch(`${API}/status`, {signal: AbortSignal.timeout(1500)});
        if (!r.ok) throw new Error();
        setStatus(await r.json());
        setLive(true);
      } catch { setLive(false); setStatus(null); }
    }
    fetchStatus();
    const id = setInterval(fetchStatus, 2000);
    return () => clearInterval(id);
  }, []);

  // ── Poll /matrix every 5 s ─────────────────────────────────────────────────
  useEffect(() => {
    async function fetchMatrix() {
      try {
        const r = await fetch(`${API}/matrix`, {signal: AbortSignal.timeout(2000)});
        if (!r.ok) throw new Error();
        const data = await r.json();
        setRows(data.rows);
        setLastUpdate(new Date().toLocaleTimeString());
      } catch { setRows(buildLocalMatrix()); }
    }
    fetchMatrix();
    const id = setInterval(fetchMatrix, 5000);
    return () => clearInterval(id);
  }, []);

  // ── Poll /oracle every 10 s ────────────────────────────────────────────────
  useEffect(() => {
    async function fetchOracle() {
      try {
        const r = await fetch(`${API}/oracle`, {signal: AbortSignal.timeout(2000)});
        if (!r.ok) throw new Error();
        setOracleData(await r.json());
      } catch { setOracleData(null); }
    }
    fetchOracle();
    const id = setInterval(fetchOracle, 10000);
    return () => clearInterval(id);
  }, []);

  const displayRows = mode === "top"
    ? STATES.map(st => {
        const stRows = rows.filter(r => r.state === st);
        return stRows.reduce((best, cur) =>
          Math.abs(parseFloat(cur.psi)) > Math.abs(parseFloat(best.psi)) ? cur : best
        );
      })
    : rows;

  return (
    <div style={{background:"#0a0a0f",minHeight:"100vh",fontFamily:"'Courier New',monospace",
      color:"#c9d1d9",padding:"16px"}}>

      {/* ── Header ── */}
      <div style={{borderBottom:"1px solid #21262d",paddingBottom:"12px",marginBottom:"12px",
        display:"flex",alignItems:"flex-start",justifyContent:"space-between",flexWrap:"wrap",gap:"8px"}}>
        <div>
          <div style={{color:"#58a6ff",fontSize:"11px",letterSpacing:"2px"}}>
            smokApp Quantum & AI Lab
          </div>
          <div style={{fontSize:"17px",fontWeight:"bold",color:"#e6edf3",marginTop:"2px"}}>
            H7 Parking Violations — Live Monitor
          </div>
          <div style={{color:"#8b949e",fontSize:"11px",marginTop:"4px"}}>
            Ψₙ = cos(π · φ · n) &nbsp;|&nbsp; φ = {PHI.toFixed(6)} &nbsp;|&nbsp; n ∈ [1,6]
          </div>
        </div>
        <div style={{display:"flex",flexDirection:"column",alignItems:"flex-end",gap:"4px"}}>
          <LiveBadge live={live}/>
          {lastUpdate && <span style={{color:"#484f58",fontSize:"10px"}}>↻ {lastUpdate}</span>}
        </div>
      </div>

      {/* ── Live status bar ── */}
      <StatusBar status={status}/>

      {/* ── Tabs ── */}
      <div style={{display:"flex",gap:"8px",marginBottom:"16px"}}>
        {[["matrix","H7 Matrix"],["oracle","Metriplex Oracle"]].map(([v,l])=>(
          <button key={v} onClick={()=>setTab(v)} style={{
            padding:"5px 14px",borderRadius:"6px",border:"1px solid",fontSize:"12px",cursor:"pointer",
            background: tab===v?"#1f6feb":"#161b22",
            borderColor: tab===v?"#388bfd":"#30363d",
            color: tab===v?"#fff":"#8b949e"}}>
            {l}
          </button>
        ))}
      </div>

      {/* ── Matrix tab ── */}
      {tab === "matrix" && <>
        {/* View mode toggle */}
        <div style={{display:"flex",gap:"6px",marginBottom:"12px"}}>
          {[["matrix","Full Matrix (4×5)"],["top","Top / State"]].map(([v,l])=>(
            <button key={v} onClick={()=>setMode(v)} style={{
              padding:"4px 12px",borderRadius:"6px",border:"1px solid",fontSize:"11px",cursor:"pointer",
              background: mode===v?"#238636":"#161b22",
              borderColor: mode===v?"#2ea043":"#30363d",
              color: mode===v?"#fff":"#8b949e"}}>
              {l}
            </button>
          ))}
        </div>

        {/* Ψ reference */}
        <div style={{display:"flex",gap:"6px",marginBottom:"14px",flexWrap:"wrap"}}>
          {N_LIST.map(n=>{const p=psi(n);const t=ternary(p);
            const lbl=t===1?"constructive":t===0?"equilibrium":"destructive";const c=COLORS[lbl];
            return(
              <div key={n} style={{background:c.bg,border:`1px solid ${c.border}`,borderRadius:"6px",
                padding:"4px 10px",fontSize:"11px",textAlign:"center"}}>
                <div style={{color:"#8b949e"}}>n={n}</div>
                <div style={{color:c.text,fontWeight:"bold"}}>{p.toFixed(4)}</div>
                <div style={{color:c.text,fontSize:"10px"}}>{SYMBOL[t]}</div>
              </div>
            );
          })}
        </div>

        {/* Table */}
        <div style={{overflowX:"auto"}}>
          <table style={{width:"100%",borderCollapse:"collapse",fontSize:"12px"}}>
            <thead>
              <tr style={{background:"#161b22",borderBottom:"1px solid #30363d"}}>
                {["State","Violation","n","State-Vec","Ψₙ","Ternary","FW","BW","Label"].map(h=>(
                  <th key={h} style={{padding:"8px 10px",textAlign:"left",color:"#58a6ff",
                    fontWeight:"normal",letterSpacing:"1px"}}>{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {displayRows.map((r,i)=>{
                const lbl=r.label; const c=COLORS[lbl]; const isSel=selected===i;
                return(
                  <tr key={i} onClick={()=>setSelected(isSel?null:i)} style={{
                    background:isSel?c.bg:i%2===0?"#0d1117":"#0a0e14",
                    borderBottom:"1px solid #21262d",cursor:"pointer",
                    borderLeft:isSel?`3px solid ${c.border}`:"3px solid transparent",
                    transition:"background 0.15s"}}>
                    <td style={{padding:"7px 10px",color:"#e6edf3",fontWeight:"bold"}}>{r.state||r.st}</td>
                    <td style={{padding:"7px 10px",color:"#c9d1d9"}}>{r.violation||r.viol}</td>
                    <td style={{padding:"7px 10px",color:"#58a6ff",fontWeight:"bold"}}>{r.n}</td>
                    <td style={{padding:"7px 10px",color:"#8b949e"}}>{r.sv}</td>
                    <td style={{padding:"7px 10px",fontWeight:"bold",
                      color:parseFloat(r.psi)>0?"#4ade80":parseFloat(r.psi)<0?"#f87171":"#fde047"}}>
                      {r.psi}</td>
                    <td style={{padding:"7px 10px"}}>
                      <span style={{background:c.badge,color:c.text,borderRadius:"4px",
                        padding:"2px 8px",fontWeight:"bold",fontSize:"13px"}}>{SYMBOL[r.t]}</span>
                    </td>
                    <td style={{padding:"7px 10px",color:"#a5d6ff",letterSpacing:"2px"}}>{r.fw}</td>
                    <td style={{padding:"7px 10px",color:"#ffa657",letterSpacing:"2px"}}>{r.bw}</td>
                    <td style={{padding:"7px 10px",color:c.text,fontSize:"11px"}}>{r.label}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {/* Summary */}
        <div style={{marginTop:"16px",display:"flex",gap:"10px",flexWrap:"wrap"}}>
          {["constructive","equilibrium","destructive"].map(lbl=>{
            const c=COLORS[lbl]; const count=displayRows.filter(r=>r.label===lbl).length;
            return(
              <div key={lbl} style={{background:c.bg,border:`1px solid ${c.border}`,
                borderRadius:"8px",padding:"10px 16px",flex:"1",minWidth:"140px"}}>
                <div style={{color:c.text,fontWeight:"bold",fontSize:"13px"}}>
                  {SYMBOL[lbl==="constructive"?1:lbl==="equilibrium"?0:-1]} {lbl}
                </div>
                <div style={{color:"#8b949e",fontSize:"11px",marginTop:"2px"}}>
                  {count} combo{count!==1?"s":""}
                </div>
              </div>
            );
          })}
        </div>
      </>}

      {/* ── Oracle tab ── */}
      {tab === "oracle" && <OraclePanel oracle={oracleData}/>}

      <div style={{marginTop:"12px",color:"#484f58",fontSize:"10px"}}>
        {live
          ? `● Connected to Python backend · Matrix: 5s · Status: 2s · Oracle: 10s`
          : `● Offline — running local JS fallback · MD5 ≡ Python hashlib.md5`}
        &nbsp;· ε = 0.25
      </div>

      <style>{`
        @keyframes pulse {
          0%,100% { opacity:1; }
          50%      { opacity:0.3; }
        }
      `}</style>
    </div>
  );
}
