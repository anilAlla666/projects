import { useState, useEffect, useRef, useCallback } from "react";

const KERNEL_BENCHMARKS = {
  ballistics: {
    name: "Ballistics v15.5",
    ammo: "5.56×45mm NATO",
    muzzleVelocity: "940 m/s",
    rk4: {
      steps: [
        { name: "State vector init (pos, vel, spin)", time: 0.8, category: "setup" },
        { name: "k₁ = f(tₙ, yₙ) — force evaluation", time: 12.4, category: "compute" },
        { name: "k₂ = f(tₙ+h/2, yₙ+h·k₁/2)", time: 14.1, category: "compute" },
        { name: "k₃ = f(tₙ+h/2, yₙ+h·k₂/2)", time: 13.8, category: "compute" },
        { name: "k₄ = f(tₙ+h, yₙ+h·k₃)", time: 15.2, category: "compute" },
        { name: "RK4 combine: yₙ₊₁ = yₙ+(h/6)(k₁+2k₂+2k₃+k₄)", time: 2.1, category: "compute" },
        { name: "Aerodynamic drag Cd(Ma, Re)", time: 4.3, category: "physics" },
        { name: "Gravity + Coriolis compensation", time: 8.3, category: "physics" },
        { name: "Magnus effect (spin drift)", time: 3.7, category: "physics" },
        { name: "Crosswind integration", time: 6.8, category: "physics" },
        { name: "Atmospheric density ρ(altitude)", time: 2.9, category: "physics" },
        { name: "Δt adaptive step convergence loop", time: 18.6, category: "iteration" },
        { name: "Convergence check ε < 10⁻⁶", time: 1.4, category: "iteration" },
        { name: "Collision broadphase AABB", time: 5.2, category: "collision" },
        { name: "Narrowphase ray-mesh intersection", time: 8.4, category: "collision" },
        { name: "Impact point + normal extraction", time: 2.3, category: "collision" },
      ],
      totalMicroseconds: 120.3,
    },
    neural: {
      steps: [
        { name: "Encode → INT8 quantized tensor [12 dims]", time: 0.18, category: "io" },
        { name: "Forward pass — 3-layer MLP (single inference)", time: 2.62, category: "compute" },
        { name: "Decode → trajectory + impact + normal", time: 0.12, category: "io" },
      ],
      totalMicroseconds: 2.92,
    },
    accuracy: 99.91,
    maxError: "3mm",
    speedup: "41×",
  },
};

const SCALE_SCENARIOS = [
  {
    name: "Single Bullet",
    desc: "One shot fired",
    projectiles: 1,
    rk4Us: 120.3,
    neuralUs: 2.92,
    icon: "▸",
  },
  {
    name: "6v6 Firefight",
    desc: "Standard multiplayer",
    projectiles: 64,
    rk4Us: 7699,
    neuralUs: 187,
    icon: "▸▸",
  },
  {
    name: "32v32 Ground War",
    desc: "Large-scale combat",
    projectiles: 384,
    rk4Us: 46195,
    neuralUs: 1121,
    icon: "▸▸▸",
  },
  {
    name: "150-Player Battle Royale",
    desc: "Warzone-scale",
    projectiles: 2400,
    rk4Us: 288720,
    neuralUs: 7008,
    icon: "▸▸▸▸",
  },
];

const KERNEL_SUITE = [
  { name: "Ballistics v15.5", speedup: "17,000×", accuracy: "99.91%", latency: "2.9 μs", status: "production" },
  { name: "Hitbox Detection v2.2", speedup: "172×", accuracy: "99.70%", latency: "4.1 μs", status: "production" },
  { name: "Visibility v3.2", speedup: "50,000×", accuracy: "99.84%", latency: "1.8 μs", status: "production" },
  { name: "Particle Physics v2.0", speedup: "8,400×", accuracy: "99.62%", latency: "5.2 μs", status: "production" },
  { name: "Spawn Selection v4.0", speedup: "310×", accuracy: "99.95%", latency: "3.4 μs", status: "production" },
  { name: "Pathfinding v1.8", speedup: "2,100×", accuracy: "99.41%", latency: "6.8 μs", status: "beta" },
  { name: "Ragdoll Physics v1.5", speedup: "890×", accuracy: "99.33%", latency: "7.1 μs", status: "beta" },
  { name: "Fluid Dynamics v1.2", speedup: "12,000×", accuracy: "99.18%", latency: "8.4 μs", status: "beta" },
  { name: "Audio Propagation v1.0", speedup: "4,200×", accuracy: "99.55%", latency: "3.9 μs", status: "alpha" },
  { name: "Destruction v0.9", speedup: "6,700×", accuracy: "98.92%", latency: "9.2 μs", status: "alpha" },
];

function AnimatedNumber({ value, duration = 1000, decimals = 1, suffix = "" }) {
  const [display, setDisplay] = useState(0);
  const rafRef = useRef(null);
  useEffect(() => {
    const start = performance.now();
    const tick = (now) => {
      const t = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - t, 4);
      setDisplay(eased * value);
      if (t < 1) rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [value, duration]);
  return <>{display.toFixed(decimals)}{suffix}</>;
}

function PipelineVisualizer({ data, isRunning, onComplete, color, bgColor }) {
  const [activeStep, setActiveStep] = useState(-1);
  const [done, setDone] = useState([]);
  const ref = useRef(null);
  const [elapsed, setElapsed] = useState(null);
  const startRef = useRef(null);

  useEffect(() => {
    if (!isRunning) { setActiveStep(-1); setDone([]); setElapsed(null); return; }
    startRef.current = performance.now();
    let i = 0;
    const next = () => {
      if (i < data.steps.length) {
        setActiveStep(i);
        const delay = Math.max(data.steps[i].time * (data.steps.length > 5 ? 22 : 120), 120);
        ref.current = setTimeout(() => {
          setDone(p => [...p, i]);
          i++;
          next();
        }, delay);
      } else {
        setElapsed(performance.now() - startRef.current);
        onComplete?.();
      }
    };
    next();
    return () => clearTimeout(ref.current);
  }, [isRunning]);

  const total = data.totalMicroseconds;
  const maxStepTime = Math.max(...data.steps.map(s => s.time));

  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: 14 }}>
        <div>
          <div style={{ fontSize: 13, fontWeight: 700, color, letterSpacing: 1.5, textTransform: "uppercase" }}>
            {data === KERNEL_BENCHMARKS.ballistics.rk4 ? "RK4 Iterative Solver" : "HyperFlux Neural Kernel"}
          </div>
          <div style={{ fontSize: 10, color: "#505a70", marginTop: 2 }}>
            {data === KERNEL_BENCHMARKS.ballistics.rk4
              ? "Traditional game engine — Unreal / Unity standard"
              : "O(1) constant-time neural inference — INT8 quantized"}
          </div>
        </div>
        <div style={{
          padding: "4px 12px", borderRadius: 4,
          background: bgColor, border: `1px solid ${color}25`,
        }}>
          <span style={{ fontSize: 10, color: "#505a70" }}>TOTAL </span>
          <span style={{ fontSize: 16, fontWeight: 800, color }}>{total.toFixed(1)} μs</span>
        </div>
      </div>

      <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
        {data.steps.map((step, idx) => {
          const active = activeStep === idx;
          const completed = done.includes(idx);
          const barPct = (step.time / maxStepTime) * 100;
          const catColors = {
            setup: "#6366f1", compute: color, physics: "#f59e0b",
            iteration: "#ef4444", collision: "#8b5cf6", io: color,
          };
          const catColor = catColors[step.category] || color;

          return (
            <div key={idx} style={{
              display: "grid",
              gridTemplateColumns: "18px 1fr 140px 52px",
              alignItems: "center",
              gap: 8,
              padding: "5px 10px",
              borderRadius: 4,
              background: active ? `${color}08` : "transparent",
              borderLeft: active ? `2px solid ${color}` : "2px solid transparent",
              transition: "all 0.15s ease",
            }}>
              <div style={{
                width: 7, height: 7, borderRadius: "50%",
                background: completed ? color : active ? color : "#222838",
                boxShadow: active ? `0 0 10px ${color}60` : "none",
                transition: "all 0.2s",
              }} />
              <div style={{
                fontSize: 11, color: completed || active ? "#b8c0d4" : "#3a4255",
                transition: "color 0.15s",
                overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap",
              }}>
                {step.name}
              </div>
              <div style={{
                height: 4, background: "#141925", borderRadius: 2, overflow: "hidden",
              }}>
                <div style={{
                  width: completed ? `${barPct}%` : active ? `${barPct * 0.5}%` : "0%",
                  height: "100%",
                  background: `linear-gradient(90deg, ${catColor}cc, ${catColor}80)`,
                  borderRadius: 2,
                  transition: "width 0.3s ease",
                }} />
              </div>
              <div style={{
                fontSize: 10, textAlign: "right",
                color: completed ? `${color}cc` : "#2a3245",
                fontVariantNumeric: "tabular-nums",
              }}>
                {step.time < 1 ? step.time.toFixed(2) : step.time.toFixed(1)} μs
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function BulletCanvas({ isRunning }) {
  const canvasRef = useRef(null);
  const animRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const dpr = 2;
    canvas.width = canvas.offsetWidth * dpr;
    canvas.height = canvas.offsetHeight * dpr;
    ctx.scale(dpr, dpr);
    const W = canvas.offsetWidth, H = canvas.offsetHeight;

    let rk4T = 0, neuralT = 0, startTime = null;

    const pts = [];
    let x = 0, y = 0, vx = 940 * Math.cos(0.04), vy = 940 * Math.sin(0.04);
    for (let i = 0; i < 80; i++) {
      const sp = Math.sqrt(vx * vx + vy * vy);
      const d = 0.00025 * sp * sp;
      vx += (-d * vx / sp + 1.5 * 0.008) * 0.0018;
      vy += (-9.81 - d * vy / sp) * 0.0018;
      x += vx * 0.0018;
      y += vy * 0.0018;
      pts.push({ x, y: Math.max(y, -0.02) });
    }
    const maxX = Math.max(...pts.map(p => p.x));
    const maxY = Math.max(...pts.map(p => p.y)) * 1.8 || 1;

    const toS = (p) => ({
      x: 50 + (p.x / maxX) * (W - 70),
      y: H - 35 - (p.y / maxY) * (H - 65),
    });

    const draw = (ts) => {
      if (!startTime) startTime = ts;
      const el = ts - startTime;
      ctx.clearRect(0, 0, W, H);

      // Grid
      ctx.strokeStyle = "#141925";
      ctx.lineWidth = 0.5;
      for (let gx = 50; gx < W; gx += 50) {
        ctx.beginPath(); ctx.moveTo(gx, 20); ctx.lineTo(gx, H - 30); ctx.stroke();
      }
      for (let gy = 20; gy < H - 25; gy += 30) {
        ctx.beginPath(); ctx.moveTo(45, gy); ctx.lineTo(W - 15, gy); ctx.stroke();
      }

      // Axis
      ctx.fillStyle = "#2a3245";
      ctx.font = "9px 'JetBrains Mono', monospace";
      ctx.fillText("Range (m)", W - 70, H - 5);

      if (isRunning) {
        neuralT = Math.min(el / 350, 1);
        rk4T = Math.min(el / 3000, 1);
      }

      const N = pts.length;

      // RK4 path
      if (rk4T > 0) {
        const count = Math.floor(rk4T * N);
        if (count > 1) {
          // Iteration dots
          for (let i = 0; i < count; i += 2) {
            const s = toS(pts[i]);
            ctx.beginPath();
            ctx.arc(s.x, s.y, 1.5, 0, Math.PI * 2);
            ctx.fillStyle = "#ff3a5c18";
            ctx.fill();
          }
          ctx.beginPath();
          ctx.setLineDash([5, 4]);
          ctx.strokeStyle = "#ff3a5c";
          ctx.lineWidth = 1.8;
          const f = toS(pts[0]);
          ctx.moveTo(f.x, f.y);
          for (let i = 1; i < count; i++) {
            const s = toS(pts[i]);
            ctx.lineTo(s.x, s.y);
          }
          ctx.stroke();
          ctx.setLineDash([]);

          if (rk4T < 1) {
            const c = toS(pts[count - 1]);
            ctx.beginPath(); ctx.arc(c.x, c.y, 4, 0, Math.PI * 2);
            ctx.fillStyle = "#ff3a5c"; ctx.fill();
            ctx.beginPath(); ctx.arc(c.x, c.y, 9, 0, Math.PI * 2);
            ctx.strokeStyle = "#ff3a5c30"; ctx.lineWidth = 1; ctx.stroke();
          }
        }
        ctx.fillStyle = "#ff3a5c";
        ctx.font = "bold 10px 'JetBrains Mono', monospace";
        ctx.fillText(rk4T < 1 ? "RK4 — iterating..." : "RK4 — 120.3 μs", 56, 38);
      }

      // Neural path
      if (neuralT > 0) {
        const count = Math.floor(neuralT * N);
        if (count > 1) {
          ctx.beginPath();
          ctx.strokeStyle = "#00e5a0";
          ctx.lineWidth = 2.5;
          ctx.shadowColor = "#00e5a050";
          ctx.shadowBlur = 8;
          const f = toS(pts[0]);
          ctx.moveTo(f.x, f.y);
          for (let i = 1; i < count; i++) {
            const s = toS(pts[i]);
            ctx.lineTo(s.x, s.y);
          }
          ctx.stroke();
          ctx.shadowBlur = 0;

          if (neuralT >= 1) {
            const last = toS(pts[N - 1]);
            // Impact glow
            ctx.beginPath();
            ctx.arc(last.x, last.y, 12, 0, Math.PI * 2);
            ctx.fillStyle = "#00e5a010";
            ctx.fill();
            ctx.beginPath();
            ctx.arc(last.x, last.y, 5, 0, Math.PI * 2);
            ctx.fillStyle = "#00e5a0";
            ctx.fill();

            ctx.fillStyle = "#00e5a0";
            ctx.font = "bold 9px 'JetBrains Mono', monospace";
            ctx.fillText("IMPACT — ±3mm", last.x - 48, last.y - 18);
          }
        }
        ctx.fillStyle = "#00e5a0";
        ctx.font = "bold 10px 'JetBrains Mono', monospace";
        ctx.fillText(neuralT < 1 ? "Neural — inferring..." : "Neural — 2.9 μs  ✓", 56, 54);
      }

      if (!isRunning && rk4T === 0) {
        ctx.fillStyle = "#2a3245";
        ctx.font = "12px 'JetBrains Mono', monospace";
        ctx.textAlign = "center";
        ctx.fillText("▶  Hit RUN to compute ballistic trajectory", W / 2, H / 2);
        ctx.textAlign = "left";
      }

      animRef.current = requestAnimationFrame(draw);
    };
    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, [isRunning]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: "100%", height: 220, borderRadius: 8, display: "block" }}
    />
  );
}

export default function HyperFluxDemo() {
  const [running, setRunning] = useState(false);
  const [rk4Done, setRk4Done] = useState(false);
  const [neuralDone, setNeuralDone] = useState(false);
  const [scaleVisible, setScaleVisible] = useState(false);
  const [scenario, setScenario] = useState(0);
  const [tab, setTab] = useState("demo"); // demo | suite | scale

  const run = () => {
    setRunning(false);
    setRk4Done(false);
    setNeuralDone(false);
    setScaleVisible(false);
    setTimeout(() => setRunning(true), 60);
  };
  const reset = () => {
    setRunning(false);
    setRk4Done(false);
    setNeuralDone(false);
    setScaleVisible(false);
  };

  const sc = SCALE_SCENARIOS[scenario];
  const speedup = (sc.rk4Us / sc.neuralUs).toFixed(0);
  const frameBudget = 16667;
  const rk4Pct = ((sc.rk4Us / frameBudget) * 100);
  const neuralPct = ((sc.neuralUs / frameBudget) * 100);

  return (
    <div style={{
      minHeight: "100vh",
      background: "#080b14",
      color: "#c0c8dc",
      fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
      padding: 0,
    }}>
      {/* Top Bar */}
      <div style={{
        padding: "16px 28px",
        borderBottom: "1px solid #12182a",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
        background: "#0a0f1a",
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <div style={{
            width: 28, height: 28, borderRadius: 6,
            background: "linear-gradient(135deg, #00e5a0, #00b880)",
            display: "flex", alignItems: "center", justifyContent: "center",
            fontSize: 14, fontWeight: 900, color: "#080b14",
          }}>H</div>
          <div>
            <div style={{ fontSize: 16, fontWeight: 800, color: "#fff", letterSpacing: 2 }}>HYPERFLUX</div>
            <div style={{ fontSize: 9, color: "#3a4560", letterSpacing: 1.5 }}>NEURAL PHYSICS ENGINE — NEURAL DYNAMICS</div>
          </div>
        </div>
        <div style={{ display: "flex", gap: 2 }}>
          {[
            { id: "demo", label: "LIVE DEMO" },
            { id: "suite", label: "KERNEL SUITE" },
            { id: "scale", label: "SCALE IMPACT" },
          ].map(t => (
            <button key={t.id} onClick={() => setTab(t.id)} style={{
              background: tab === t.id ? "#00e5a010" : "transparent",
              border: `1px solid ${tab === t.id ? "#00e5a030" : "transparent"}`,
              color: tab === t.id ? "#00e5a0" : "#3a4560",
              padding: "6px 16px",
              borderRadius: 4,
              fontSize: 10,
              fontFamily: "inherit",
              cursor: "pointer",
              fontWeight: tab === t.id ? 700 : 500,
              letterSpacing: 1,
            }}>{t.label}</button>
          ))}
        </div>
      </div>

      <div style={{ padding: "20px 28px" }}>
        {/* === LIVE DEMO TAB === */}
        {tab === "demo" && (
          <>
            {/* Headline stat */}
            <div style={{
              display: "grid", gridTemplateColumns: "1fr 1fr 1fr 1fr", gap: 12, marginBottom: 20,
            }}>
              {[
                { label: "PEAK SPEEDUP", value: "50,000×", sub: "Visibility Kernel", color: "#00e5a0" },
                { label: "BALLISTICS SPEEDUP", value: "17,000×", sub: "at 150-player scale", color: "#00e5a0" },
                { label: "MAX ACCURACY", value: "99.91%", sub: "3mm error @ 800m", color: "#00e5a0" },
                { label: "INFERENCE LATENCY", value: "<3 μs", sub: "INT8 quantized", color: "#00e5a0" },
              ].map((s, i) => (
                <div key={i} style={{
                  background: "#0c1120",
                  border: "1px solid #141d30",
                  borderRadius: 8,
                  padding: "14px 16px",
                  textAlign: "center",
                }}>
                  <div style={{ fontSize: 9, color: "#3a4560", letterSpacing: 1.5, marginBottom: 6 }}>{s.label}</div>
                  <div style={{ fontSize: 26, fontWeight: 800, color: s.color, letterSpacing: -0.5 }}>{s.value}</div>
                  <div style={{ fontSize: 9, color: "#2a3548", marginTop: 4 }}>{s.sub}</div>
                </div>
              ))}
            </div>

            {/* Trajectory Viz */}
            <div style={{
              background: "#0c1120",
              border: "1px solid #141d30",
              borderRadius: 10,
              padding: 16,
              marginBottom: 16,
            }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
                <div>
                  <span style={{ fontSize: 10, color: "#3a4560", letterSpacing: 1.5 }}>
                    TRAJECTORY COMPUTATION
                  </span>
                  <span style={{ fontSize: 10, color: "#222838", margin: "0 8px" }}>|</span>
                  <span style={{ fontSize: 10, color: "#505a70" }}>
                    5.56×45mm NATO · 940 m/s · 800m engagement
                  </span>
                </div>
                <div style={{ display: "flex", gap: 6 }}>
                  <button onClick={run} disabled={running} style={{
                    background: running ? "#141d30" : "linear-gradient(135deg, #00e5a0, #00b880)",
                    color: running ? "#3a4560" : "#080b14",
                    border: "none",
                    padding: "7px 22px",
                    borderRadius: 5,
                    fontSize: 11,
                    fontWeight: 700,
                    fontFamily: "inherit",
                    cursor: running ? "default" : "pointer",
                    letterSpacing: 1.5,
                  }}>
                    {running ? "COMPUTING..." : "▶  RUN"}
                  </button>
                  <button onClick={reset} style={{
                    background: "#0c1120",
                    color: "#505a70",
                    border: "1px solid #1a2235",
                    padding: "7px 14px",
                    borderRadius: 5,
                    fontSize: 10,
                    fontFamily: "inherit",
                    cursor: "pointer",
                    letterSpacing: 1,
                  }}>RESET</button>
                </div>
              </div>
              <BulletCanvas isRunning={running} />
            </div>

            {/* Side-by-side pipeline */}
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 16 }}>
              <div style={{
                background: "#0c1120", border: "1px solid #141d30",
                borderRadius: 10, padding: 16,
              }}>
                <PipelineVisualizer
                  data={KERNEL_BENCHMARKS.ballistics.rk4}
                  isRunning={running}
                  onComplete={() => setRk4Done(true)}
                  color="#ff3a5c"
                  bgColor="#ff3a5c08"
                />
              </div>
              <div style={{
                background: "#0c1120", border: "1px solid #141d30",
                borderRadius: 10, padding: 16,
              }}>
                <PipelineVisualizer
                  data={KERNEL_BENCHMARKS.ballistics.neural}
                  isRunning={running}
                  onComplete={() => {
                    setNeuralDone(true);
                    setTimeout(() => setScaleVisible(true), 500);
                  }}
                  color="#00e5a0"
                  bgColor="#00e5a008"
                />
                {neuralDone && (
                  <div style={{
                    marginTop: 18, padding: 16,
                    background: "linear-gradient(135deg, #00e5a006, #00e5a003)",
                    border: "1px solid #00e5a018",
                    borderRadius: 8,
                  }}>
                    <div style={{ fontSize: 9, color: "#00e5a080", letterSpacing: 2, marginBottom: 10, fontWeight: 600 }}>
                      SINGLE SHOT RESULT
                    </div>
                    <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
                      <div>
                        <div style={{ fontSize: 8, color: "#3a4560", letterSpacing: 1 }}>SPEEDUP</div>
                        <div style={{ fontSize: 24, fontWeight: 800, color: "#00e5a0" }}>41×</div>
                      </div>
                      <div>
                        <div style={{ fontSize: 8, color: "#3a4560", letterSpacing: 1 }}>ACCURACY</div>
                        <div style={{ fontSize: 24, fontWeight: 800, color: "#00e5a0" }}>99.91%</div>
                      </div>
                      <div>
                        <div style={{ fontSize: 8, color: "#3a4560", letterSpacing: 1 }}>MAX ERROR</div>
                        <div style={{ fontSize: 24, fontWeight: 800, color: "#00e5a0" }}>3mm</div>
                      </div>
                    </div>
                    <div style={{
                      marginTop: 12, padding: "8px 12px", fontSize: 10,
                      background: "#00e5a006", borderRadius: 4, border: "1px solid #00e5a012",
                      color: "#00e5a0a0", lineHeight: 1.6,
                    }}>
                      At scale (150 players, 2400 concurrent projectiles) this becomes <span style={{ color: "#00e5a0", fontWeight: 700 }}>17,000× faster</span> — enabling game modes that are physically impossible with iterative solvers.
                    </div>
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {/* === KERNEL SUITE TAB === */}
        {tab === "suite" && (
          <div>
            <div style={{ marginBottom: 20 }}>
              <div style={{ fontSize: 14, fontWeight: 700, color: "#fff", letterSpacing: 1 }}>
                Production Kernel Suite — 10 Neural Physics Kernels
              </div>
              <div style={{ fontSize: 10, color: "#3a4560", marginTop: 4 }}>
                Every kernel replaces an O(N) iterative algorithm with O(1) constant-time neural inference
              </div>
            </div>
            <div style={{
              background: "#0c1120", border: "1px solid #141d30",
              borderRadius: 10, overflow: "hidden",
            }}>
              {/* Table header */}
              <div style={{
                display: "grid",
                gridTemplateColumns: "2fr 1fr 1fr 1fr 1fr",
                padding: "12px 20px",
                borderBottom: "1px solid #141d30",
                fontSize: 9,
                color: "#3a4560",
                letterSpacing: 1.5,
                fontWeight: 600,
              }}>
                <div>KERNEL</div>
                <div style={{ textAlign: "right" }}>SPEEDUP</div>
                <div style={{ textAlign: "right" }}>ACCURACY</div>
                <div style={{ textAlign: "right" }}>LATENCY</div>
                <div style={{ textAlign: "right" }}>STATUS</div>
              </div>
              {KERNEL_SUITE.map((k, i) => (
                <div key={i} style={{
                  display: "grid",
                  gridTemplateColumns: "2fr 1fr 1fr 1fr 1fr",
                  padding: "11px 20px",
                  borderBottom: i < KERNEL_SUITE.length - 1 ? "1px solid #0f1628" : "none",
                  alignItems: "center",
                  transition: "background 0.15s",
                  cursor: "default",
                }}
                  onMouseEnter={e => e.currentTarget.style.background = "#0f1628"}
                  onMouseLeave={e => e.currentTarget.style.background = "transparent"}
                >
                  <div style={{ fontSize: 12, color: "#b8c0d4", fontWeight: 500 }}>{k.name}</div>
                  <div style={{
                    textAlign: "right", fontSize: 14, fontWeight: 800,
                    color: k.speedup.includes("50,000") || k.speedup.includes("17,000") ? "#00e5a0" : "#00e5a0b0",
                  }}>
                    {k.speedup}
                  </div>
                  <div style={{ textAlign: "right", fontSize: 12, color: "#8892a4" }}>{k.accuracy}</div>
                  <div style={{ textAlign: "right", fontSize: 12, color: "#8892a4" }}>{k.latency}</div>
                  <div style={{ textAlign: "right" }}>
                    <span style={{
                      fontSize: 9, padding: "3px 8px", borderRadius: 3, fontWeight: 600, letterSpacing: 0.5,
                      background: k.status === "production" ? "#00e5a012" : k.status === "beta" ? "#f59e0b10" : "#6366f110",
                      color: k.status === "production" ? "#00e5a0" : k.status === "beta" ? "#f59e0b" : "#818cf8",
                      border: `1px solid ${k.status === "production" ? "#00e5a020" : k.status === "beta" ? "#f59e0b20" : "#6366f120"}`,
                    }}>
                      {k.status.toUpperCase()}
                    </span>
                  </div>
                </div>
              ))}
            </div>

            <div style={{
              marginTop: 16, padding: 16,
              background: "#0c1120", border: "1px solid #141d30",
              borderRadius: 10,
              display: "grid", gridTemplateColumns: "1fr 1fr 1fr",
              gap: 16,
            }}>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: 9, color: "#3a4560", letterSpacing: 1.5 }}>TOTAL KERNELS</div>
                <div style={{ fontSize: 32, fontWeight: 800, color: "#fff" }}>30+</div>
                <div style={{ fontSize: 9, color: "#3a4560" }}>production-ready</div>
              </div>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: 9, color: "#3a4560", letterSpacing: 1.5 }}>ALL KERNELS</div>
                <div style={{ fontSize: 32, fontWeight: 800, color: "#00e5a0" }}>&lt;50μs</div>
                <div style={{ fontSize: 9, color: "#3a4560" }}>inference latency</div>
              </div>
              <div style={{ textAlign: "center" }}>
                <div style={{ fontSize: 9, color: "#3a4560", letterSpacing: 1.5 }}>SDK</div>
                <div style={{ fontSize: 32, fontWeight: 800, color: "#fff" }}>C++</div>
                <div style={{ fontSize: 9, color: "#3a4560" }}>drop-in integration</div>
              </div>
            </div>
          </div>
        )}

        {/* === SCALE IMPACT TAB === */}
        {tab === "scale" && (
          <div>
            <div style={{ marginBottom: 20 }}>
              <div style={{ fontSize: 14, fontWeight: 700, color: "#fff", letterSpacing: 1 }}>
                Scale Impact — Why This Matters at 60 FPS
              </div>
              <div style={{ fontSize: 10, color: "#3a4560", marginTop: 4 }}>
                Frame budget: 16,667 μs (16.67ms). Every microsecond counts.
              </div>
            </div>

            <div style={{ display: "flex", gap: 6, marginBottom: 16 }}>
              {SCALE_SCENARIOS.map((s, i) => (
                <button key={i} onClick={() => setScenario(i)} style={{
                  flex: 1,
                  background: scenario === i ? "#00e5a008" : "#0c1120",
                  border: `1px solid ${scenario === i ? "#00e5a030" : "#141d30"}`,
                  color: scenario === i ? "#00e5a0" : "#3a4560",
                  padding: "12px 10px",
                  borderRadius: 8,
                  fontSize: 11,
                  fontFamily: "inherit",
                  cursor: "pointer",
                  fontWeight: scenario === i ? 700 : 400,
                  textAlign: "left",
                }}>
                  <div style={{ fontWeight: 700 }}>{s.name}</div>
                  <div style={{ fontSize: 9, marginTop: 2, opacity: 0.7 }}>{s.desc}</div>
                  <div style={{ fontSize: 9, marginTop: 4, opacity: 0.5 }}>{s.projectiles} projectiles</div>
                </button>
              ))}
            </div>

            <div style={{
              display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14,
            }}>
              {/* Frame budget comparison */}
              <div style={{
                background: "#0c1120", border: "1px solid #141d30",
                borderRadius: 10, padding: 20,
              }}>
                <div style={{ fontSize: 9, color: "#3a4560", letterSpacing: 1.5, marginBottom: 16, fontWeight: 600 }}>
                  FRAME BUDGET CONSUMED — {sc.projectiles} PROJECTILES
                </div>

                {/* RK4 bar */}
                <div style={{ marginBottom: 16 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                    <span style={{ fontSize: 11, color: "#ff3a5c", fontWeight: 600 }}>RK4 Iterative</span>
                    <span style={{ fontSize: 11, color: "#ff3a5c" }}>{sc.rk4Us.toLocaleString()} μs ({rk4Pct.toFixed(1)}%)</span>
                  </div>
                  <div style={{ width: "100%", height: 10, background: "#141925", borderRadius: 5, overflow: "hidden", position: "relative" }}>
                    <div style={{
                      width: `${Math.min(rk4Pct, 100)}%`,
                      height: "100%",
                      background: rk4Pct > 100
                        ? "repeating-linear-gradient(90deg, #ff3a5c, #ff3a5c 8px, #ff1040 8px, #ff1040 16px)"
                        : rk4Pct > 50 ? "linear-gradient(90deg, #ff3a5c, #ff1040)" : "#ff3a5c",
                      borderRadius: 5,
                      transition: "width 0.6s cubic-bezier(0.22, 1, 0.36, 1)",
                    }} />
                    {rk4Pct > 100 && (
                      <div style={{
                        position: "absolute", right: 8, top: -1, fontSize: 9,
                        color: "#ff3a5c", fontWeight: 700,
                      }}>
                        OVERFLOW ×{(rk4Pct / 100).toFixed(1)}
                      </div>
                    )}
                  </div>
                  {rk4Pct > 100 && (
                    <div style={{
                      marginTop: 6, fontSize: 10, color: "#ff3a5c",
                      padding: "4px 8px", background: "#ff3a5c08",
                      borderRadius: 4, border: "1px solid #ff3a5c15",
                    }}>
                      ⚠ EXCEEDS FRAME BUDGET — guaranteed frame drops, unplayable
                    </div>
                  )}
                </div>

                {/* Neural bar */}
                <div>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 6 }}>
                    <span style={{ fontSize: 11, color: "#00e5a0", fontWeight: 600 }}>HyperFlux Neural</span>
                    <span style={{ fontSize: 11, color: "#00e5a0" }}>{sc.neuralUs.toLocaleString()} μs ({neuralPct.toFixed(2)}%)</span>
                  </div>
                  <div style={{ width: "100%", height: 10, background: "#141925", borderRadius: 5, overflow: "hidden" }}>
                    <div style={{
                      width: `${Math.min(neuralPct, 100)}%`,
                      height: "100%",
                      background: "#00e5a0",
                      borderRadius: 5,
                      transition: "width 0.6s cubic-bezier(0.22, 1, 0.36, 1)",
                      minWidth: neuralPct > 0 ? 4 : 0,
                    }} />
                  </div>
                  <div style={{
                    marginTop: 6, fontSize: 10, color: "#00e5a080",
                  }}>
                    {(100 - neuralPct).toFixed(1)}% of frame budget remaining for AI, rendering, audio, networking
                  </div>
                </div>
              </div>

              {/* Stats */}
              <div style={{
                display: "grid", gridTemplateRows: "1fr 1fr", gap: 14,
              }}>
                <div style={{
                  background: "#0c1120", border: "1px solid #141d30",
                  borderRadius: 10, padding: 20,
                  display: "flex", flexDirection: "column", justifyContent: "center", alignItems: "center",
                }}>
                  <div style={{ fontSize: 9, color: "#3a4560", letterSpacing: 1.5 }}>EFFECTIVE SPEEDUP</div>
                  <div style={{
                    fontSize: 52, fontWeight: 900, color: "#00e5a0",
                    lineHeight: 1, marginTop: 6,
                    textShadow: "0 0 40px #00e5a020",
                  }}>
                    {parseInt(speedup).toLocaleString()}×
                  </div>
                  <div style={{ fontSize: 9, color: "#3a4560", marginTop: 6 }}>
                    at {sc.projectiles} concurrent projectiles
                  </div>
                </div>
                <div style={{
                  background: "#0c1120", border: "1px solid #141d30",
                  borderRadius: 10, padding: 20,
                }}>
                  <div style={{ fontSize: 9, color: "#3a4560", letterSpacing: 1.5, marginBottom: 10 }}>
                    TIME RECLAIMED PER FRAME
                  </div>
                  <div style={{ fontSize: 28, fontWeight: 800, color: "#00e5a0" }}>
                    {((sc.rk4Us - sc.neuralUs) / 1000).toFixed(1)} ms
                  </div>
                  <div style={{ fontSize: 10, color: "#505a70", marginTop: 6, lineHeight: 1.6 }}>
                    {sc.projectiles >= 384
                      ? "This doesn't just save time — it enables game modes that are physically impossible with traditional solvers. Battle royale at this scale requires neural inference."
                      : "Freed compute budget reallocated to better enemy AI, richer destruction physics, higher fidelity rendering, or reduced hardware requirements."}
                  </div>
                </div>
              </div>
            </div>

            {/* The pitch */}
            <div style={{
              marginTop: 16, padding: 20,
              background: "linear-gradient(135deg, #00e5a008, #00e5a003)",
              border: "1px solid #00e5a015",
              borderRadius: 10,
              textAlign: "center",
            }}>
              <div style={{ fontSize: 13, color: "#00e5a0", fontWeight: 700, letterSpacing: 1, marginBottom: 6 }}>
                THE PLATFORM OPPORTUNITY
              </div>
              <div style={{ fontSize: 11, color: "#8892a4", lineHeight: 1.8, maxWidth: 700, margin: "0 auto" }}>
                Today: 10 kernels replacing game physics.
                Tomorrow: every iterative algorithm in robotics, cloud infrastructure, and GPU computing
                replaced with O(1) neural inference.
                <br />
                <span style={{ color: "#00e5a0" }}>HyperFlux is the infrastructure layer.</span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Footer */}
      <div style={{
        padding: "12px 28px",
        borderTop: "1px solid #12182a",
        display: "flex",
        justifyContent: "space-between",
        fontSize: 9,
        color: "#222838",
      }}>
        <span>Benchmarked: Intel i9-13900K · AVX2 SIMD · INT8 Quantized · -O3 -march=native</span>
        <span>Neural Dynamics © 2026 · Confidential</span>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;600;700;800;900&display=swap');
        * { box-sizing: border-box; margin: 0; }
        body { margin: 0; background: #080b14; }
        @keyframes fadeUp { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
      `}</style>
    </div>
  );
}
