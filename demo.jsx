import { useState, useEffect, useRef } from "react";

const PHYSICS_DATA = {
  rk4: {
    label: "RK4 Iterative Solver",
    subtitle: "Traditional Game Engine (Unreal/Unity)",
    steps: [
      { name: "Initialize state vector", time: 0.8 },
      { name: "Compute k₁ = f(tₙ, yₙ)", time: 12.4 },
      { name: "Compute k₂ = f(tₙ + h/2, yₙ + h·k₁/2)", time: 14.1 },
      { name: "Compute k₃ = f(tₙ + h/2, yₙ + h·k₂/2)", time: 13.8 },
      { name: "Compute k₄ = f(tₙ + h, yₙ + h·k₃)", time: 15.2 },
      { name: "Combine: yₙ₊₁ = yₙ + (h/6)(k₁ + 2k₂ + 2k₃ + k₄)", time: 2.1 },
      { name: "Apply drag coefficient (Cd)", time: 4.3 },
      { name: "Apply gravity compensation", time: 3.2 },
      { name: "Wind deflection integration", time: 6.8 },
      { name: "Coriolis effect (long range)", time: 5.1 },
      { name: "Iterate for Δt convergence", time: 18.6 },
      { name: "Collision ray-cast", time: 8.4 },
    ],
  },
  neural: {
    label: "HyperFlux Neural Kernel",
    subtitle: "O(1) Constant-Time Inference",
    steps: [
      { name: "Input encode → INT8 quantized tensor", time: 0.3 },
      { name: "Neural forward pass (single inference)", time: 2.8 },
      { name: "Output decode → trajectory + impact", time: 0.2 },
    ],
  },
};

const SCENARIOS = [
  {
    name: "Single Bullet",
    icon: "•",
    projectiles: 1,
    rk4Time: 104.8,
    neuralTime: 3.3,
    accuracy: 99.91,
    errorMm: 3,
  },
  {
    name: "Squad Fight (6v6)",
    icon: "••",
    projectiles: 48,
    rk4Time: 5030,
    neuralTime: 158,
    accuracy: 99.91,
    errorMm: 3,
  },
  {
    name: "Ground War (32v32)",
    icon: "•••",
    projectiles: 256,
    rk4Time: 26828,
    neuralTime: 845,
    accuracy: 99.91,
    errorMm: 3,
  },
  {
    name: "Battle Royale (150)",
    icon: "••••",
    projectiles: 1200,
    rk4Time: 125760,
    neuralTime: 3960,
    accuracy: 99.91,
    errorMm: 3,
  },
];

const TRAJECTORY_PARAMS = {
  v0: 900,
  angle: 2.5,
  gravity: 9.81,
  drag: 0.0003,
  wind: 1.2,
  steps: 60,
};

function generateTrajectoryPoints() {
  const { v0, angle, gravity, drag, wind, steps } = TRAJECTORY_PARAMS;
  const rad = (angle * Math.PI) / 180;
  const points = [];
  let x = 0, y = 0, vx = v0 * Math.cos(rad), vy = v0 * Math.sin(rad);
  const dt = 0.002;
  for (let i = 0; i < steps; i++) {
    const speed = Math.sqrt(vx * vx + vy * vy);
    const dragF = drag * speed * speed;
    const ax = -dragF * (vx / speed) + wind * 0.01;
    const ay = -gravity - dragF * (vy / speed);
    vx += ax * dt;
    vy += ay * dt;
    x += vx * dt;
    y += vy * dt;
    points.push({ x, y: Math.max(y, -0.01) });
  }
  return points;
}

function AnimatedCounter({ target, duration = 1200, suffix = "", prefix = "", decimals = 1 }) {
  const [val, setVal] = useState(0);
  const ref = useRef(null);
  useEffect(() => {
    const start = performance.now();
    const animate = (now) => {
      const elapsed = now - start;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setVal(eased * target);
      if (progress < 1) ref.current = requestAnimationFrame(animate);
    };
    ref.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(ref.current);
  }, [target, duration]);
  return <span>{prefix}{val.toFixed(decimals)}{suffix}</span>;
}

function StepVisualizer({ data, isRunning, onComplete, color }) {
  const [activeStep, setActiveStep] = useState(-1);
  const [completedSteps, setCompletedSteps] = useState([]);
  const timerRef = useRef(null);

  useEffect(() => {
    if (!isRunning) {
      setActiveStep(-1);
      setCompletedSteps([]);
      return;
    }
    let step = 0;
    const runStep = () => {
      if (step < data.steps.length) {
        setActiveStep(step);
        const scaledTime = Math.max(data.steps[step].time * (data.steps.length > 5 ? 18 : 80), 100);
        timerRef.current = setTimeout(() => {
          setCompletedSteps((p) => [...p, step]);
          step++;
          runStep();
        }, scaledTime);
      } else {
        onComplete?.();
      }
    };
    runStep();
    return () => clearTimeout(timerRef.current);
  }, [isRunning]);

  const totalTime = data.steps.reduce((s, st) => s + st.time, 0);

  return (
    <div style={{ fontFamily: "'JetBrains Mono', 'Fira Code', monospace", fontSize: 12 }}>
      <div style={{ marginBottom: 12, display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <div style={{ fontSize: 14, fontWeight: 700, color, letterSpacing: 1 }}>{data.label}</div>
          <div style={{ fontSize: 11, color: "#8892a4", marginTop: 2 }}>{data.subtitle}</div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: 11, color: "#8892a4" }}>TOTAL LATENCY</div>
          <div style={{ fontSize: 18, fontWeight: 700, color }}>{totalTime.toFixed(1)} μs</div>
        </div>
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        {data.steps.map((step, i) => {
          const isActive = activeStep === i;
          const isDone = completedSteps.includes(i);
          const barWidth = (step.time / 20) * 100;
          return (
            <div
              key={i}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                padding: "4px 8px",
                borderRadius: 4,
                background: isActive ? `${color}12` : isDone ? `${color}08` : "transparent",
                border: `1px solid ${isActive ? `${color}40` : "transparent"}`,
                transition: "all 0.2s ease",
              }}
            >
              <div style={{
                width: 6, height: 6, borderRadius: "50%",
                background: isDone ? color : isActive ? color : "#2a3040",
                boxShadow: isActive ? `0 0 8px ${color}80` : "none",
                transition: "all 0.2s ease",
                flexShrink: 0,
              }} />
              <div style={{ flex: 1, color: isDone || isActive ? "#c8d0e0" : "#4a5568", transition: "color 0.2s" }}>
                {step.name}
              </div>
              <div style={{
                width: 120, height: 4, background: "#1a1f2e", borderRadius: 2, overflow: "hidden", flexShrink: 0,
              }}>
                <div style={{
                  width: `${isDone ? barWidth : isActive ? barWidth * 0.6 : 0}%`,
                  height: "100%",
                  background: `linear-gradient(90deg, ${color}, ${color}90)`,
                  borderRadius: 2,
                  transition: "width 0.3s ease",
                }} />
              </div>
              <div style={{ width: 48, textAlign: "right", color: isDone ? color : "#4a5568", fontSize: 11 }}>
                {step.time}μs
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function TrajectoryCanvas({ isRunning, rk4Done, neuralDone }) {
  const canvasRef = useRef(null);
  const animRef = useRef(null);
  const points = useRef(generateTrajectoryPoints());

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const W = canvas.width, H = canvas.height;
    let rk4Progress = 0, neuralProgress = 0;
    let startTime = null;

    const draw = (timestamp) => {
      if (!startTime) startTime = timestamp;
      ctx.clearRect(0, 0, W, H);

      // Grid
      ctx.strokeStyle = "#1a2030";
      ctx.lineWidth = 0.5;
      for (let x = 0; x < W; x += 40) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
      }
      for (let y = 0; y < H; y += 40) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
      }

      // Axis labels
      ctx.fillStyle = "#3a4560";
      ctx.font = "10px 'JetBrains Mono', monospace";
      ctx.fillText("Distance (m)", W - 80, H - 8);
      ctx.save();
      ctx.translate(12, 80);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText("Height (m)", 0, 0);
      ctx.restore();

      const pts = points.current;
      const maxX = Math.max(...pts.map(p => p.x));
      const maxY = Math.max(...pts.map(p => p.y)) * 1.3;
      const padL = 40, padR = 20, padT = 25, padB = 30;
      const scaleX = (W - padL - padR) / maxX;
      const scaleY = (H - padT - padB) / maxY;
      const toScreen = (p) => ({ x: padL + p.x * scaleX, y: H - padB - p.y * scaleY });

      if (isRunning) {
        const elapsed = timestamp - startTime;
        neuralProgress = Math.min(elapsed / 400, 1);
        rk4Progress = Math.min(elapsed / 2400, 1);
      }

      const totalPts = pts.length;

      // RK4 trajectory
      if (rk4Progress > 0) {
        const rk4Count = Math.floor(rk4Progress * totalPts);
        if (rk4Count > 1) {
          // Iteration markers
          for (let i = 0; i < rk4Count; i += 3) {
            const sp = toScreen(pts[i]);
            ctx.fillStyle = "#ff4c6a20";
            ctx.beginPath();
            ctx.arc(sp.x, sp.y, 2, 0, Math.PI * 2);
            ctx.fill();
          }
          // Main line
          ctx.beginPath();
          ctx.strokeStyle = "#ff4c6a";
          ctx.lineWidth = 2;
          ctx.setLineDash([4, 3]);
          const first = toScreen(pts[0]);
          ctx.moveTo(first.x, first.y);
          for (let i = 1; i < rk4Count; i++) {
            const sp = toScreen(pts[i]);
            ctx.lineTo(sp.x, sp.y);
          }
          ctx.stroke();
          ctx.setLineDash([]);

          // Current position marker
          if (rk4Progress < 1) {
            const cp = toScreen(pts[rk4Count - 1]);
            ctx.beginPath();
            ctx.arc(cp.x, cp.y, 4, 0, Math.PI * 2);
            ctx.fillStyle = "#ff4c6a";
            ctx.fill();
            ctx.beginPath();
            ctx.arc(cp.x, cp.y, 8, 0, Math.PI * 2);
            ctx.strokeStyle = "#ff4c6a40";
            ctx.lineWidth = 1;
            ctx.stroke();
          }

          // Label
          ctx.fillStyle = "#ff4c6a";
          ctx.font = "bold 10px 'JetBrains Mono', monospace";
          ctx.fillText(`RK4 — ${rk4Progress < 1 ? "computing..." : "104.8 μs"}`, padL + 8, padT + 14);
        }
      }

      // Neural trajectory
      if (neuralProgress > 0) {
        const nCount = Math.floor(neuralProgress * totalPts);
        if (nCount > 1) {
          ctx.beginPath();
          ctx.strokeStyle = "#00e5a0";
          ctx.lineWidth = 2.5;
          ctx.shadowColor = "#00e5a040";
          ctx.shadowBlur = 6;
          const first = toScreen(pts[0]);
          ctx.moveTo(first.x, first.y);
          for (let i = 1; i < nCount; i++) {
            const sp = toScreen(pts[i]);
            ctx.lineTo(sp.x, sp.y);
          }
          ctx.stroke();
          ctx.shadowBlur = 0;

          // Impact point
          if (neuralProgress >= 1) {
            const last = toScreen(pts[totalPts - 1]);
            ctx.beginPath();
            ctx.arc(last.x, last.y, 5, 0, Math.PI * 2);
            ctx.fillStyle = "#00e5a0";
            ctx.fill();
            ctx.beginPath();
            ctx.arc(last.x, last.y, 10, 0, Math.PI * 2);
            ctx.strokeStyle = "#00e5a050";
            ctx.lineWidth = 1.5;
            ctx.stroke();

            // Accuracy callout
            ctx.fillStyle = "#00e5a0";
            ctx.font = "bold 10px 'JetBrains Mono', monospace";
            ctx.fillText("±3mm error", last.x - 40, last.y - 16);
          }

          ctx.fillStyle = "#00e5a0";
          ctx.font = "bold 10px 'JetBrains Mono', monospace";
          ctx.fillText(`Neural — ${neuralProgress < 1 ? "inferring..." : "3.3 μs ✓"}`, padL + 8, padT + 30);
        }
      }

      if (!isRunning && rk4Progress === 0) {
        ctx.fillStyle = "#3a4560";
        ctx.font = "13px 'JetBrains Mono', monospace";
        ctx.textAlign = "center";
        ctx.fillText("Press RUN to compute ballistic trajectory", W / 2, H / 2);
        ctx.textAlign = "left";
      }

      animRef.current = requestAnimationFrame(draw);
    };

    animRef.current = requestAnimationFrame(draw);
    return () => cancelAnimationFrame(animRef.current);
  }, [isRunning]);

  return <canvas ref={canvasRef} width={700} height={260} style={{ width: "100%", height: 260, borderRadius: 8 }} />;
}

export default function HyperFluxDemo() {
  const [isRunning, setIsRunning] = useState(false);
  const [rk4Done, setRk4Done] = useState(false);
  const [neuralDone, setNeuralDone] = useState(false);
  const [selectedScenario, setSelectedScenario] = useState(0);
  const [showScale, setShowScale] = useState(false);

  const handleRun = () => {
    setIsRunning(false);
    setRk4Done(false);
    setNeuralDone(false);
    setShowScale(false);
    setTimeout(() => setIsRunning(true), 50);
  };

  const handleReset = () => {
    setIsRunning(false);
    setRk4Done(false);
    setNeuralDone(false);
    setShowScale(false);
  };

  const sc = SCENARIOS[selectedScenario];
  const speedup = (sc.rk4Time / sc.neuralTime).toFixed(0);
  const frameBudget = 16667;
  const rk4Pct = ((sc.rk4Time / frameBudget) * 100).toFixed(1);
  const neuralPct = ((sc.neuralTime / frameBudget) * 100).toFixed(2);

  return (
    <div style={{
      minHeight: "100vh",
      background: "#0a0e17",
      color: "#c8d0e0",
      fontFamily: "'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace",
      padding: "24px 28px",
      boxSizing: "border-box",
    }}>
      {/* Header */}
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 28 }}>
        <div>
          <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
            <div style={{
              width: 8, height: 8, borderRadius: "50%",
              background: "#00e5a0",
              boxShadow: "0 0 12px #00e5a060",
            }} />
            <span style={{ fontSize: 22, fontWeight: 800, letterSpacing: 2, color: "#fff" }}>
              HYPERFLUX
            </span>
            <span style={{
              fontSize: 10, padding: "2px 8px", borderRadius: 3,
              background: "#00e5a018", color: "#00e5a0", border: "1px solid #00e5a030",
              fontWeight: 600, letterSpacing: 1,
            }}>
              NEURAL PHYSICS ENGINE
            </span>
          </div>
          <div style={{ fontSize: 11, color: "#4a5568", marginTop: 4, letterSpacing: 0.5 }}>
            O(N) Iterative → O(1) Neural Inference  ·  Ballistics Kernel v15.5
          </div>
        </div>
        <div style={{ textAlign: "right" }}>
          <div style={{ fontSize: 10, color: "#4a5568", letterSpacing: 1 }}>NEURAL DYNAMICS</div>
          <div style={{ fontSize: 10, color: "#3a4560" }}>neuraldynamicsteam.io</div>
        </div>
      </div>

      {/* Trajectory Visualization */}
      <div style={{
        background: "#0d1220",
        border: "1px solid #1a2030",
        borderRadius: 10,
        padding: 16,
        marginBottom: 20,
      }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
          <div style={{ fontSize: 11, color: "#4a5568", letterSpacing: 1, fontWeight: 600 }}>
            TRAJECTORY COMPUTATION — 5.56×45mm NATO @ 940 m/s
          </div>
          <div style={{ display: "flex", gap: 8 }}>
            <button
              onClick={handleRun}
              style={{
                background: isRunning ? "#1a2030" : "linear-gradient(135deg, #00e5a0, #00c488)",
                color: isRunning ? "#4a5568" : "#0a0e17",
                border: "none",
                padding: "6px 20px",
                borderRadius: 5,
                fontSize: 11,
                fontWeight: 700,
                fontFamily: "inherit",
                cursor: isRunning ? "default" : "pointer",
                letterSpacing: 1,
              }}
            >
              {isRunning ? "RUNNING..." : "▶  RUN"}
            </button>
            <button
              onClick={handleReset}
              style={{
                background: "#1a2030",
                color: "#8892a4",
                border: "1px solid #2a3040",
                padding: "6px 14px",
                borderRadius: 5,
                fontSize: 11,
                fontFamily: "inherit",
                cursor: "pointer",
                letterSpacing: 1,
              }}
            >
              RESET
            </button>
          </div>
        </div>
        <TrajectoryCanvas isRunning={isRunning} rk4Done={rk4Done} neuralDone={neuralDone} />
      </div>

      {/* Step-by-step Comparison */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 20 }}>
        <div style={{
          background: "#0d1220",
          border: "1px solid #1a2030",
          borderRadius: 10,
          padding: 16,
        }}>
          <StepVisualizer
            data={PHYSICS_DATA.rk4}
            isRunning={isRunning}
            onComplete={() => setRk4Done(true)}
            color="#ff4c6a"
          />
        </div>
        <div style={{
          background: "#0d1220",
          border: "1px solid #1a2030",
          borderRadius: 10,
          padding: 16,
        }}>
          <StepVisualizer
            data={PHYSICS_DATA.neural}
            isRunning={isRunning}
            onComplete={() => {
              setNeuralDone(true);
              setTimeout(() => setShowScale(true), 600);
            }}
            color="#00e5a0"
          />
          {neuralDone && (
            <div style={{
              marginTop: 16,
              padding: "12px 16px",
              background: "#00e5a008",
              border: "1px solid #00e5a020",
              borderRadius: 8,
            }}>
              <div style={{ fontSize: 11, color: "#00e5a0", fontWeight: 600, letterSpacing: 1, marginBottom: 6 }}>
                RESULT
              </div>
              <div style={{ display: "flex", gap: 24 }}>
                <div>
                  <div style={{ fontSize: 10, color: "#4a5568" }}>SPEEDUP</div>
                  <div style={{ fontSize: 22, fontWeight: 800, color: "#00e5a0" }}>
                    <AnimatedCounter target={1200} suffix="×" decimals={0} />
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 10, color: "#4a5568" }}>ACCURACY</div>
                  <div style={{ fontSize: 22, fontWeight: 800, color: "#00e5a0" }}>
                    <AnimatedCounter target={99.91} suffix="%" />
                  </div>
                </div>
                <div>
                  <div style={{ fontSize: 10, color: "#4a5568" }}>MAX ERROR</div>
                  <div style={{ fontSize: 22, fontWeight: 800, color: "#00e5a0" }}>
                    <AnimatedCounter target={3} suffix="mm" decimals={0} />
                  </div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Scale Impact */}
      {showScale && (
        <div style={{
          background: "#0d1220",
          border: "1px solid #1a2030",
          borderRadius: 10,
          padding: 20,
          animation: "fadeIn 0.5s ease",
        }}>
          <div style={{ fontSize: 12, color: "#8892a4", fontWeight: 600, letterSpacing: 1, marginBottom: 14 }}>
            WHAT THIS UNLOCKS — SCALE IMPACT @ 60 FPS (16.67ms frame budget)
          </div>

          {/* Scenario Tabs */}
          <div style={{ display: "flex", gap: 6, marginBottom: 16 }}>
            {SCENARIOS.map((s, i) => (
              <button
                key={i}
                onClick={() => setSelectedScenario(i)}
                style={{
                  background: selectedScenario === i ? "#00e5a015" : "#111827",
                  border: `1px solid ${selectedScenario === i ? "#00e5a040" : "#1a2030"}`,
                  color: selectedScenario === i ? "#00e5a0" : "#4a5568",
                  padding: "8px 14px",
                  borderRadius: 6,
                  fontSize: 11,
                  fontFamily: "inherit",
                  cursor: "pointer",
                  fontWeight: selectedScenario === i ? 700 : 400,
                }}
              >
                {s.name}
              </button>
            ))}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 16 }}>
            {/* Frame Budget */}
            <div style={{ background: "#111827", borderRadius: 8, padding: 16 }}>
              <div style={{ fontSize: 10, color: "#4a5568", letterSpacing: 1, marginBottom: 8 }}>
                FRAME BUDGET CONSUMED — {sc.projectiles} PROJECTILES
              </div>
              <div style={{ marginBottom: 8 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                  <span style={{ fontSize: 11, color: "#ff4c6a" }}>RK4</span>
                  <span style={{ fontSize: 11, color: "#ff4c6a" }}>{rk4Pct}%</span>
                </div>
                <div style={{ width: "100%", height: 6, background: "#1a2030", borderRadius: 3, overflow: "hidden" }}>
                  <div style={{
                    width: `${Math.min(parseFloat(rk4Pct), 100)}%`,
                    height: "100%",
                    background: parseFloat(rk4Pct) > 50 ? "linear-gradient(90deg, #ff4c6a, #ff2040)" : "#ff4c6a",
                    borderRadius: 3,
                    transition: "width 0.5s ease",
                  }} />
                </div>
              </div>
              <div>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                  <span style={{ fontSize: 11, color: "#00e5a0" }}>Neural</span>
                  <span style={{ fontSize: 11, color: "#00e5a0" }}>{neuralPct}%</span>
                </div>
                <div style={{ width: "100%", height: 6, background: "#1a2030", borderRadius: 3, overflow: "hidden" }}>
                  <div style={{
                    width: `${Math.min(parseFloat(neuralPct), 100)}%`,
                    height: "100%",
                    background: "#00e5a0",
                    borderRadius: 3,
                    transition: "width 0.5s ease",
                  }} />
                </div>
              </div>
              {parseFloat(rk4Pct) > 50 && (
                <div style={{
                  marginTop: 10, fontSize: 10, color: "#ff4c6a", padding: "4px 8px",
                  background: "#ff4c6a10", borderRadius: 4, border: "1px solid #ff4c6a20",
                }}>
                  ⚠ RK4 EXCEEDS BUDGET — FRAME DROP
                </div>
              )}
            </div>

            {/* Speedup */}
            <div style={{
              background: "#111827", borderRadius: 8, padding: 16, textAlign: "center",
              display: "flex", flexDirection: "column", justifyContent: "center",
            }}>
              <div style={{ fontSize: 10, color: "#4a5568", letterSpacing: 1, marginBottom: 4 }}>SPEEDUP</div>
              <div style={{ fontSize: 36, fontWeight: 800, color: "#00e5a0" }}>
                <AnimatedCounter target={parseFloat(speedup)} suffix="×" decimals={0} duration={800} />
              </div>
              <div style={{ fontSize: 10, color: "#4a5568", marginTop: 4 }}>FASTER COMPUTATION</div>
            </div>

            {/* Time Saved */}
            <div style={{ background: "#111827", borderRadius: 8, padding: 16 }}>
              <div style={{ fontSize: 10, color: "#4a5568", letterSpacing: 1, marginBottom: 8 }}>
                TIME RECLAIMED PER FRAME
              </div>
              <div style={{ fontSize: 24, fontWeight: 800, color: "#00e5a0", marginBottom: 4 }}>
                {((sc.rk4Time - sc.neuralTime) / 1000).toFixed(1)} ms
              </div>
              <div style={{ fontSize: 10, color: "#8892a4", lineHeight: 1.5 }}>
                freed for better AI, richer physics, higher fidelity rendering, or more players
              </div>
              <div style={{
                marginTop: 10, fontSize: 10, color: "#00e5a0", padding: "6px 8px",
                background: "#00e5a008", borderRadius: 4, border: "1px solid #00e5a020", lineHeight: 1.5,
              }}>
                {sc.projectiles >= 256
                  ? "→ enables modes that are currently impossible with iterative solvers"
                  : "→ GPU cycles redirected to gameplay systems"}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Footer */}
      <div style={{
        marginTop: 20,
        padding: "12px 16px",
        background: "#0d1220",
        borderRadius: 8,
        border: "1px solid #1a2030",
        display: "flex",
        justifyContent: "space-between",
        alignItems: "center",
      }}>
        <div style={{ fontSize: 10, color: "#3a4560" }}>
          Benchmarked on Intel i9-13900K · AVX2 SIMD · INT8 Quantized · Kernel compiled with -O3 -march=native
        </div>
        <div style={{ fontSize: 10, color: "#3a4560" }}>
          Neural Dynamics © 2026 · Confidential
        </div>
      </div>

      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700;800&display=swap');
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(10px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}
