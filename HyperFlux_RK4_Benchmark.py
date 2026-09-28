"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         HYPERFLUX v16.0 - ACCURACY vs RK4 GROUND TRUTH                       ║
║                                                                              ║
║  Compares O(1) Neural Network vs O(N) RK4 Physics Simulation                 ║
║  This is the REAL accuracy test.                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
import glob

print("=" * 70)
print("  HYPERFLUX: O(1) Neural Net vs O(N) RK4 Ground Truth")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

NUM_SAMPLES = 10000
RK4_SUBSTEPS = 100  # High accuracy ground truth

# Physics constants
GRAVITY = 9.81
AIR_DENSITY = 1.225
SPEED_OF_SOUND = 343.0

# ═══════════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from google.colab import drive
    drive.mount('/content/drive')
except:
    pass

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"  Device: {device}")

# Find models
v155_path = None
v160_path = None

for p in glob.glob("/content/drive/MyDrive/HyperFlux_v15.5_Unified_*/unified_ballistics_fp16.pt"):
    v155_path = p
    break

for p in glob.glob("/content/drive/MyDrive/HyperFlux_v16.0_INT8_*/int8_weights.npz"):
    v160_path = p
    break

print(f"  v15.5: {v155_path}")
print(f"  v16.0: {v160_path}")

# ═══════════════════════════════════════════════════════════════════════════════
# RK4 PHYSICS ENGINE (GROUND TRUTH)
# ═══════════════════════════════════════════════════════════════════════════════

class RK4Engine:
    """O(N) RK4 physics simulation - this is ground truth."""
    
    def __init__(self, gravity=9.81, air_density=1.225):
        self.g = gravity
        self.rho = air_density
    
    def drag_accel(self, vel, Cd, A, m):
        """Quadratic drag: F = 0.5 * rho * Cd * A * v^2"""
        speed = np.linalg.norm(vel)
        if speed < 1e-9:
            return np.zeros(3)
        drag_mag = 0.5 * self.rho * Cd * A * speed * speed
        return -drag_mag / m * (vel / speed)
    
    def simulate(self, v0, t, Cd=0.3, A=0.0001, m=0.01, substeps=100):
        """
        Simulate projectile motion.
        
        Args:
            v0: Initial velocity [vx, vy, vz] in m/s
            t: Total time in seconds
            Cd: Drag coefficient
            A: Cross-sectional area in m²
            m: Mass in kg
            substeps: Number of RK4 steps
        
        Returns:
            Position delta [dx, dy, dz] in meters
        """
        dt = t / substeps
        pos = np.zeros(3)
        vel = np.array(v0, dtype=np.float64)
        
        for _ in range(substeps):
            # RK4 integration
            k1v = np.array([0, -self.g, 0]) + self.drag_accel(vel, Cd, A, m)
            k1x = vel
            
            v2 = vel + 0.5 * dt * k1v
            k2v = np.array([0, -self.g, 0]) + self.drag_accel(v2, Cd, A, m)
            k2x = v2
            
            v3 = vel + 0.5 * dt * k2v
            k3v = np.array([0, -self.g, 0]) + self.drag_accel(v3, Cd, A, m)
            k3x = v3
            
            v4 = vel + dt * k3v
            k4v = np.array([0, -self.g, 0]) + self.drag_accel(v4, Cd, A, m)
            k4x = v4
            
            vel = vel + (dt / 6.0) * (k1v + 2*k2v + 2*k3v + k4v)
            pos = pos + (dt / 6.0) * (k1x + 2*k2x + 2*k3x + k4x)
        
        return pos

rk4 = RK4Engine(GRAVITY, AIR_DENSITY)
print("  ✓ RK4 Engine ready")

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════════════════════════════

# v15.5 FP16
class NetV155(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(15, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, 512), nn.LayerNorm(512), nn.SiLU(),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256, 256), nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256, 3)
        )
    def forward(self, x):
        return self.net(x)

v155_model = None
if v155_path:
    v155_model = NetV155()
    ckpt = torch.load(v155_path, map_location='cpu', weights_only=False)
    state = ckpt.get('model_state_dict', ckpt.get('state_dict', ckpt))
    v155_model.load_state_dict(state)
    v155_model = v155_model.to(device).half().eval()
    print("  ✓ v15.5 FP16 loaded")

# v16.0 INT8
class Int8Linear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.register_buffer('w_int8', torch.zeros(out_f, in_f, dtype=torch.int8))
        self.register_buffer('w_scale', torch.ones(out_f))
        self.register_buffer('bias', torch.zeros(out_f))
    
    def load(self, w, s, b):
        self.w_int8.copy_(torch.from_numpy(w))
        self.w_scale.copy_(torch.from_numpy(s.astype(np.float32)))
        self.bias.copy_(torch.from_numpy(b.astype(np.float32)))
    
    def forward(self, x):
        w = self.w_int8.float() * self.w_scale.unsqueeze(1)
        return nn.functional.linear(x, w, self.bias)

class NetV160(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        for i in range(len(dims)-2):
            self.linears.append(Int8Linear(dims[i], dims[i+1]))
            self.norms.append(nn.LayerNorm(dims[i+1]))
        self.output = Int8Linear(dims[-2], dims[-1])
    
    def forward(self, x):
        for lin, norm in zip(self.linears, self.norms):
            x = torch.nn.functional.silu(norm(lin(x)))
        return self.output(x)

v160_model = None
if v160_path:
    npz = np.load(v160_path)
    dims = []
    i = 0
    while f"layer_{i}_weight_int8" in npz.files:
        w = npz[f"layer_{i}_weight_int8"]
        if i == 0: dims.append(w.shape[1])
        dims.append(w.shape[0])
        i += 1
    dims.append(npz["output_weight_int8"].shape[0])
    
    v160_model = NetV160(dims)
    for i in range(len(dims)-2):
        v160_model.linears[i].load(
            npz[f"layer_{i}_weight_int8"],
            npz[f"layer_{i}_weight_scale"],
            npz[f"layer_{i}_bias"]
        )
        v160_model.norms[i].weight.data.copy_(torch.from_numpy(npz[f"layer_{i}_ln_weight"].astype(np.float32)))
        v160_model.norms[i].bias.data.copy_(torch.from_numpy(npz[f"layer_{i}_ln_bias"].astype(np.float32)))
    v160_model.output.load(npz["output_weight_int8"], npz["output_weight_scale"], npz["output_bias"])
    v160_model = v160_model.to(device).eval()
    print(f"  ✓ v16.0 INT8 loaded (arch: {dims})")

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATE TEST DATA
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n  Generating {NUM_SAMPLES:,} test scenarios...")

np.random.seed(42)

# Realistic ballistics parameters
speeds = np.random.uniform(300, 1000, NUM_SAMPLES)  # m/s (rifle range)
angles_h = np.random.uniform(-0.05, 0.05, NUM_SAMPLES)  # horizontal deviation
angles_v = np.random.uniform(-0.1, 0.1, NUM_SAMPLES)  # vertical angle

vx = speeds * np.cos(angles_v) * np.cos(angles_h)
vy = speeds * np.sin(angles_v)
vz = speeds * np.cos(angles_v) * np.sin(angles_h)

times = np.random.uniform(0.01, 0.5, NUM_SAMPLES)  # 10ms to 500ms flight time
drag_coefs = np.random.uniform(0.2, 0.5, NUM_SAMPLES)
masses = np.random.uniform(0.005, 0.02, NUM_SAMPLES)  # 5g to 20g bullets
areas = np.random.uniform(0.00005, 0.0001, NUM_SAMPLES)  # cross-section

print("  ✓ Test data generated")

# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE RK4 GROUND TRUTH
# ═══════════════════════════════════════════════════════════════════════════════

print(f"\n  Computing RK4 ground truth ({RK4_SUBSTEPS} substeps each)...")
start = time.time()

ground_truth = np.zeros((NUM_SAMPLES, 3))
for i in range(NUM_SAMPLES):
    ground_truth[i] = rk4.simulate(
        v0=[vx[i], vy[i], vz[i]],
        t=times[i],
        Cd=drag_coefs[i],
        A=areas[i],
        m=masses[i],
        substeps=RK4_SUBSTEPS
    )
    if (i + 1) % 2000 == 0:
        print(f"    {i+1}/{NUM_SAMPLES}...")

rk4_time = time.time() - start
print(f"  ✓ RK4 completed in {rk4_time:.1f}s ({NUM_SAMPLES/rk4_time:.0f} samples/sec)")

# ═══════════════════════════════════════════════════════════════════════════════
# ENCODE INPUTS FOR NEURAL NETS
# ═══════════════════════════════════════════════════════════════════════════════

# Must match training encoding exactly!
inputs = np.zeros((NUM_SAMPLES, 15), dtype=np.float32)

inputs[:, 0] = vx / 1000.0                          # vx normalized
inputs[:, 1] = vy / 1000.0                          # vy normalized
inputs[:, 2] = vz / 1000.0                          # vz normalized
inputs[:, 3] = times * 2.0                          # t normalized (t * 2)
inputs[:, 4] = speeds / 1000.0                      # muzzle velocity
inputs[:, 5] = angles_v                             # elevation angle
inputs[:, 6] = np.log(drag_coefs + 1e-6) / 2.0      # log drag
inputs[:, 7] = 1.0                                  # G1 drag model (one-hot)
inputs[:, 8] = 0.0                                  # G7
inputs[:, 9] = 0.0                                  # G5
inputs[:, 10] = 0.0                                 # linear
inputs[:, 11] = 0.0                                 # altitude (sea level)
inputs[:, 12] = (speeds > SPEED_OF_SOUND).astype(np.float32)  # supersonic flag
inputs[:, 13] = 0.5                                 # spin rate (normalized)
inputs[:, 14] = 320.0 / 512.0                       # barrel length (normalized)

x_tensor = torch.from_numpy(inputs).to(device)

# ═══════════════════════════════════════════════════════════════════════════════
# RUN NEURAL NETWORK PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n  Running neural network predictions...")

# v15.5
pred_v155 = None
if v155_model:
    with torch.no_grad():
        pred_v155 = v155_model(x_tensor.half()).float().cpu().numpy()
    print("  ✓ v15.5 predictions done")

# v16.0
pred_v160 = None
if v160_model:
    with torch.no_grad():
        pred_v160 = v160_model(x_tensor.float()).cpu().numpy()
    print("  ✓ v16.0 predictions done")

# ═══════════════════════════════════════════════════════════════════════════════
# CALCULATE ACCURACY METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def calc_metrics(pred, truth, name):
    """Calculate accuracy metrics vs ground truth."""
    errors_m = np.linalg.norm(pred - truth, axis=1)  # meters
    errors_mm = errors_m * 1000  # millimeters
    
    mae = np.mean(errors_mm)
    max_err = np.max(errors_mm)
    std_err = np.std(errors_mm)
    median_err = np.median(errors_mm)
    
    within_1mm = np.mean(errors_mm < 1) * 100
    within_3mm = np.mean(errors_mm < 3) * 100
    within_5mm = np.mean(errors_mm < 5) * 100
    within_10mm = np.mean(errors_mm < 10) * 100
    
    p50 = np.percentile(errors_mm, 50)
    p90 = np.percentile(errors_mm, 90)
    p95 = np.percentile(errors_mm, 95)
    p99 = np.percentile(errors_mm, 99)
    
    return {
        'name': name,
        'mae': mae,
        'max': max_err,
        'std': std_err,
        'median': median_err,
        'within_1mm': within_1mm,
        'within_3mm': within_3mm,
        'within_5mm': within_5mm,
        'within_10mm': within_10mm,
        'p50': p50,
        'p90': p90,
        'p95': p95,
        'p99': p99,
        'errors_mm': errors_mm
    }

print("\n" + "=" * 70)
print("  ACCURACY vs RK4 GROUND TRUTH")
print("=" * 70)

results = []

if pred_v155 is not None:
    m = calc_metrics(pred_v155, ground_truth, "v15.5 FP16")
    results.append(m)

if pred_v160 is not None:
    m = calc_metrics(pred_v160, ground_truth, "v16.0 INT8")
    results.append(m)

# Print table
print(f"\n  {'Metric':<25}", end="")
for r in results:
    print(f"{r['name']:<18}", end="")
print()
print("  " + "-" * (25 + 18 * len(results)))

print(f"  {'MAE (mm)':<25}", end="")
for r in results:
    print(f"{r['mae']:<18.2f}", end="")
print()

print(f"  {'Max Error (mm)':<25}", end="")
for r in results:
    print(f"{r['max']:<18.2f}", end="")
print()

print(f"  {'Std Dev (mm)':<25}", end="")
for r in results:
    print(f"{r['std']:<18.2f}", end="")
print()

print(f"  {'Median (mm)':<25}", end="")
for r in results:
    print(f"{r['median']:<18.2f}", end="")
print()

print("  " + "-" * (25 + 18 * len(results)))

print(f"  {'Within 1mm (%)':<25}", end="")
for r in results:
    print(f"{r['within_1mm']:<18.1f}", end="")
print()

print(f"  {'Within 3mm (%)':<25}", end="")
for r in results:
    print(f"{r['within_3mm']:<18.1f}", end="")
print()

print(f"  {'Within 5mm (%)':<25}", end="")
for r in results:
    print(f"{r['within_5mm']:<18.1f}", end="")
print()

print(f"  {'Within 10mm (%)':<25}", end="")
for r in results:
    print(f"{r['within_10mm']:<18.1f}", end="")
print()

print("  " + "-" * (25 + 18 * len(results)))

print(f"  {'P50 (mm)':<25}", end="")
for r in results:
    print(f"{r['p50']:<18.2f}", end="")
print()

print(f"  {'P90 (mm)':<25}", end="")
for r in results:
    print(f"{r['p90']:<18.2f}", end="")
print()

print(f"  {'P95 (mm)':<25}", end="")
for r in results:
    print(f"{r['p95']:<18.2f}", end="")
print()

print(f"  {'P99 (mm)':<25}", end="")
for r in results:
    print(f"{r['p99']:<18.2f}", end="")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# SPEED COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  SPEED: O(1) vs O(N)")
print("=" * 70)

# Neural net timing
if v160_model:
    torch.cuda.synchronize() if device == "cuda" else None
    start = time.perf_counter()
    for _ in range(100):
        with torch.no_grad():
            _ = v160_model(x_tensor.float())
    torch.cuda.synchronize() if device == "cuda" else None
    nn_time = (time.perf_counter() - start) / 100
    nn_throughput = NUM_SAMPLES / nn_time

print(f"""
  RK4 (O(N) with N={RK4_SUBSTEPS}):
    Time:       {rk4_time:.1f}s for {NUM_SAMPLES:,} samples
    Throughput: {NUM_SAMPLES/rk4_time:,.0f} samples/sec
  
  Neural Net (O(1)):
    Time:       {nn_time*1000:.2f}ms for {NUM_SAMPLES:,} samples
    Throughput: {nn_throughput:,.0f} samples/sec
  
  ═══════════════════════════════════════════════════════════════════════
  SPEEDUP:      {nn_throughput / (NUM_SAMPLES/rk4_time):,.0f}x faster
  ═══════════════════════════════════════════════════════════════════════
""")

# ═══════════════════════════════════════════════════════════════════════════════
# ERROR DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  ERROR DISTRIBUTION (v16.0 INT8)")
print("=" * 70)

if pred_v160 is not None:
    errors = results[-1]['errors_mm']
    
    buckets = [
        (0, 1, "0-1mm"),
        (1, 2, "1-2mm"),
        (2, 3, "2-3mm"),
        (3, 5, "3-5mm"),
        (5, 10, "5-10mm"),
        (10, 20, "10-20mm"),
        (20, 50, "20-50mm"),
        (50, float('inf'), ">50mm"),
    ]
    
    print(f"\n  {'Range':<15} {'Count':<12} {'Percentage':<12} {'Cumulative':<12}")
    print("  " + "-" * 50)
    
    cumulative = 0
    for lo, hi, label in buckets:
        if hi == float('inf'):
            count = np.sum(errors >= lo)
        else:
            count = np.sum((errors >= lo) & (errors < hi))
        pct = 100.0 * count / len(errors)
        cumulative += pct
        print(f"  {label:<15} {count:<12} {pct:<12.2f} {cumulative:<12.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)

if results:
    v160_mae = results[-1]['mae'] if pred_v160 is not None else 0
    v160_p99 = results[-1]['p99'] if pred_v160 is not None else 0
    v160_within = results[-1]['within_10mm'] if pred_v160 is not None else 0
    
    print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  HyperFlux v16.0 INT8 Performance                                   │
  ├─────────────────────────────────────────────────────────────────────┤
  │                                                                     │
  │  ACCURACY vs RK4 Ground Truth:                                      │
  │    • MAE:           {v160_mae:>8.2f} mm                                      │
  │    • P99:           {v160_p99:>8.2f} mm                                      │
  │    • Within 10mm:   {v160_within:>8.1f}%                                      │
  │                                                                     │
  │  SPEED:                                                             │
  │    • RK4 (O(N)):    {NUM_SAMPLES/rk4_time:>8.0f} samples/sec                          │
  │    • Neural (O(1)): {nn_throughput:>8.0f} samples/sec                          │
  │    • Speedup:       {nn_throughput/(NUM_SAMPLES/rk4_time):>8.0f}x                                       │
  │                                                                     │
  │  MODEL:                                                             │
  │    • Size:          0.72 MB (INT8 quantized)                        │
  │    • Architecture:  15 → 512 → 512 → 512 → 256 → 256 → 3            │
  │                                                                     │
  └─────────────────────────────────────────────────────────────────────┘
""")

print("=" * 70)
