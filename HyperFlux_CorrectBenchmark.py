"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         HYPERFLUX v16.0 INT8 vs v15.5 FP16 - CORRECT ENCODING                ║
║                                                                              ║
║  Uses the EXACT input encoding from v15.5 training script                    ║
║  Compares against TEACHER models (the actual ground truth for training)      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import glob

print("=" * 70)
print("  HYPERFLUX: Correct Encoding Benchmark")
print("  (Using exact v15.5 training encoding)")
print("=" * 70)

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

# ═══════════════════════════════════════════════════════════════════════════════
# TEACHER MODEL (v14.2 CfC - Ground Truth for distillation)
# ═══════════════════════════════════════════════════════════════════════════════

class CfCCell(nn.Module):
    """Exact copy of v14.2 CfCCell."""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.f_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Softplus()
        )
        self.g_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        self.h_net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x, t):
        features = self.backbone(x)
        f = self.f_net(features) + 0.1
        g = self.g_net(features)
        h = self.h_net(features)
        gate = torch.sigmoid(-f * t)
        return gate * g + (1 - gate) * h


class OracleKernel(nn.Module):
    """Exact copy of v14.2 OracleKernel."""
    def __init__(self, hidden_dim: int = 320):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cfc = CfCCell(input_dim=4, hidden_dim=hidden_dim, output_dim=3)
        self.register_buffer('v0_mean', torch.zeros(3))
        self.register_buffer('v0_std', torch.ones(3))
        self.register_buffer('pos_mean', torch.zeros(3))
        self.register_buffer('pos_std', torch.ones(3))
        self.t_max = 1.0

    def forward(self, v0, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        v0_norm = (v0 - self.v0_mean) / self.v0_std
        t_norm = t / self.t_max
        x = torch.cat([v0_norm, t_norm], dim=-1)
        raw_pos = self.cfc(x, t_norm)
        return raw_pos * self.pos_std + self.pos_mean

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD TEACHERS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n  Loading teacher models...")
teacher_dir = "/content/drive/MyDrive/HyperFlux_Models"
bin_files = sorted(glob.glob(os.path.join(teacher_dir, "HyperFlux_*.bin")))

teachers = {}
for bin_path in bin_files:
    filename = os.path.basename(bin_path)
    parts = filename.replace("HyperFlux_", "").replace(".bin", "").rsplit("_S", 1)
    if len(parts) != 2:
        continue
    
    weapon_name = parts[0]
    try:
        segment = int(parts[1])
    except:
        continue
    
    try:
        ckpt = torch.load(bin_path, map_location=device, weights_only=False)
        hidden_dim = ckpt.get('hidden_dim', 320)
        t_max = ckpt.get('t_max', 1.0)
        weapon_info = ckpt.get('weapon', {})
        
        model = OracleKernel(hidden_dim=hidden_dim)
        model.load_state_dict(ckpt['model_state'])
        model.t_max = t_max
        model.to(device).eval()
        
        if weapon_name not in teachers:
            teachers[weapon_name] = {}
        
        teachers[weapon_name][segment] = {
            'model': model,
            't_max': t_max,
            'hidden_dim': hidden_dim,
            'weapon': weapon_info,
        }
    except:
        pass

num_teachers = sum(len(s) for s in teachers.values())
print(f"  ✓ Loaded {num_teachers} teachers ({len(teachers)} weapons)")

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD STUDENT MODELS
# ═══════════════════════════════════════════════════════════════════════════════

# v15.5 FP16
v155_path = glob.glob("/content/drive/MyDrive/HyperFlux_v15.5_Unified_*/unified_ballistics_fp16.pt")[0]
v155_ckpt = torch.load(v155_path, map_location='cpu', weights_only=False)

class StudentNet(nn.Module):
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

v155_model = StudentNet()
v155_model.load_state_dict(v155_ckpt['model_state_dict'])
v155_model = v155_model.to(device).half().eval()
print(f"  ✓ v15.5 FP16 loaded")

# v16.0 INT8
v160_path = glob.glob("/content/drive/MyDrive/HyperFlux_v16.0_INT8_*/int8_weights.npz")[0]

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
        return F.linear(x, w, self.bias)

class StudentNetINT8(nn.Module):
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
            x = F.silu(norm(lin(x)))
        return self.output(x)

npz = np.load(v160_path)
dims = []
i = 0
while f"layer_{i}_weight_int8" in npz.files:
    w = npz[f"layer_{i}_weight_int8"]
    if i == 0: dims.append(w.shape[1])
    dims.append(w.shape[0])
    i += 1
dims.append(npz["output_weight_int8"].shape[0])

v160_model = StudentNetINT8(dims)
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
print(f"  ✓ v16.0 INT8 loaded")

# ═══════════════════════════════════════════════════════════════════════════════
# DRAG ENCODING (EXACT from training script)
# ═══════════════════════════════════════════════════════════════════════════════

def get_drag_encoding(drag_type: str):
    """EXACT encoding from v15.5 training - order: G7, G1, SPHERE, ROCKET"""
    if drag_type == "G7": return [1, 0, 0, 0]
    elif drag_type == "G1": return [0, 1, 0, 0]
    elif drag_type == "SPHERE": return [0, 0, 1, 0]
    elif drag_type == "ROCKET": return [0, 0, 0, 1]
    return [1, 0, 0, 0]

# ═══════════════════════════════════════════════════════════════════════════════
# GENERATE TEST DATA (EXACT same as training)
# ═══════════════════════════════════════════════════════════════════════════════

NUM_SAMPLES_PER_SEGMENT = 500  # Smaller for faster test
np.random.seed(42)

print(f"\n  Generating test data ({NUM_SAMPLES_PER_SEGMENT} per segment)...")

all_inputs = []
all_teacher_outputs = []

for weapon_name, segments in teachers.items():
    for seg_id, seg_data in segments.items():
        model = seg_data['model']
        t_max = seg_data['t_max']
        hidden_dim = seg_data['hidden_dim']
        weapon_info = seg_data.get('weapon', {})
        
        muzzle_vel = weapon_info.get('muzzle_velocity', 800.0)
        bc = weapon_info.get('bc', 0.15)
        bullet_mass = weapon_info.get('bullet_mass', 0.01)
        drag_type = weapon_info.get('drag_type', 'G7')
        effective_range = weapon_info.get('effective_range', 500.0)
        has_thrust = weapon_info.get('has_thrust', False)
        
        n = NUM_SAMPLES_PER_SEGMENT
        
        # Generate velocities (EXACT same as training)
        azimuth = torch.rand(n, device=device) * 2 * 3.14159
        elevation = (torch.rand(n, device=device) - 0.3) * 0.6
        
        dx = torch.cos(elevation) * torch.cos(azimuth)
        dy = torch.sin(elevation)
        dz = torch.cos(elevation) * torch.sin(azimuth)
        direction = F.normalize(torch.stack([dx, dy, dz], dim=1), dim=1)
        
        v0 = direction * muzzle_vel
        t = torch.rand(n, device=device) * t_max * 0.95 + t_max * 0.05
        
        # Query teacher
        with torch.no_grad():
            teacher_output = model(v0, t)
        
        # Create student input (EXACT same encoding as training)
        v0_norm = v0 / 1000.0
        t_norm = t.unsqueeze(-1) / 10.0
        
        drag_enc = get_drag_encoding(drag_type)
        
        weapon_features = torch.tensor([
            muzzle_vel / 1000.0,
            bc / 0.5,
            np.log10(bullet_mass + 1e-6) / 2 + 1,
            drag_enc[0], drag_enc[1], drag_enc[2], drag_enc[3],
            effective_range / 2000.0,
            float(has_thrust),
            t_max / 5.0,
            hidden_dim / 512.0
        ], device=device).unsqueeze(0).expand(n, -1)
        
        student_input = torch.cat([v0_norm, t_norm, weapon_features], dim=1)
        
        all_inputs.append(student_input.cpu())
        all_teacher_outputs.append(teacher_output.cpu())

inputs = torch.cat(all_inputs).float()
teacher_outputs = torch.cat(all_teacher_outputs).float()

# Shuffle
perm = torch.randperm(len(inputs))
inputs = inputs[perm]
teacher_outputs = teacher_outputs[perm]

print(f"  ✓ Generated {len(inputs):,} test samples")

# ═══════════════════════════════════════════════════════════════════════════════
# RUN PREDICTIONS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n  Running predictions...")

x = inputs.to(device)
y_true = teacher_outputs.to(device)

with torch.no_grad():
    # v15.5 FP16
    pred_v155 = v155_model(x.half()).float()
    
    # v16.0 INT8
    pred_v160 = v160_model(x.float())

print("  ✓ Predictions done")

# ═══════════════════════════════════════════════════════════════════════════════
# CALCULATE METRICS
# ═══════════════════════════════════════════════════════════════════════════════

def calc_metrics(pred, truth):
    errors_mm = (pred - truth).norm(dim=1).cpu().numpy() * 1000
    return {
        'mae': np.mean(errors_mm),
        'max': np.max(errors_mm),
        'std': np.std(errors_mm),
        'median': np.median(errors_mm),
        'within_1mm': np.mean(errors_mm < 1) * 100,
        'within_3mm': np.mean(errors_mm < 3) * 100,
        'within_5mm': np.mean(errors_mm < 5) * 100,
        'within_10mm': np.mean(errors_mm < 10) * 100,
        'p90': np.percentile(errors_mm, 90),
        'p95': np.percentile(errors_mm, 95),
        'p99': np.percentile(errors_mm, 99),
    }

r155 = calc_metrics(pred_v155, y_true)
r160 = calc_metrics(pred_v160, y_true)

# ═══════════════════════════════════════════════════════════════════════════════
# PRINT RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  ACCURACY vs TEACHER (Ground Truth for Training)")
print("=" * 70)

print(f"\n  {'Metric':<25} {'v15.5 FP16':<18} {'v16.0 INT8':<18}")
print("  " + "-" * 60)
print(f"  {'MAE (mm)':<25} {r155['mae']:<18.2f} {r160['mae']:<18.2f}")
print(f"  {'Max Error (mm)':<25} {r155['max']:<18.2f} {r160['max']:<18.2f}")
print(f"  {'Median (mm)':<25} {r155['median']:<18.2f} {r160['median']:<18.2f}")
print(f"  {'Std Dev (mm)':<25} {r155['std']:<18.2f} {r160['std']:<18.2f}")
print("  " + "-" * 60)
print(f"  {'Within 1mm (%)':<25} {r155['within_1mm']:<18.1f} {r160['within_1mm']:<18.1f}")
print(f"  {'Within 3mm (%)':<25} {r155['within_3mm']:<18.1f} {r160['within_3mm']:<18.1f}")
print(f"  {'Within 5mm (%)':<25} {r155['within_5mm']:<18.1f} {r160['within_5mm']:<18.1f}")
print(f"  {'Within 10mm (%)':<25} {r155['within_10mm']:<18.1f} {r160['within_10mm']:<18.1f}")
print("  " + "-" * 60)
print(f"  {'P90 (mm)':<25} {r155['p90']:<18.2f} {r160['p90']:<18.2f}")
print(f"  {'P95 (mm)':<25} {r155['p95']:<18.2f} {r160['p95']:<18.2f}")
print(f"  {'P99 (mm)':<25} {r155['p99']:<18.2f} {r160['p99']:<18.2f}")

# ═══════════════════════════════════════════════════════════════════════════════
# SPEED COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  SPEED COMPARISON")
print("=" * 70)

# Warmup
for _ in range(10):
    with torch.no_grad():
        _ = v155_model(x.half())
        _ = v160_model(x.float())

torch.cuda.synchronize() if device == "cuda" else None

# v15.5
start = time.perf_counter()
for _ in range(100):
    with torch.no_grad():
        _ = v155_model(x.half())
torch.cuda.synchronize() if device == "cuda" else None
t155 = (time.perf_counter() - start) / 100

# v16.0
start = time.perf_counter()
for _ in range(100):
    with torch.no_grad():
        _ = v160_model(x.float())
torch.cuda.synchronize() if device == "cuda" else None
t160 = (time.perf_counter() - start) / 100

# Teacher (sample one)
sample_teacher = list(list(teachers.values())[0].values())[0]['model']
test_v0 = torch.randn(1000, 3, device=device) * 500
test_t = torch.rand(1000, device=device)

torch.cuda.synchronize() if device == "cuda" else None
start = time.perf_counter()
for _ in range(100):
    with torch.no_grad():
        _ = sample_teacher(test_v0, test_t)
torch.cuda.synchronize() if device == "cuda" else None
t_teacher = (time.perf_counter() - start) / 100

print(f"""
  Teacher (CfC):     {1000/t_teacher:>12,.0f} samples/sec
  v15.5 FP16:        {len(x)/t155:>12,.0f} samples/sec
  v16.0 INT8:        {len(x)/t160:>12,.0f} samples/sec
  
  Speedup vs Teacher:
    v15.5 FP16:      {(len(x)/t155) / (1000/t_teacher):>8.1f}x
    v16.0 INT8:      {(len(x)/t160) / (1000/t_teacher):>8.1f}x
""")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL-TO-MODEL COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  v16.0 INT8 vs v15.5 FP16 (Direct Comparison)")
print("=" * 70)

diff = (pred_v160 - pred_v155).norm(dim=1).cpu().numpy() * 1000
print(f"""
  Mean difference:   {np.mean(diff):.4f} mm
  Max difference:    {np.max(diff):.4f} mm
  Std difference:    {np.std(diff):.4f} mm
""")

# ═══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("=" * 70)
print("  FINAL SUMMARY")
print("=" * 70)

expected_mae = v155_ckpt.get('best_mae_mm', 3.06)

print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │  Expected MAE (from training):  {expected_mae:.2f} mm                         │
  │                                                                     │
  │  Measured MAE vs Teacher:                                           │
  │    • v15.5 FP16:  {r155['mae']:>8.2f} mm                                     │
  │    • v16.0 INT8:  {r160['mae']:>8.2f} mm                                     │
  │                                                                     │
  │  INT8 vs FP16 difference: {np.mean(diff):.4f} mm                            │
  │                                                                     │
  │  Model Size:                                                        │
  │    • v15.5 FP16:  1.44 MB                                           │
  │    • v16.0 INT8:  0.72 MB (2x smaller)                              │
  └─────────────────────────────────────────────────────────────────────┘
""")

if r155['mae'] < 5 and r160['mae'] < 5:
    print("  ✅ Both models working correctly!")
    print(f"  ✅ INT8 quantization successful - {r160['mae']:.2f}mm vs {r155['mae']:.2f}mm")
else:
    print("  ❌ Results don't match expected ~3mm MAE")

print("\n" + "=" * 70)
