"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           HYPERFLUX v16.0 INT8 vs v15.5 FP16 vs RK4 BENCHMARK                ║
║                                                                              ║
║  Comprehensive comparison of:                                                ║
║  - v16.0 INT8 (new quantized model)                                          ║
║  - v15.5 FP16 (previous baseline)                                            ║
║  - RK4 Ground Truth (physics simulation)                                     ║
║                                                                              ║
║  Metrics:                                                                    ║
║  - Accuracy: MAE, Max Error, % within tolerance                              ║
║  - Speed: Inferences/sec, Latency                                            ║
║  - Memory: Model size, Runtime memory                                        ║
║  - Complexity: Parameters, FLOPs                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os
import glob
from dataclasses import dataclass
from typing import Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("  HYPERFLUX BENCHMARK: v16.0 INT8 vs v15.5 FP16 vs RK4")
print("=" * 70)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkConfig:
    # Test parameters
    num_test_samples: int = 10000
    num_timing_iterations: int = 1000
    batch_sizes: list = None
    
    # Accuracy tolerances
    tolerance_mm: float = 10.0  # 10mm tolerance for "within spec"
    
    # Physics parameters
    gravity: float = 9.81
    air_density: float = 1.225
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    def __post_init__(self):
        if self.batch_sizes is None:
            self.batch_sizes = [1, 8, 32, 128, 512]

config = BenchmarkConfig()
print(f"\n  Device: {config.device}")
print(f"  Test samples: {config.num_test_samples:,}")

# ═══════════════════════════════════════════════════════════════════════════════
# MOUNT DRIVE & FIND MODELS
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from google.colab import drive
    drive.mount('/content/drive')
except:
    pass

# Find v15.5 model
v155_patterns = [
    "/content/drive/MyDrive/HyperFlux_v15.5_Unified_*/unified_ballistics_fp16.pt",
    "/content/drive/MyDrive/HyperFlux_v15.5_*/unified_ballistics_fp16.pt",
]

v155_path = None
for pattern in v155_patterns:
    matches = glob.glob(pattern)
    if matches:
        matches.sort(key=os.path.getmtime, reverse=True)
        v155_path = matches[0]
        break

# Find v16.0 INT8 weights
v160_patterns = [
    "/content/drive/MyDrive/HyperFlux_v16.0_INT8_*/int8_weights.npz",
    "/content/drive/MyDrive/HyperFlux_v16_INT8_*/int8_weights.npz",
]

v160_path = None
for pattern in v160_patterns:
    matches = glob.glob(pattern)
    if matches:
        matches.sort(key=os.path.getmtime, reverse=True)
        v160_path = matches[0]
        break

print(f"\n  v15.5 FP16: {v155_path}")
print(f"  v16.0 INT8: {v160_path}")

if v155_path is None:
    print("  WARNING: v15.5 model not found!")
if v160_path is None:
    print("  WARNING: v16.0 INT8 weights not found!")

# ═══════════════════════════════════════════════════════════════════════════════
# RK4 GROUND TRUTH ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class RK4BallisticsEngine:
    """High-fidelity RK4 physics simulation for ground truth."""
    
    def __init__(self, gravity=9.81, air_density=1.225):
        self.gravity = gravity
        self.air_density = air_density
    
    def compute_drag_acceleration(self, velocity: np.ndarray, 
                                   drag_coef: float, 
                                   cross_section: float,
                                   mass: float) -> np.ndarray:
        """Compute drag acceleration using quadratic drag model."""
        speed = np.linalg.norm(velocity)
        if speed < 1e-6:
            return np.zeros(3)
        
        # F_drag = 0.5 * rho * Cd * A * v^2
        drag_magnitude = 0.5 * self.air_density * drag_coef * cross_section * speed * speed
        drag_acceleration = -drag_magnitude / mass * (velocity / speed)
        return drag_acceleration
    
    def derivatives(self, state: np.ndarray, 
                    drag_coef: float, 
                    cross_section: float, 
                    mass: float,
                    wind: np.ndarray) -> np.ndarray:
        """Compute state derivatives [vx, vy, vz, ax, ay, az]."""
        velocity = state[:3] - wind  # Relative velocity
        
        # Gravity
        gravity_acc = np.array([0, -self.gravity, 0])
        
        # Drag
        drag_acc = self.compute_drag_acceleration(velocity, drag_coef, cross_section, mass)
        
        acceleration = gravity_acc + drag_acc
        
        return np.concatenate([state[:3], acceleration])
    
    def step_rk4(self, position: np.ndarray, velocity: np.ndarray,
                 dt: float, drag_coef: float, cross_section: float,
                 mass: float, wind: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Single RK4 integration step."""
        state = np.concatenate([velocity, position])
        
        k1 = self.derivatives(state, drag_coef, cross_section, mass, wind)
        k2 = self.derivatives(state + 0.5 * dt * k1, drag_coef, cross_section, mass, wind)
        k3 = self.derivatives(state + 0.5 * dt * k2, drag_coef, cross_section, mass, wind)
        k4 = self.derivatives(state + dt * k3, drag_coef, cross_section, mass, wind)
        
        new_state = state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
        
        new_velocity = new_state[:3]
        new_position = position + (dt / 6.0) * (state[:3] + 2*k2[:3] + 2*k3[:3] + k4[:3])
        
        return new_position, new_velocity
    
    def simulate(self, velocity: np.ndarray, t: float,
                 drag_coef: float = 0.3, 
                 cross_section: float = 0.0001,
                 mass: float = 0.01,
                 wind: np.ndarray = None,
                 substeps: int = 100) -> np.ndarray:
        """Simulate trajectory and return position delta."""
        if wind is None:
            wind = np.zeros(3)
        
        position = np.zeros(3)
        vel = velocity.copy()
        
        dt = t / substeps
        for _ in range(substeps):
            position, vel = self.step_rk4(position, vel, dt, drag_coef, 
                                          cross_section, mass, wind)
        
        return position
    
    def simulate_batch(self, velocities: np.ndarray, times: np.ndarray,
                       drag_coefs: np.ndarray, cross_sections: np.ndarray,
                       masses: np.ndarray, winds: np.ndarray,
                       substeps: int = 100) -> np.ndarray:
        """Batch simulate trajectories."""
        n = len(velocities)
        results = np.zeros((n, 3))
        
        for i in range(n):
            results[i] = self.simulate(
                velocities[i], times[i],
                drag_coefs[i], cross_sections[i], masses[i],
                winds[i], substeps
            )
        
        return results

rk4_engine = RK4BallisticsEngine(gravity=config.gravity, air_density=config.air_density)
print("\n  ✓ RK4 Engine initialized")

# ═══════════════════════════════════════════════════════════════════════════════
# v15.5 FP16 MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class BallisticsNetV155(nn.Module):
    """v15.5 architecture: 15 -> [512,512,512,256,256] -> 3"""
    
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

def load_v155_model(path: str, device: str) -> Optional[nn.Module]:
    """Load v15.5 model."""
    if path is None or not os.path.exists(path):
        return None
    
    model = BallisticsNetV155()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    model.load_state_dict(state_dict)
    model = model.to(device).half().eval()
    
    return model

v155_model = load_v155_model(v155_path, config.device)
if v155_model is not None:
    print("  ✓ v15.5 FP16 model loaded")

# ═══════════════════════════════════════════════════════════════════════════════
# v16.0 INT8 MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class Int8Linear(nn.Module):
    """INT8 quantized linear layer with FP32 computation."""
    
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # INT8 weights + per-channel scales
        self.register_buffer('weight_int8', torch.zeros(out_features, in_features, dtype=torch.int8))
        self.register_buffer('weight_scale', torch.ones(out_features))
        self.register_buffer('bias', torch.zeros(out_features))
    
    def load_weights(self, weight_int8: np.ndarray, weight_scale: np.ndarray, bias: np.ndarray):
        self.weight_int8.copy_(torch.from_numpy(weight_int8))
        self.weight_scale.copy_(torch.from_numpy(weight_scale.astype(np.float32)))
        self.bias.copy_(torch.from_numpy(bias.astype(np.float32)))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Dequantize: W_fp32 = W_int8 * scale
        weight_fp32 = self.weight_int8.float() * self.weight_scale.unsqueeze(1)
        return nn.functional.linear(x, weight_fp32, self.bias)


class BallisticsNetV160INT8(nn.Module):
    """v16.0 INT8 architecture - auto-detects dimensions from weights."""
    
    def __init__(self, dims: list):
        super().__init__()
        
        self.input_dim = dims[0]
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(len(dims) - 2):
            self.linears.append(Int8Linear(dims[i], dims[i+1]))
            self.norms.append(nn.LayerNorm(dims[i+1]))
        
        # Output layer (no norm)
        self.output_linear = Int8Linear(dims[-2], dims[-1])
        self.activation = nn.SiLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for linear, norm in zip(self.linears, self.norms):
            x = linear(x)
            x = norm(x)
            x = self.activation(x)
        
        return self.output_linear(x)


def load_v160_model(path: str, device: str) -> Optional[nn.Module]:
    """Load v16.0 INT8 model from NPZ - auto-detects architecture."""
    if path is None or not os.path.exists(path):
        return None
    
    npz = np.load(path)
    
    # Auto-detect architecture from weight shapes
    dims = []
    i = 0
    while f"layer_{i}_weight_int8" in npz.files:
        w = npz[f"layer_{i}_weight_int8"]
        if i == 0:
            dims.append(w.shape[1])  # Input dim
        dims.append(w.shape[0])  # Output dim of this layer
        i += 1
    
    # Add output layer dimension
    out_w = npz["output_weight_int8"]
    dims.append(out_w.shape[0])
    
    print(f"    Auto-detected architecture: {dims}")
    
    model = BallisticsNetV160INT8(dims)
    
    # Load hidden layers
    num_hidden = len(dims) - 2
    for i in range(num_hidden):
        prefix = f"layer_{i}_"
        model.linears[i].load_weights(
            npz[f"{prefix}weight_int8"],
            npz[f"{prefix}weight_scale"],
            npz[f"{prefix}bias"]
        )
        
        # LayerNorm
        model.norms[i].weight.data.copy_(
            torch.from_numpy(npz[f"{prefix}ln_weight"].astype(np.float32))
        )
        model.norms[i].bias.data.copy_(
            torch.from_numpy(npz[f"{prefix}ln_bias"].astype(np.float32))
        )
    
    # Output layer
    model.output_linear.load_weights(
        npz["output_weight_int8"],
        npz["output_weight_scale"],
        npz["output_bias"]
    )
    
    return model.to(device).eval()

v160_model = load_v160_model(v160_path, config.device)
if v160_model is not None:
    print("  ✓ v16.0 INT8 model loaded")

# ═══════════════════════════════════════════════════════════════════════════════
# TEST DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════

def generate_test_data(n: int, seed: int = 42) -> dict:
    """Generate test data with varied parameters."""
    np.random.seed(seed)
    
    # Velocities (m/s)
    speeds = np.random.uniform(200, 1200, n)
    angles_h = np.random.uniform(-0.1, 0.1, n)  # Horizontal deviation
    angles_v = np.random.uniform(-0.15, 0.15, n)  # Vertical angle
    
    vx = speeds * np.cos(angles_v) * np.cos(angles_h)
    vy = speeds * np.sin(angles_v)
    vz = speeds * np.cos(angles_v) * np.sin(angles_h)
    
    velocities = np.stack([vx, vy, vz], axis=1)
    
    # Times (seconds)
    times = np.random.uniform(0.001, 2.0, n)
    
    # Projectile properties
    masses = np.random.uniform(0.004, 0.05, n)  # 4g to 50g
    drag_coefs = np.random.uniform(0.2, 0.5, n)
    cross_sections = np.random.uniform(0.00005, 0.0002, n)
    
    # Environment
    gravity_scales = np.random.uniform(0.9, 1.1, n)
    air_densities = np.random.uniform(0.9, 1.1, n)
    
    # Wind
    wind_speeds = np.random.uniform(0, 10, n)
    wind_dirs = np.random.uniform(0, 2*np.pi, n)
    winds = np.stack([
        wind_speeds * np.cos(wind_dirs),
        np.zeros(n),
        wind_speeds * np.sin(wind_dirs)
    ], axis=1)
    
    # Spin (optional features)
    spin_rates = np.random.uniform(0, 0.5, n)
    spin_axes = np.random.uniform(0, 2*np.pi, n)
    
    return {
        'velocities': velocities,
        'times': times,
        'masses': masses,
        'drag_coefs': drag_coefs,
        'cross_sections': cross_sections,
        'gravity_scales': gravity_scales,
        'air_densities': air_densities,
        'winds': winds,
        'spin_rates': spin_rates,
        'spin_axes': spin_axes,
        'n': n
    }

print(f"\n  Generating {config.num_test_samples:,} test samples...")
test_data = generate_test_data(config.num_test_samples)
print("  ✓ Test data generated")

# ═══════════════════════════════════════════════════════════════════════════════
# INPUT ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

def encode_input_v155(data: dict, idx: int) -> np.ndarray:
    """Encode input for v15.5 (15 features)."""
    vel = data['velocities'][idx]
    speed = np.linalg.norm(vel)
    is_supersonic = 1.0 if speed > 343.0 else 0.0
    drag_model = 0  # G1
    
    return np.array([
        vel[0] / 1000.0,
        vel[1] / 1000.0,
        vel[2] / 1000.0,
        data['times'][idx] * 2.0,
        speed / 1000.0,  # muzzle_velocity normalized
        0.0,  # elevation_angle
        np.log(data['drag_coefs'][idx] + 1e-6) / 2.0,
        1.0 if drag_model == 0 else 0.0,  # G1
        1.0 if drag_model == 1 else 0.0,  # G7
        1.0 if drag_model == 2 else 0.0,  # G5
        1.0 if drag_model == 3 else 0.0,  # linear
        0.0,  # altitude
        is_supersonic,
        data['spin_rates'][idx] / 5.0,
        320.0 / 512.0,  # barrel_length
    ], dtype=np.float32)


def encode_input_v160(data: dict, idx: int, input_dim: int = 15) -> np.ndarray:
    """Encode input for v16.0 - matches training format."""
    vel = data['velocities'][idx]
    speed = np.linalg.norm(vel)
    is_supersonic = 1.0 if speed > 343.0 else 0.0
    drag_model = 0  # G1
    
    if input_dim == 15:
        # Same encoding as v15.5
        return np.array([
            vel[0] / 1000.0,
            vel[1] / 1000.0,
            vel[2] / 1000.0,
            data['times'][idx] * 2.0,
            speed / 1000.0,  # muzzle_velocity normalized
            0.0,  # elevation_angle
            np.log(data['drag_coefs'][idx] + 1e-6) / 2.0,
            1.0 if drag_model == 0 else 0.0,  # G1
            1.0 if drag_model == 1 else 0.0,  # G7
            1.0 if drag_model == 2 else 0.0,  # G5
            1.0 if drag_model == 3 else 0.0,  # linear
            0.0,  # altitude
            is_supersonic,
            data['spin_rates'][idx] / 5.0,
            320.0 / 512.0,  # barrel_length
        ], dtype=np.float32)
    else:
        # 14-feature encoding
        wind = data['winds'][idx]
        return np.array([
            vel[0] / 1000.0,
            vel[1] / 1000.0,
            vel[2] / 1000.0,
            data['times'][idx] / 10.0,
            data['gravity_scales'][idx],
            data['air_densities'][idx],
            wind[0] / 1000.0,
            wind[1] / 1000.0,
            wind[2] / 1000.0,
            data['masses'][idx],
            data['drag_coefs'][idx],
            data['cross_sections'][idx],
            data['spin_rates'][idx],
            data['spin_axes'][idx],
        ], dtype=np.float32)


def encode_batch_v155(data: dict) -> np.ndarray:
    """Encode full batch for v15.5."""
    return np.stack([encode_input_v155(data, i) for i in range(data['n'])])


def encode_batch_v160(data: dict, input_dim: int = 15) -> np.ndarray:
    """Encode full batch for v16.0."""
    return np.stack([encode_input_v160(data, i, input_dim) for i in range(data['n'])])

# ═══════════════════════════════════════════════════════════════════════════════
# COMPUTE GROUND TRUTH
# ═══════════════════════════════════════════════════════════════════════════════

print("\n  Computing RK4 ground truth...")
start_time = time.time()

ground_truth = rk4_engine.simulate_batch(
    test_data['velocities'],
    test_data['times'],
    test_data['drag_coefs'],
    test_data['cross_sections'],
    test_data['masses'],
    test_data['winds'],
    substeps=100
)

rk4_time = time.time() - start_time
print(f"  ✓ RK4 completed in {rk4_time:.2f}s ({config.num_test_samples / rk4_time:.0f} samples/sec)")

# ═══════════════════════════════════════════════════════════════════════════════
# BENCHMARK FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class BenchmarkResults:
    name: str
    mae_mm: float
    max_error_mm: float
    within_tolerance_pct: float
    inferences_per_sec: float
    latency_us: float
    model_size_mb: float
    num_parameters: int
    flops_per_inference: int


def compute_accuracy_metrics(predictions: np.ndarray, 
                            ground_truth: np.ndarray,
                            tolerance_mm: float) -> Tuple[float, float, float]:
    """Compute accuracy metrics."""
    # Convert to mm
    errors = np.linalg.norm(predictions - ground_truth, axis=1) * 1000.0
    
    mae = np.mean(errors)
    max_error = np.max(errors)
    within_tol = np.mean(errors <= tolerance_mm) * 100.0
    
    return mae, max_error, within_tol


def count_parameters(model: nn.Module) -> int:
    """Count model parameters."""
    return sum(p.numel() for p in model.parameters())


def estimate_flops(model_name: str) -> int:
    """Estimate FLOPs per inference."""
    # Architecture: 14/15 -> 512 -> 512 -> 512 -> 256 -> 256 -> 3
    # FLOPs ≈ 2 * (in * out) for each linear layer
    
    if "v155" in model_name.lower() or "15.5" in model_name:
        # 15 -> 512 -> 512 -> 512 -> 256 -> 256 -> 3
        flops = 2 * (15*512 + 512*512 + 512*512 + 512*256 + 256*256 + 256*3)
    else:
        # 14 -> 512 -> 512 -> 512 -> 256 -> 256 -> 3
        flops = 2 * (14*512 + 512*512 + 512*512 + 512*256 + 256*256 + 256*3)
    
    # Add LayerNorm (approximate: 5 ops per element)
    flops += 5 * (512 + 512 + 512 + 256 + 256)
    
    # Add SiLU (approximate: 4 ops per element)
    flops += 4 * (512 + 512 + 512 + 256 + 256)
    
    return flops


def benchmark_model(model: nn.Module, 
                   input_batch: np.ndarray,
                   ground_truth: np.ndarray,
                   name: str,
                   model_path: str,
                   device: str,
                   num_timing_iters: int,
                   tolerance_mm: float) -> BenchmarkResults:
    """Run full benchmark on a model."""
    
    # Convert input to tensor
    x = torch.from_numpy(input_batch).to(device)
    if "v155" in name.lower() or "15.5" in name:
        x = x.half()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(x)
    
    # Accuracy
    with torch.no_grad():
        predictions = model(x).cpu().numpy()
        if predictions.dtype == np.float16:
            predictions = predictions.astype(np.float32)
    
    mae, max_err, within_tol = compute_accuracy_metrics(predictions, ground_truth, tolerance_mm)
    
    # Timing (single sample)
    x_single = x[:1]
    if device == "cuda":
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_timing_iters):
            _ = model(x_single)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    latency_us = (elapsed / num_timing_iters) * 1e6
    
    # Timing (batch)
    if device == "cuda":
        torch.cuda.synchronize()
    
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(100):
            _ = model(x)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    batch_elapsed = time.perf_counter() - start
    inferences_per_sec = (100 * len(x)) / batch_elapsed
    
    # Model size
    if model_path and os.path.exists(model_path):
        model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    else:
        # Estimate from parameters
        num_params = count_parameters(model)
        bytes_per_param = 1 if "int8" in name.lower() else 2  # INT8 or FP16
        model_size_mb = (num_params * bytes_per_param) / (1024 * 1024)
    
    return BenchmarkResults(
        name=name,
        mae_mm=mae,
        max_error_mm=max_err,
        within_tolerance_pct=within_tol,
        inferences_per_sec=inferences_per_sec,
        latency_us=latency_us,
        model_size_mb=model_size_mb,
        num_parameters=count_parameters(model),
        flops_per_inference=estimate_flops(name)
    )

# ═══════════════════════════════════════════════════════════════════════════════
# RUN BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  RUNNING BENCHMARKS")
print("=" * 70)

results = []

# RK4 Baseline (for reference)
rk4_results = BenchmarkResults(
    name="RK4 Ground Truth",
    mae_mm=0.0,
    max_error_mm=0.0,
    within_tolerance_pct=100.0,
    inferences_per_sec=config.num_test_samples / rk4_time,
    latency_us=(rk4_time / config.num_test_samples) * 1e6,
    model_size_mb=0.0,
    num_parameters=0,
    flops_per_inference=100 * 50  # ~50 ops per substep, 100 substeps
)
results.append(rk4_results)

# v15.5 FP16
if v155_model is not None:
    print("\n  Benchmarking v15.5 FP16...")
    input_v155 = encode_batch_v155(test_data)
    
    v155_results = benchmark_model(
        v155_model, input_v155, ground_truth,
        "v15.5 FP16", v155_path, config.device,
        config.num_timing_iterations, config.tolerance_mm
    )
    results.append(v155_results)
    print(f"    MAE: {v155_results.mae_mm:.2f}mm")

# v16.0 INT8
if v160_model is not None:
    print("\n  Benchmarking v16.0 INT8...")
    v160_input_dim = v160_model.input_dim
    input_v160 = encode_batch_v160(test_data, v160_input_dim)
    
    v160_results = benchmark_model(
        v160_model, input_v160, ground_truth,
        "v16.0 INT8", v160_path, config.device,
        config.num_timing_iterations, config.tolerance_mm
    )
    results.append(v160_results)
    print(f"    MAE: {v160_results.mae_mm:.2f}mm")

# ═══════════════════════════════════════════════════════════════════════════════
# RESULTS TABLE
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  BENCHMARK RESULTS")
print("=" * 70)

# Print table header
print(f"\n{'Metric':<25} ", end="")
for r in results:
    print(f"{r.name:<18} ", end="")
print()
print("-" * (25 + 19 * len(results)))

# Accuracy
print(f"{'MAE (mm)':<25} ", end="")
for r in results:
    print(f"{r.mae_mm:<18.2f} ", end="")
print()

print(f"{'Max Error (mm)':<25} ", end="")
for r in results:
    print(f"{r.max_error_mm:<18.2f} ", end="")
print()

print(f"{'Within 10mm (%)':<25} ", end="")
for r in results:
    print(f"{r.within_tolerance_pct:<18.1f} ", end="")
print()

print("-" * (25 + 19 * len(results)))

# Speed
print(f"{'Inferences/sec':<25} ", end="")
for r in results:
    if r.inferences_per_sec > 1e6:
        print(f"{r.inferences_per_sec/1e6:<18.1f}M", end="")
    else:
        print(f"{r.inferences_per_sec:<18,.0f} ", end="")
print()

print(f"{'Latency (µs)':<25} ", end="")
for r in results:
    print(f"{r.latency_us:<18.1f} ", end="")
print()

print("-" * (25 + 19 * len(results)))

# Size
print(f"{'Model Size (MB)':<25} ", end="")
for r in results:
    print(f"{r.model_size_mb:<18.2f} ", end="")
print()

print(f"{'Parameters':<25} ", end="")
for r in results:
    if r.num_parameters > 0:
        print(f"{r.num_parameters:<18,} ", end="")
    else:
        print(f"{'N/A':<18} ", end="")
print()

print(f"{'FLOPs/inference':<25} ", end="")
for r in results:
    if r.flops_per_inference > 0:
        print(f"{r.flops_per_inference:<18,} ", end="")
    else:
        print(f"{'~5,000':<18} ", end="")
print()

# ═══════════════════════════════════════════════════════════════════════════════
# COMPARISON SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  COMPARISON SUMMARY")
print("=" * 70)

if v155_model is not None and v160_model is not None:
    v155 = results[1]
    v160 = results[2]
    
    print(f"""
  v16.0 INT8 vs v15.5 FP16:
  ─────────────────────────────────────────────────────────────────────
  Accuracy:
    • MAE:           {v160.mae_mm:.2f}mm vs {v155.mae_mm:.2f}mm  ({'+' if v160.mae_mm > v155.mae_mm else ''}{v160.mae_mm - v155.mae_mm:.2f}mm)
    • Max Error:     {v160.max_error_mm:.2f}mm vs {v155.max_error_mm:.2f}mm
    • Within 10mm:   {v160.within_tolerance_pct:.1f}% vs {v155.within_tolerance_pct:.1f}%
  
  Speed:
    • Throughput:    {v160.inferences_per_sec/1e6:.1f}M vs {v155.inferences_per_sec/1e6:.1f}M inf/sec
    • Latency:       {v160.latency_us:.1f}µs vs {v155.latency_us:.1f}µs
    
  Size:
    • Model:         {v160.model_size_mb:.2f} MB vs {v155.model_size_mb:.2f} MB ({v155.model_size_mb/v160.model_size_mb:.1f}x smaller)
    • Parameters:    {v160.num_parameters:,} (same architecture)
  
  v16.0 INT8 vs RK4 Ground Truth:
  ─────────────────────────────────────────────────────────────────────
    • Speedup:       {v160.inferences_per_sec / rk4_results.inferences_per_sec:,.0f}x faster
    • MAE:           {v160.mae_mm:.2f}mm
""")

elif v160_model is not None:
    v160 = results[1]
    print(f"""
  v16.0 INT8 vs RK4 Ground Truth:
  ─────────────────────────────────────────────────────────────────────
    • Speedup:       {v160.inferences_per_sec / rk4_results.inferences_per_sec:,.0f}x faster
    • MAE:           {v160.mae_mm:.2f}mm
    • Within 10mm:   {v160.within_tolerance_pct:.1f}%
    • Model Size:    {v160.model_size_mb:.2f} MB
""")

# ═══════════════════════════════════════════════════════════════════════════════
# BATCH SIZE SCALING
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  BATCH SIZE SCALING")
print("=" * 70)

if v160_model is not None:
    print(f"\n  {'Batch Size':<12} {'v16.0 INT8 (inf/sec)':<25} {'Latency (µs)':<15}")
    print("  " + "-" * 50)
    
    v160_input_dim = v160_model.input_dim
    input_v160 = encode_batch_v160(test_data, v160_input_dim)
    x_full = torch.from_numpy(input_v160).to(config.device)
    
    for bs in config.batch_sizes:
        x_batch = x_full[:bs]
        
        # Warmup
        with torch.no_grad():
            for _ in range(5):
                _ = v160_model(x_batch)
        
        if config.device == "cuda":
            torch.cuda.synchronize()
        
        # Time
        start = time.perf_counter()
        with torch.no_grad():
            for _ in range(100):
                _ = v160_model(x_batch)
        
        if config.device == "cuda":
            torch.cuda.synchronize()
        
        elapsed = time.perf_counter() - start
        inf_per_sec = (100 * bs) / elapsed
        latency = (elapsed / 100) * 1e6
        
        print(f"  {bs:<12} {inf_per_sec:<25,.0f} {latency:<15.1f}")

# ═══════════════════════════════════════════════════════════════════════════════
# ERROR DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("  ERROR DISTRIBUTION")
print("=" * 70)

if v160_model is not None:
    v160_input_dim = v160_model.input_dim
    input_v160 = encode_batch_v160(test_data, v160_input_dim)
    x = torch.from_numpy(input_v160).to(config.device)
    
    with torch.no_grad():
        preds = v160_model(x).cpu().numpy()
    
    errors = np.linalg.norm(preds - ground_truth, axis=1) * 1000.0  # mm
    
    percentiles = [50, 90, 95, 99, 99.9]
    
    print(f"\n  v16.0 INT8 Error Percentiles:")
    print("  " + "-" * 40)
    for p in percentiles:
        val = np.percentile(errors, p)
        print(f"    P{p:<5}: {val:.2f}mm")
    
    print(f"\n  Error Buckets:")
    print("  " + "-" * 40)
    buckets = [(0, 1), (1, 3), (3, 5), (5, 10), (10, 20), (20, float('inf'))]
    for lo, hi in buckets:
        if hi == float('inf'):
            count = np.sum(errors > lo)
            pct = 100.0 * count / len(errors)
            print(f"    > {lo}mm:      {count:>6} ({pct:.2f}%)")
        else:
            count = np.sum((errors >= lo) & (errors < hi))
            pct = 100.0 * count / len(errors)
            print(f"    {lo}-{hi}mm:     {count:>6} ({pct:.2f}%)")

print("\n" + "=" * 70)
print("  BENCHMARK COMPLETE")
print("=" * 70)
