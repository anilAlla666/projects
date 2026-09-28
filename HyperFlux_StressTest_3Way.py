"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    HYPERFLUX BRUTAL STRESS TEST v1.0                         ║
║                                                                              ║
║  3-WAY COMPARISON:                                                          ║
║    1. RK4 Physics O(N)   - Ground truth (100+ integration steps)            ║
║    2. 229 Teachers O(1)  - Specialized neural networks                      ║
║    3. v15.5 Student O(1) - Unified 1.5MB model                              ║
║                                                                              ║
║  BRUTAL TEST CONDITIONS:                                                    ║
║    - All 66 weapons                                                         ║
║    - All 229 segments                                                       ║
║    - 1000 random shots per segment = 229,000 total shots                    ║
║    - Extreme angles, edge cases, worst-case scenarios                       ║
║    - Full coordinate comparison (x, y, z)                                   ║
║    - Speed benchmarking                                                     ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import glob
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

print("╔══════════════════════════════════════════════════════════════════════════════╗")
print("║                    HYPERFLUX BRUTAL STRESS TEST                              ║")
print("║                    RK4 Physics vs Teachers vs Student                        ║")
print("╚══════════════════════════════════════════════════════════════════════════════╝")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"  Device: {device}")
if device.type == 'cuda':
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class TestConfig:
    teacher_dir: str = "/content/drive/MyDrive/HyperFlux_Models"
    student_path: str = "/content/drive/MyDrive/HyperFlux_v15.5_Unified_20260112_050055/unified_ballistics_fp32.pt"
    
    # Alternative paths if the above doesn't work:
    # student_path: str = "/content/drive/MyDrive/HyperFlux_v15_Unified_20260111_160721/unified_ballistics_fp32.pt"  # v15.3
    
    shots_per_segment: int = 1000      # 1000 shots per segment = 229,000 total
    include_extreme_cases: bool = True  # Edge cases and worst-case scenarios
    verbose: bool = True                # Show per-weapon results
    
CFG = TestConfig()

# ═══════════════════════════════════════════════════════════════════════════════
# PHYSICS CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

GRAVITY = 9.80665       # m/s²
AIR_DENSITY = 1.225     # kg/m³
SPEED_OF_SOUND = 343.0  # m/s

# ═══════════════════════════════════════════════════════════════════════════════
# DRAG COEFFICIENT TABLES (G1, G7, SPHERE)
# ═══════════════════════════════════════════════════════════════════════════════

# G7 - Boat tail bullets (long range rifle)
G7_MACH = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0]
G7_CD = [0.120, 0.120, 0.122, 0.128, 0.145, 0.170, 0.210, 0.290, 0.395, 0.420, 0.430, 0.432, 0.430, 0.420, 0.408, 0.395, 0.382, 0.358, 0.335, 0.315, 0.290, 0.255]

# G1 - Flat base bullets (pistol, older rifle)
G1_MACH = [0.0, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.15, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.0, 2.2, 2.5, 3.0]
G1_CD = [0.230, 0.230, 0.235, 0.250, 0.290, 0.350, 0.420, 0.480, 0.520, 0.510, 0.495, 0.480, 0.465, 0.440, 0.420, 0.400, 0.382, 0.350, 0.325, 0.305, 0.280, 0.250]

# SPHERE - Grenades, balls
SPHERE_MACH = [0.0, 0.5, 0.8, 1.0, 1.2, 1.5, 2.0, 3.0]
SPHERE_CD = [0.47, 0.47, 0.49, 0.52, 0.50, 0.48, 0.45, 0.42]


def interpolate_cd(mach: float, mach_table: List[float], cd_table: List[float]) -> float:
    """Interpolate drag coefficient from table."""
    if mach <= mach_table[0]:
        return cd_table[0]
    if mach >= mach_table[-1]:
        return cd_table[-1]
    
    for i in range(len(mach_table) - 1):
        if mach_table[i] <= mach <= mach_table[i + 1]:
            t = (mach - mach_table[i]) / (mach_table[i + 1] - mach_table[i])
            return cd_table[i] + t * (cd_table[i + 1] - cd_table[i])
    
    return cd_table[-1]


def get_drag_coefficient(speed: float, drag_type: str) -> float:
    """Get drag coefficient based on speed and drag model."""
    mach = speed / SPEED_OF_SOUND
    
    if drag_type == "G7":
        return interpolate_cd(mach, G7_MACH, G7_CD)
    elif drag_type == "G1":
        return interpolate_cd(mach, G1_MACH, G1_CD)
    elif drag_type == "SPHERE":
        return interpolate_cd(mach, SPHERE_MACH, SPHERE_CD)
    elif drag_type == "ROCKET":
        return 0.3  # Simplified rocket drag
    else:
        return interpolate_cd(mach, G7_MACH, G7_CD)

# ═══════════════════════════════════════════════════════════════════════════════
# RK4 PHYSICS SIMULATION - GROUND TRUTH O(N)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_acceleration(pos: np.ndarray, vel: np.ndarray, 
                         bc: float, bullet_mass: float, drag_type: str,
                         has_thrust: bool = False, thrust_time: float = 0.0,
                         t: float = 0.0) -> np.ndarray:
    """Compute acceleration from gravity and drag."""
    # Gravity
    acc = np.array([0.0, -GRAVITY, 0.0])
    
    # Drag
    speed = np.linalg.norm(vel)
    if speed > 0.1:
        cd = get_drag_coefficient(speed, drag_type)
        
        # Drag force: F = 0.5 * rho * v^2 * Cd * A
        # A = reference area derived from BC
        # BC = mass / (Cd * d^2), so Cd * A ≈ mass / BC (simplified)
        drag_factor = (AIR_DENSITY * speed * cd) / (2 * bc * 7000)  # 7000 converts BC units
        drag_acc = -drag_factor * vel
        acc += drag_acc
    
    # Rocket thrust (simplified)
    if has_thrust and t < thrust_time:
        thrust_acc = 50.0  # m/s² additional thrust
        if speed > 0.1:
            thrust_dir = vel / speed
            acc += thrust_acc * thrust_dir
    
    return acc


def rk4_step(pos: np.ndarray, vel: np.ndarray, dt: float,
             bc: float, bullet_mass: float, drag_type: str,
             has_thrust: bool, thrust_time: float, t: float) -> Tuple[np.ndarray, np.ndarray]:
    """Single RK4 integration step."""
    
    # k1
    a1 = compute_acceleration(pos, vel, bc, bullet_mass, drag_type, has_thrust, thrust_time, t)
    k1_v = a1 * dt
    k1_p = vel * dt
    
    # k2
    a2 = compute_acceleration(pos + k1_p/2, vel + k1_v/2, bc, bullet_mass, drag_type, has_thrust, thrust_time, t + dt/2)
    k2_v = a2 * dt
    k2_p = (vel + k1_v/2) * dt
    
    # k3
    a3 = compute_acceleration(pos + k2_p/2, vel + k2_v/2, bc, bullet_mass, drag_type, has_thrust, thrust_time, t + dt/2)
    k3_v = a3 * dt
    k3_p = (vel + k2_v/2) * dt
    
    # k4
    a4 = compute_acceleration(pos + k3_p, vel + k3_v, bc, bullet_mass, drag_type, has_thrust, thrust_time, t + dt)
    k4_v = a4 * dt
    k4_p = (vel + k3_v) * dt
    
    # Combine
    new_pos = pos + (k1_p + 2*k2_p + 2*k3_p + k4_p) / 6
    new_vel = vel + (k1_v + 2*k2_v + 2*k3_v + k4_v) / 6
    
    return new_pos, new_vel


def simulate_rk4(v0: np.ndarray, t_final: float, 
                 bc: float, bullet_mass: float, drag_type: str,
                 has_thrust: bool = False, thrust_time: float = 0.5,
                 dt: float = 0.001) -> np.ndarray:
    """
    Full RK4 simulation - GROUND TRUTH O(N).
    
    This is what games currently use (or similar).
    Each shot requires N integration steps where N = t_final / dt.
    For t=1s with dt=0.001: N = 1000 steps per shot!
    """
    pos = np.array([0.0, 0.0, 0.0])
    vel = v0.copy()
    t = 0.0
    
    n_steps = int(t_final / dt) + 1
    
    for _ in range(n_steps):
        if t >= t_final:
            break
        
        actual_dt = min(dt, t_final - t)
        pos, vel = rk4_step(pos, vel, actual_dt, bc, bullet_mass, drag_type, 
                           has_thrust, thrust_time, t)
        t += actual_dt
    
    return pos


def simulate_rk4_batch(v0_batch: np.ndarray, t_batch: np.ndarray,
                       bc: float, bullet_mass: float, drag_type: str,
                       has_thrust: bool = False) -> np.ndarray:
    """Batch RK4 simulation for multiple shots."""
    results = []
    for i in range(len(v0_batch)):
        pos = simulate_rk4(v0_batch[i], t_batch[i], bc, bullet_mass, drag_type, has_thrust)
        results.append(pos)
    return np.array(results)

print("✓ RK4 Physics Engine ready (Ground Truth O(N))")

# ═══════════════════════════════════════════════════════════════════════════════
# TEACHER MODEL - O(1) SPECIALIZED
# ═══════════════════════════════════════════════════════════════════════════════

class CfCCell(nn.Module):
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
    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        f = self.f_net(features) + 0.1
        g = self.g_net(features)
        h = self.h_net(features)
        gate = torch.sigmoid(-f * t)
        return gate * g + (1 - gate) * h


class OracleKernel(nn.Module):
    def __init__(self, hidden_dim: int = 320):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.cfc = CfCCell(input_dim=4, hidden_dim=hidden_dim, output_dim=3)
        self.register_buffer('v0_mean', torch.zeros(3))
        self.register_buffer('v0_std', torch.ones(3))
        self.register_buffer('pos_mean', torch.zeros(3))
        self.register_buffer('pos_std', torch.ones(3))
        self.t_max = 1.0
    
    def forward(self, v0: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        v0_norm = (v0 - self.v0_mean) / self.v0_std
        t_norm = t / self.t_max
        x = torch.cat([v0_norm, t_norm], dim=-1)
        raw_pos = self.cfc(x, t_norm)
        return raw_pos * self.pos_std + self.pos_mean

print("✓ Teacher architecture ready (O(1) Specialized)")

# ═══════════════════════════════════════════════════════════════════════════════
# STUDENT MODEL - O(1) UNIFIED
# ═══════════════════════════════════════════════════════════════════════════════

class StudentUnifiedModel(nn.Module):
    def __init__(self, input_dim=15, hidden_dims=[512, 512, 512, 256, 256], output_dim=3):
        super().__init__()
        
        layers = []
        prev = input_dim
        
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.LayerNorm(h),
                nn.SiLU()
            ])
            prev = h
        
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

print("✓ Student architecture ready (O(1) Unified)")

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def load_teachers(teacher_dir: str) -> Dict:
    """Load all 229 teacher models."""
    print(f"\n  Loading teachers from: {teacher_dir}")
    
    bin_files = sorted(glob.glob(os.path.join(teacher_dir, "HyperFlux_*.bin")))
    print(f"  Found {len(bin_files)} files")
    
    teachers = {}
    
    for bin_path in bin_files:
        filename = os.path.basename(bin_path)
        parts = filename.replace("HyperFlux_", "").replace(".bin", "").rsplit("_S", 1)
        
        if len(parts) != 2:
            continue
        
        weapon_name = parts[0]
        try:
            segment = int(parts[1])
        except ValueError:
            continue
        
        try:
            ckpt = torch.load(bin_path, map_location=device, weights_only=False)
            hidden_dim = ckpt.get('hidden_dim', 320)
            t_max = ckpt.get('t_max', 1.0)
            weapon_info = ckpt.get('weapon', {})
            
            model = OracleKernel(hidden_dim=hidden_dim)
            model.load_state_dict(ckpt['model_state'])
            model.t_max = t_max
            model.to(device)
            model.eval()
            
            if weapon_name not in teachers:
                teachers[weapon_name] = {}
            
            teachers[weapon_name][segment] = {
                'model': model,
                't_max': t_max,
                'hidden_dim': hidden_dim,
                'weapon': weapon_info,
            }
            
        except Exception as e:
            pass
    
    total = sum(len(s) for s in teachers.values())
    print(f"  ✓ Loaded {total} teachers ({len(teachers)} weapons)")
    return teachers


def load_student(student_path: str) -> nn.Module:
    """Load unified student model."""
    print(f"\n  Loading student from: {student_path}")
    
    ckpt = torch.load(student_path, map_location=device, weights_only=False)
    config = ckpt.get('config', {})
    
    student = StudentUnifiedModel(
        input_dim=config.get('input_dim', 15),
        hidden_dims=config.get('hidden_dims', [512, 512, 512, 256, 256]),
        output_dim=config.get('output_dim', 3)
    )
    
    student.load_state_dict(ckpt['model_state_dict'])
    student.to(device)
    student.eval()
    
    print(f"  ✓ Student loaded (v{ckpt.get('version', '?')}, MAE: {ckpt.get('best_mae_mm', '?')}mm)")
    return student

# ═══════════════════════════════════════════════════════════════════════════════
# DRAG ENCODING FOR STUDENT
# ═══════════════════════════════════════════════════════════════════════════════

def get_drag_encoding(drag_type: str) -> List[float]:
    if drag_type == "G7": return [1, 0, 0, 0]
    elif drag_type == "G1": return [0, 1, 0, 0]
    elif drag_type == "SPHERE": return [0, 0, 1, 0]
    elif drag_type == "ROCKET": return [0, 0, 0, 1]
    return [1, 0, 0, 0]

# ═══════════════════════════════════════════════════════════════════════════════
# BRUTAL STRESS TEST
# ═══════════════════════════════════════════════════════════════════════════════

def generate_test_shots(n: int, muzzle_vel: float, t_max: float, 
                        include_extreme: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Generate diverse test shots including edge cases."""
    
    v0_list = []
    t_list = []
    
    # Normal shots (70%)
    n_normal = int(n * 0.7)
    azimuth = np.random.rand(n_normal) * 2 * np.pi
    elevation = (np.random.rand(n_normal) - 0.3) * 0.6  # -18° to +12°
    
    dx = np.cos(elevation) * np.cos(azimuth)
    dy = np.sin(elevation)
    dz = np.cos(elevation) * np.sin(azimuth)
    
    directions = np.stack([dx, dy, dz], axis=1)
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    directions = directions / norms
    
    v0_normal = directions * muzzle_vel
    t_normal = np.random.rand(n_normal) * t_max * 0.9 + t_max * 0.05
    
    v0_list.append(v0_normal)
    t_list.append(t_normal)
    
    if include_extreme:
        # Extreme angles (15%) - steep up/down shots
        n_extreme = int(n * 0.15)
        azimuth = np.random.rand(n_extreme) * 2 * np.pi
        elevation = (np.random.rand(n_extreme) - 0.5) * 1.2  # -30° to +30°
        
        dx = np.cos(elevation) * np.cos(azimuth)
        dy = np.sin(elevation)
        dz = np.cos(elevation) * np.sin(azimuth)
        
        directions = np.stack([dx, dy, dz], axis=1)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / norms
        
        v0_extreme = directions * muzzle_vel
        t_extreme = np.random.rand(n_extreme) * t_max * 0.95 + t_max * 0.025
        
        v0_list.append(v0_extreme)
        t_list.append(t_extreme)
        
        # Edge case times (15%) - very early and very late
        n_edge = n - n_normal - n_extreme
        azimuth = np.random.rand(n_edge) * 2 * np.pi
        elevation = (np.random.rand(n_edge) - 0.3) * 0.6
        
        dx = np.cos(elevation) * np.cos(azimuth)
        dy = np.sin(elevation)
        dz = np.cos(elevation) * np.sin(azimuth)
        
        directions = np.stack([dx, dy, dz], axis=1)
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        directions = directions / norms
        
        v0_edge = directions * muzzle_vel
        
        # Mix of very early (t < 0.1*t_max) and very late (t > 0.95*t_max)
        t_edge = np.concatenate([
            np.random.rand(n_edge // 2) * t_max * 0.1 + 0.001,
            np.random.rand(n_edge - n_edge // 2) * t_max * 0.05 + t_max * 0.95
        ])
        
        v0_list.append(v0_edge)
        t_list.append(t_edge)
    
    v0 = np.concatenate(v0_list, axis=0)
    t = np.concatenate(t_list, axis=0)
    
    return v0, t


def run_stress_test(teachers: Dict, student: nn.Module, 
                    shots_per_segment: int = 1000,
                    verbose: bool = True) -> Dict:
    """
    BRUTAL 3-WAY STRESS TEST
    
    Compares:
    1. RK4 Physics (Ground Truth) - O(N)
    2. Teacher Models - O(1)
    3. Student Model - O(1)
    """
    
    print("\n" + "█"*78)
    print("█" + " "*25 + "BRUTAL STRESS TEST" + " "*31 + "█")
    print("█" + " "*15 + "RK4 Physics vs Teachers vs Student" + " "*24 + "█")
    print("█"*78)
    
    total_segments = sum(len(s) for s in teachers.values())
    total_shots = total_segments * shots_per_segment
    
    print(f"\n  Test Configuration:")
    print(f"    Weapons:           {len(teachers)}")
    print(f"    Segments:          {total_segments}")
    print(f"    Shots/segment:     {shots_per_segment:,}")
    print(f"    Total shots:       {total_shots:,}")
    print(f"    Extreme cases:     {'Yes' if CFG.include_extreme_cases else 'No'}")
    
    # Results storage
    all_results = {
        'weapons': {},
        'summary': {
            'teacher_vs_rk4': [],
            'student_vs_rk4': [],
            'student_vs_teacher': [],
        },
        'timing': {
            'rk4_total': 0,
            'teacher_total': 0,
            'student_total': 0,
        },
        'sample_shots': []  # Store some sample shots for display
    }
    
    print(f"\n  {'='*74}")
    print(f"  {'Weapon':<20} | {'Seg':>3} | {'RK4 Time':>10} | {'Teacher Δ':>10} | {'Student Δ':>10} | {'Stud vs T':>10}")
    print(f"  {'='*74}")
    
    processed = 0
    sample_count = 0
    max_samples = 50  # Store 50 sample shots for detailed display
    
    for weapon_name, segments in teachers.items():
        weapon_results = {
            'segments': {},
            'teacher_vs_rk4': [],
            'student_vs_rk4': [],
            'student_vs_teacher': [],
        }
        
        for seg_id, seg_data in segments.items():
            teacher_model = seg_data['model']
            t_max = seg_data['t_max']
            hidden_dim = seg_data['hidden_dim']
            weapon_info = seg_data.get('weapon', {})
            
            # Weapon properties
            muzzle_vel = weapon_info.get('muzzle_velocity', 800.0)
            bc = weapon_info.get('bc', 0.15)
            bullet_mass = weapon_info.get('bullet_mass', 0.01)
            drag_type = weapon_info.get('drag_type', 'G7')
            effective_range = weapon_info.get('effective_range', 500.0)
            has_thrust = weapon_info.get('has_thrust', False)
            
            # Generate test shots
            v0_np, t_np = generate_test_shots(
                shots_per_segment, muzzle_vel, t_max, CFG.include_extreme_cases
            )
            
            # ═══════════════════════════════════════════════════════════════
            # 1. RK4 PHYSICS - GROUND TRUTH O(N)
            # ═══════════════════════════════════════════════════════════════
            t0 = time.time()
            rk4_positions = simulate_rk4_batch(v0_np, t_np, bc, bullet_mass, drag_type, has_thrust)
            rk4_time = time.time() - t0
            all_results['timing']['rk4_total'] += rk4_time
            
            # ═══════════════════════════════════════════════════════════════
            # 2. TEACHER MODEL - O(1)
            # ═══════════════════════════════════════════════════════════════
            v0_torch = torch.tensor(v0_np, dtype=torch.float32, device=device)
            t_torch = torch.tensor(t_np, dtype=torch.float32, device=device)
            
            t0 = time.time()
            with torch.no_grad():
                teacher_positions = teacher_model(v0_torch, t_torch).cpu().numpy()
            teacher_time = time.time() - t0
            all_results['timing']['teacher_total'] += teacher_time
            
            # ═══════════════════════════════════════════════════════════════
            # 3. STUDENT MODEL - O(1)
            # ═══════════════════════════════════════════════════════════════
            
            # Prepare student input
            v0_norm = v0_torch / 1000.0
            t_norm = t_torch.unsqueeze(-1) / 10.0
            
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
            ], device=device).unsqueeze(0).expand(len(v0_np), -1)
            
            student_input = torch.cat([v0_norm, t_norm, weapon_features], dim=1)
            
            t0 = time.time()
            with torch.no_grad():
                student_positions = student(student_input).cpu().numpy()
            student_time = time.time() - t0
            all_results['timing']['student_total'] += student_time
            
            # ═══════════════════════════════════════════════════════════════
            # COMPUTE ERRORS
            # ═══════════════════════════════════════════════════════════════
            
            # Teacher vs RK4
            teacher_errors = np.linalg.norm(teacher_positions - rk4_positions, axis=1) * 1000  # mm
            
            # Student vs RK4
            student_errors = np.linalg.norm(student_positions - rk4_positions, axis=1) * 1000  # mm
            
            # Student vs Teacher
            student_teacher_errors = np.linalg.norm(student_positions - teacher_positions, axis=1) * 1000  # mm
            
            # Store results
            weapon_results['segments'][seg_id] = {
                'teacher_vs_rk4': {
                    'mean': np.mean(teacher_errors),
                    'std': np.std(teacher_errors),
                    'max': np.max(teacher_errors),
                    'p95': np.percentile(teacher_errors, 95),
                    'p99': np.percentile(teacher_errors, 99),
                },
                'student_vs_rk4': {
                    'mean': np.mean(student_errors),
                    'std': np.std(student_errors),
                    'max': np.max(student_errors),
                    'p95': np.percentile(student_errors, 95),
                    'p99': np.percentile(student_errors, 99),
                },
                'student_vs_teacher': {
                    'mean': np.mean(student_teacher_errors),
                    'std': np.std(student_teacher_errors),
                    'max': np.max(student_teacher_errors),
                    'p95': np.percentile(student_teacher_errors, 95),
                    'p99': np.percentile(student_teacher_errors, 99),
                },
            }
            
            weapon_results['teacher_vs_rk4'].extend(teacher_errors)
            weapon_results['student_vs_rk4'].extend(student_errors)
            weapon_results['student_vs_teacher'].extend(student_teacher_errors)
            
            all_results['summary']['teacher_vs_rk4'].extend(teacher_errors)
            all_results['summary']['student_vs_rk4'].extend(student_errors)
            all_results['summary']['student_vs_teacher'].extend(student_teacher_errors)
            
            # Store sample shots for display
            if sample_count < max_samples:
                for i in range(min(5, shots_per_segment)):
                    if sample_count >= max_samples:
                        break
                    all_results['sample_shots'].append({
                        'weapon': weapon_name,
                        'segment': seg_id,
                        'v0': v0_np[i],
                        't': t_np[i],
                        'rk4_pos': rk4_positions[i],
                        'teacher_pos': teacher_positions[i],
                        'student_pos': student_positions[i],
                        'teacher_error': teacher_errors[i],
                        'student_error': student_errors[i],
                    })
                    sample_count += 1
            
            # Print progress
            if verbose:
                print(f"  {weapon_name:<20} | {seg_id:>3} | {rk4_time*1000:>8.1f}ms | {np.mean(teacher_errors):>8.2f}mm | {np.mean(student_errors):>8.2f}mm | {np.mean(student_teacher_errors):>8.2f}mm")
            
            processed += 1
        
        all_results['weapons'][weapon_name] = weapon_results
        
        if not verbose and processed % 50 == 0:
            print(f"  Progress: {processed}/{total_segments} segments...")
    
    return all_results


def print_results(results: Dict):
    """Print comprehensive stress test results."""
    
    print("\n" + "█"*78)
    print("█" + " "*28 + "STRESS TEST RESULTS" + " "*27 + "█")
    print("█"*78)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SAMPLE SHOTS - ACTUAL COORDINATES
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n" + "="*100)
    print("  SAMPLE SHOTS - ACTUAL IMPACT COORDINATES")
    print("="*100)
    print(f"  {'Weapon':<15} | {'V0 (m/s)':<20} | {'Time':>6} | {'RK4 (Ground Truth)':<25} | {'Teacher':<25} | {'Student':<25} | {'Errors':<15}")
    print(f"  {'-'*95}")
    
    for shot in results['sample_shots'][:20]:  # Show first 20
        v0_str = f"({shot['v0'][0]:.0f},{shot['v0'][1]:.0f},{shot['v0'][2]:.0f})"
        rk4_str = f"({shot['rk4_pos'][0]:.2f},{shot['rk4_pos'][1]:.2f},{shot['rk4_pos'][2]:.2f})"
        teacher_str = f"({shot['teacher_pos'][0]:.2f},{shot['teacher_pos'][1]:.2f},{shot['teacher_pos'][2]:.2f})"
        student_str = f"({shot['student_pos'][0]:.2f},{shot['student_pos'][1]:.2f},{shot['student_pos'][2]:.2f})"
        error_str = f"T:{shot['teacher_error']:.1f} S:{shot['student_error']:.1f}mm"
        
        print(f"  {shot['weapon']:<15} | {v0_str:<20} | {shot['t']:>5.3f}s | {rk4_str:<25} | {teacher_str:<25} | {student_str:<25} | {error_str:<15}")
    
    print(f"  ... and {len(results['sample_shots']) - 20} more sample shots")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # OVERALL STATISTICS
    # ═══════════════════════════════════════════════════════════════════════════
    
    teacher_errors = np.array(results['summary']['teacher_vs_rk4'])
    student_errors = np.array(results['summary']['student_vs_rk4'])
    student_teacher_errors = np.array(results['summary']['student_vs_teacher'])
    
    print("\n" + "="*78)
    print("  OVERALL ERROR STATISTICS (vs RK4 Ground Truth)")
    print("="*78)
    
    print(f"""
  ╔════════════════════════════════════════════════════════════════════════════╗
  ║                           ERROR SUMMARY                                    ║
  ╠════════════════════════════════════════════════════════════════════════════╣
  ║                    │  Teacher (229 models)  │  Student (1 model)           ║
  ╠════════════════════════════════════════════════════════════════════════════╣
  ║  Mean Error        │  {np.mean(teacher_errors):>8.2f} mm           │  {np.mean(student_errors):>8.2f} mm                  ║
  ║  Std Dev           │  {np.std(teacher_errors):>8.2f} mm           │  {np.std(student_errors):>8.2f} mm                  ║
  ║  Median            │  {np.median(teacher_errors):>8.2f} mm           │  {np.median(student_errors):>8.2f} mm                  ║
  ║  P95               │  {np.percentile(teacher_errors, 95):>8.2f} mm           │  {np.percentile(student_errors, 95):>8.2f} mm                  ║
  ║  P99               │  {np.percentile(teacher_errors, 99):>8.2f} mm           │  {np.percentile(student_errors, 99):>8.2f} mm                  ║
  ║  Max Error         │  {np.max(teacher_errors):>8.2f} mm           │  {np.max(student_errors):>8.2f} mm                  ║
  ║  Min Error         │  {np.min(teacher_errors):>8.2f} mm           │  {np.min(student_errors):>8.2f} mm                  ║
  ╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    print(f"""
  ╔════════════════════════════════════════════════════════════════════════════╗
  ║                    STUDENT vs TEACHER COMPARISON                           ║
  ╠════════════════════════════════════════════════════════════════════════════╣
  ║  Mean Difference   │  {np.mean(student_teacher_errors):>8.2f} mm                                      ║
  ║  P95 Difference    │  {np.percentile(student_teacher_errors, 95):>8.2f} mm                                      ║
  ║  P99 Difference    │  {np.percentile(student_teacher_errors, 99):>8.2f} mm                                      ║
  ║  Max Difference    │  {np.max(student_teacher_errors):>8.2f} mm                                      ║
  ╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TIMING / SPEED COMPARISON
    # ═══════════════════════════════════════════════════════════════════════════
    
    total_shots = len(teacher_errors)
    rk4_time = results['timing']['rk4_total']
    teacher_time = results['timing']['teacher_total']
    student_time = results['timing']['student_total']
    
    rk4_per_shot = rk4_time / total_shots * 1000 * 1000  # microseconds
    teacher_per_shot = teacher_time / total_shots * 1000 * 1000
    student_per_shot = student_time / total_shots * 1000 * 1000
    
    rk4_throughput = total_shots / rk4_time
    teacher_throughput = total_shots / teacher_time
    student_throughput = total_shots / student_time
    
    print(f"""
  ╔════════════════════════════════════════════════════════════════════════════╗
  ║                           SPEED COMPARISON                                 ║
  ╠════════════════════════════════════════════════════════════════════════════╣
  ║                    │  RK4 Physics    │  Teacher        │  Student          ║
  ╠════════════════════════════════════════════════════════════════════════════╣
  ║  Total Time        │  {rk4_time:>8.2f} s      │  {teacher_time:>8.4f} s     │  {student_time:>8.4f} s        ║
  ║  Per Shot          │  {rk4_per_shot:>8.1f} μs    │  {teacher_per_shot:>8.2f} μs    │  {student_per_shot:>8.2f} μs       ║
  ║  Throughput        │  {rk4_throughput:>8.0f}/s     │  {teacher_throughput:>8.0f}/s    │  {student_throughput:>8.0f}/s       ║
  ║  Speedup vs RK4    │  {1:>8.0f}x        │  {rk4_time/teacher_time:>8.0f}x       │  {rk4_time/student_time:>8.0f}x          ║
  ╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PER-WEAPON SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n" + "="*78)
    print("  PER-WEAPON SUMMARY")
    print("="*78)
    print(f"  {'Weapon':<20} | {'Segments':>8} | {'Teacher Mean':>12} | {'Student Mean':>12} | {'Student P99':>11}")
    print(f"  {'-'*74}")
    
    for weapon_name, weapon_data in results['weapons'].items():
        n_segs = len(weapon_data['segments'])
        t_mean = np.mean(weapon_data['teacher_vs_rk4'])
        s_mean = np.mean(weapon_data['student_vs_rk4'])
        s_p99 = np.percentile(weapon_data['student_vs_rk4'], 99)
        
        print(f"  {weapon_name:<20} | {n_segs:>8} | {t_mean:>10.2f}mm | {s_mean:>10.2f}mm | {s_p99:>9.2f}mm")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL VERDICT
    # ═══════════════════════════════════════════════════════════════════════════
    
    print("\n" + "█"*78)
    print("█" + " "*30 + "FINAL VERDICT" + " "*31 + "█")
    print("█"*78)
    
    student_mean = np.mean(student_errors)
    student_p99 = np.percentile(student_errors, 99)
    speedup = rk4_time / student_time
    
    print(f"""
  ╔════════════════════════════════════════════════════════════════════════════╗
  ║  HYPERFLUX v15.5 UNIFIED MODEL - STRESS TEST RESULTS                       ║
  ╠════════════════════════════════════════════════════════════════════════════╣
  ║                                                                            ║
  ║  Total Shots Tested:     {total_shots:>10,}                                     ║
  ║                                                                            ║
  ║  ACCURACY vs Ground Truth (RK4):                                           ║
  ║    Mean Error:           {student_mean:>10.2f} mm                                    ║
  ║    P99 Error:            {student_p99:>10.2f} mm                                    ║
  ║                                                                            ║
  ║  SPEED:                                                                    ║
  ║    Speedup vs RK4:       {speedup:>10.0f}x                                       ║
  ║    Throughput:           {student_throughput:>10,.0f} shots/sec                          ║
  ║                                                                            ║
  ║  SIZE:                                                                     ║
  ║    Model Size:           1.5 MB (FP16)                                     ║
  ║    Compression:          483x (from 729 MB)                                ║
  ║                                                                            ║
  ╚════════════════════════════════════════════════════════════════════════════╝
    """)
    
    # Pass/Fail criteria
    if student_mean < 10 and student_p99 < 50:
        print("  ✅ STRESS TEST PASSED - Model is production ready!")
    elif student_mean < 20 and student_p99 < 100:
        print("  ⚠️  STRESS TEST MARGINAL - Model acceptable for most use cases")
    else:
        print("  ❌ STRESS TEST FAILED - Model needs improvement")
    
    print("\n" + "█"*78)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # Mount drive
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        print("✓ Drive mounted")
    except:
        print("⚠ Not in Colab")
    
    # Load models
    teachers = load_teachers(CFG.teacher_dir)
    student = load_student(CFG.student_path)
    
    if not teachers:
        print("❌ Failed to load teachers!")
        return
    
    # Run brutal stress test
    results = run_stress_test(
        teachers, 
        student,
        shots_per_segment=CFG.shots_per_segment,
        verbose=CFG.verbose
    )
    
    # Print comprehensive results
    print_results(results)
    
    return results


if __name__ == "__main__":
    results = main()
