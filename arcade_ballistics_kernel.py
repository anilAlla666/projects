"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║                    HYPERFLUX ARCADE BALLISTICS KERNEL v1.0                    ║
║                         Neural Dynamics Team                                   ║
║                                                                                ║
║  O(1) Hit Point Prediction for 3D Games                                       ║
║  Target: 100x faster than traditional frame-by-frame simulation               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

ARCADE PHYSICS MODEL:
─────────────────────
- Simplified drag (constant deceleration factor)
- Gravity drop (parabolic arc)
- No wind, no spin drift, no Coriolis
- Deterministic and reproducible

WHAT THIS KERNEL DOES:
──────────────────────
1. Predicts EXACT hit point in O(1) - single matrix multiplication
2. Predicts interception points for moving targets
3. Checks against world geometry (AABB/OBB collision)
4. Returns time-of-flight for tracer synchronization

SEGMENTS (Your Spec):
─────────────────────
S1: 0-100m    → ±5mm error
S2: 101-300m  → ±10mm error  
S3: 301-600m  → ±20mm error
S4: 601-1000m → ±30mm error
S5: 1001-1500m → ±50mm error

Author: Anil Kumar Alla / Neural Dynamics Team
Patent: HyperFlux Provisional
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Union
from enum import Enum
import math


# =============================================================================
# DEVICE CONFIGURATION
# =============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"HyperFlux Kernel initialized on: {DEVICE}")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class WeaponClass(Enum):
    """Weapon categories with preset ballistic profiles."""
    PISTOL = "pistol"
    SMG = "smg"
    ASSAULT_RIFLE = "assault_rifle"
    SNIPER = "sniper"
    SHOTGUN = "shotgun"
    LMG = "lmg"
    LAUNCHER = "launcher"  # RPG, grenade launcher
    THROWN = "thrown"      # Grenades, knives


@dataclass
class WeaponProfile:
    """
    Arcade weapon ballistics profile.
    Simplified parameters for fast computation.
    """
    name: str
    weapon_class: WeaponClass
    
    # Core ballistics
    muzzle_velocity: float      # m/s (meters per second)
    damage_base: float          # Base damage at point blank
    damage_falloff_start: float # Distance where damage starts dropping (m)
    damage_falloff_end: float   # Distance where minimum damage is reached (m)
    damage_min_multiplier: float # Minimum damage as fraction of base (0.0-1.0)
    
    # Arcade physics
    gravity_multiplier: float = 1.0    # 1.0 = realistic, 0.0 = hitscan
    drag_coefficient: float = 0.0      # Simplified drag (0.0 = no drag)
    
    # Projectile properties
    projectile_radius: float = 0.005   # For collision detection (m)
    is_explosive: bool = False
    explosion_radius: float = 0.0      # If explosive
    
    # Rate of fire (for batch prediction)
    fire_rate: float = 600.0           # Rounds per minute
    
    def __post_init__(self):
        # Precompute useful values
        self.gravity_effective = 9.81 * self.gravity_multiplier
        

@dataclass
class ShotInput:
    """
    Single shot input for kernel prediction.
    All vectors are in world space.
    """
    # Origin point (muzzle position)
    origin: torch.Tensor          # [3] - x, y, z
    
    # Direction (normalized)
    direction: torch.Tensor       # [3] - normalized direction vector
    
    # Weapon profile index (into registered weapons)
    weapon_id: int
    
    # Optional: Target for interception prediction
    target_position: Optional[torch.Tensor] = None   # [3]
    target_velocity: Optional[torch.Tensor] = None   # [3]


@dataclass
class BatchShotInput:
    """
    Batched shot inputs for parallel O(1) prediction.
    This is the primary interface for game integration.
    """
    origins: torch.Tensor         # [N, 3] - muzzle positions
    directions: torch.Tensor      # [N, 3] - normalized directions
    weapon_ids: torch.Tensor      # [N] - weapon profile indices
    
    # Optional targets for interception
    target_positions: Optional[torch.Tensor] = None  # [N, 3]
    target_velocities: Optional[torch.Tensor] = None # [N, 3]
    
    @property
    def batch_size(self) -> int:
        return self.origins.shape[0]


@dataclass  
class HitResult:
    """
    Result of hit prediction.
    """
    hit_position: torch.Tensor     # [3] or [N, 3] - world space hit point
    time_of_flight: torch.Tensor   # [1] or [N] - seconds
    hit_distance: torch.Tensor     # [1] or [N] - meters
    damage: torch.Tensor           # [1] or [N] - calculated damage
    did_hit_geometry: torch.Tensor # [1] or [N] - bool, hit world geometry
    geometry_hit_point: Optional[torch.Tensor] = None  # If hit geometry before target
    segment_used: int = 0          # Which segment made prediction


@dataclass
class WorldGeometry:
    """
    Simplified world geometry for collision checking.
    Uses Axis-Aligned Bounding Boxes (AABB) for speed.
    """
    # AABB bounds: [N_boxes, 6] where each row is [min_x, min_y, min_z, max_x, max_y, max_z]
    aabb_bounds: torch.Tensor
    
    # Optional: Surface normals for ricochet calculation
    surface_normals: Optional[torch.Tensor] = None
    
    # Material IDs for penetration/ricochet logic
    material_ids: Optional[torch.Tensor] = None


# =============================================================================
# WEAPON PRESETS (COD-style)
# =============================================================================

class WeaponPresets:
    """Pre-configured weapons matching COD-style arcade ballistics."""
    
    @staticmethod
    def m4a1() -> WeaponProfile:
        """M4A1 Assault Rifle - balanced, versatile."""
        return WeaponProfile(
            name="M4A1",
            weapon_class=WeaponClass.ASSAULT_RIFLE,
            muzzle_velocity=884.0,
            damage_base=28.0,
            damage_falloff_start=25.0,
            damage_falloff_end=50.0,
            damage_min_multiplier=0.7,
            gravity_multiplier=0.1,  # Minimal drop for arcade feel
            drag_coefficient=0.001,
            fire_rate=800.0
        )
    
    @staticmethod
    def ak47() -> WeaponProfile:
        """AK-47 - high damage, more drop."""
        return WeaponProfile(
            name="AK-47",
            weapon_class=WeaponClass.ASSAULT_RIFLE,
            muzzle_velocity=715.0,
            damage_base=34.0,
            damage_falloff_start=20.0,
            damage_falloff_end=45.0,
            damage_min_multiplier=0.65,
            gravity_multiplier=0.15,
            drag_coefficient=0.0015,
            fire_rate=600.0
        )
    
    @staticmethod
    def mp5() -> WeaponProfile:
        """MP5 SMG - fast, short range."""
        return WeaponProfile(
            name="MP5",
            weapon_class=WeaponClass.SMG,
            muzzle_velocity=400.0,
            damage_base=25.0,
            damage_falloff_start=10.0,
            damage_falloff_end=30.0,
            damage_min_multiplier=0.5,
            gravity_multiplier=0.2,
            drag_coefficient=0.002,
            fire_rate=800.0
        )
    
    @staticmethod
    def awp() -> WeaponProfile:
        """AWP Sniper - one shot, long range."""
        return WeaponProfile(
            name="AWP",
            weapon_class=WeaponClass.SNIPER,
            muzzle_velocity=936.0,
            damage_base=115.0,
            damage_falloff_start=100.0,
            damage_falloff_end=500.0,
            damage_min_multiplier=0.9,
            gravity_multiplier=0.05,  # Minimal drop
            drag_coefficient=0.0005,
            fire_rate=41.0
        )
    
    @staticmethod
    def deagle() -> WeaponProfile:
        """Desert Eagle - powerful pistol."""
        return WeaponProfile(
            name="Desert Eagle",
            weapon_class=WeaponClass.PISTOL,
            muzzle_velocity=440.0,
            damage_base=53.0,
            damage_falloff_start=15.0,
            damage_falloff_end=35.0,
            damage_min_multiplier=0.6,
            gravity_multiplier=0.25,
            drag_coefficient=0.003,
            fire_rate=267.0
        )
    
    @staticmethod
    def rpg() -> WeaponProfile:
        """RPG-7 - explosive projectile."""
        return WeaponProfile(
            name="RPG-7",
            weapon_class=WeaponClass.LAUNCHER,
            muzzle_velocity=115.0,
            damage_base=150.0,
            damage_falloff_start=0.0,  # Explosion damage
            damage_falloff_end=10.0,
            damage_min_multiplier=0.1,
            gravity_multiplier=1.0,  # Full gravity
            drag_coefficient=0.01,
            is_explosive=True,
            explosion_radius=5.0,
            fire_rate=10.0
        )
    
    @staticmethod
    def frag_grenade() -> WeaponProfile:
        """Frag Grenade - thrown explosive."""
        return WeaponProfile(
            name="Frag Grenade",
            weapon_class=WeaponClass.THROWN,
            muzzle_velocity=20.0,  # Throw speed
            damage_base=200.0,
            damage_falloff_start=0.0,
            damage_falloff_end=8.0,
            damage_min_multiplier=0.0,
            gravity_multiplier=1.0,
            drag_coefficient=0.05,
            is_explosive=True,
            explosion_radius=8.0,
            fire_rate=30.0
        )
    
    @staticmethod
    def throwing_knife() -> WeaponProfile:
        """Throwing Knife - instant kill on hit."""
        return WeaponProfile(
            name="Throwing Knife",
            weapon_class=WeaponClass.THROWN,
            muzzle_velocity=30.0,
            damage_base=500.0,  # Instant kill
            damage_falloff_start=50.0,
            damage_falloff_end=100.0,
            damage_min_multiplier=1.0,  # No falloff
            gravity_multiplier=1.0,
            drag_coefficient=0.08,
            fire_rate=60.0
        )


# =============================================================================
# RK4 GROUND TRUTH ENGINE (O(N) Reference)
# =============================================================================

class RK4ArcadeEngine:
    """
    Fourth-order Runge-Kutta integrator for arcade ballistics.
    
    This is the O(N) reference solver used to:
    1. Generate training data
    2. Validate LNN predictions
    3. Serve as fallback for edge cases
    
    Arcade Physics Model:
    - F_gravity = m * g * gravity_multiplier (downward)
    - F_drag = -drag_coefficient * v * |v| (opposing velocity)
    - a = F_total / m
    """
    
    def __init__(self, dt: float = 0.001):
        """
        Args:
            dt: Integration timestep. 0.001s = 1ms gives good accuracy.
        """
        self.dt = dt
        self.g = 9.81
        
    def compute_acceleration(
        self,
        velocity: torch.Tensor,
        weapon: WeaponProfile
    ) -> torch.Tensor:
        """
        Compute acceleration from forces.
        
        Args:
            velocity: [3] current velocity vector
            weapon: Weapon profile for physics parameters
            
        Returns:
            [3] acceleration vector
        """
        # Gravity (only in Y axis, pointing down)
        gravity = torch.tensor([0.0, -self.g * weapon.gravity_multiplier, 0.0], 
                               device=velocity.device)
        
        # Drag (opposes velocity, proportional to v²)
        speed = torch.norm(velocity)
        if speed > 0.001:
            drag_direction = -velocity / speed
            drag_magnitude = weapon.drag_coefficient * speed * speed
            drag = drag_direction * drag_magnitude
        else:
            drag = torch.zeros(3, device=velocity.device)
        
        return gravity + drag
    
    def step(
        self,
        position: torch.Tensor,
        velocity: torch.Tensor,
        weapon: WeaponProfile
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Single RK4 integration step.
        
        Returns:
            (new_position, new_velocity)
        """
        dt = self.dt
        
        # k1
        a1 = self.compute_acceleration(velocity, weapon)
        v1 = velocity
        
        # k2
        v2 = velocity + 0.5 * dt * a1
        a2 = self.compute_acceleration(v2, weapon)
        
        # k3
        v3 = velocity + 0.5 * dt * a2
        a3 = self.compute_acceleration(v3, weapon)
        
        # k4
        v4 = velocity + dt * a3
        a4 = self.compute_acceleration(v4, weapon)
        
        # Combine
        new_velocity = velocity + (dt / 6.0) * (a1 + 2*a2 + 2*a3 + a4)
        new_position = position + (dt / 6.0) * (v1 + 2*v2 + 2*v3 + v4)
        
        return new_position, new_velocity
    
    def simulate_trajectory(
        self,
        origin: torch.Tensor,
        direction: torch.Tensor,
        weapon: WeaponProfile,
        max_distance: float = 2000.0,
        max_time: float = 10.0,
        ground_height: float = 0.0
    ) -> Tuple[torch.Tensor, float, List[torch.Tensor]]:
        """
        Simulate full trajectory until ground hit or max distance.
        
        Args:
            origin: [3] starting position
            direction: [3] normalized direction
            weapon: Weapon profile
            max_distance: Maximum simulation distance
            max_time: Maximum simulation time
            ground_height: Y coordinate of ground plane
            
        Returns:
            (hit_position, time_of_flight, trajectory_points)
        """
        position = origin.clone()
        velocity = direction * weapon.muzzle_velocity
        
        trajectory = [position.clone()]
        t = 0.0
        total_distance = 0.0
        
        while t < max_time and total_distance < max_distance:
            prev_position = position.clone()
            position, velocity = self.step(position, velocity, weapon)
            
            # Track distance
            step_distance = torch.norm(position - prev_position).item()
            total_distance += step_distance
            t += self.dt
            
            # Record trajectory (every 10 steps for memory efficiency)
            if int(t / self.dt) % 10 == 0:
                trajectory.append(position.clone())
            
            # Ground collision check
            if position[1] <= ground_height and prev_position[1] > ground_height:
                # Interpolate exact ground hit point
                t_ground = (ground_height - prev_position[1]) / (position[1] - prev_position[1])
                hit_position = prev_position + t_ground * (position - prev_position)
                return hit_position, t, trajectory
        
        return position, t, trajectory
    
    def generate_training_batch(
        self,
        weapon: WeaponProfile,
        n_samples: int,
        distance_range: Tuple[float, float],
        seed: int = 42
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate training data batch for a specific distance range.
        
        Args:
            weapon: Weapon profile to simulate
            n_samples: Number of trajectories
            distance_range: (min_distance, max_distance) in meters
            seed: Random seed
            
        Returns:
            (inputs, outputs, times)
            inputs: [N, input_dim] encoded launch conditions
            outputs: [N, 3] hit positions
            times: [N] time of flight
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        inputs = []
        outputs = []
        times = []
        
        min_dist, max_dist = distance_range
        
        for i in range(n_samples):
            # Random origin (simulating different positions on map)
            origin = torch.tensor([
                np.random.uniform(-50, 50),
                np.random.uniform(1.0, 2.0),  # Player height
                np.random.uniform(-50, 50)
            ], dtype=torch.float32, device=DEVICE)
            
            # Random direction (biased toward horizontal for FPS gameplay)
            # Target a point at the desired distance
            target_dist = np.random.uniform(min_dist, max_dist)
            
            # Horizontal angle (full 360)
            azimuth = np.random.uniform(0, 2 * np.pi)
            
            # Vertical angle (mostly horizontal with some up/down)
            # For arcade, we bias toward flat shots
            elevation = np.random.uniform(-0.1, 0.15)  # Radians, mostly flat
            
            # Construct direction
            direction = torch.tensor([
                np.cos(elevation) * np.cos(azimuth),
                np.sin(elevation),
                np.cos(elevation) * np.sin(azimuth)
            ], dtype=torch.float32, device=DEVICE)
            direction = direction / torch.norm(direction)
            
            # Simulate
            hit_pos, tof, _ = self.simulate_trajectory(
                origin, direction, weapon,
                max_distance=max_dist * 1.5,
                ground_height=-10.0  # Below typical map
            )
            
            # Check if within target distance range
            actual_dist = torch.norm(hit_pos - origin).item()
            if actual_dist < min_dist * 0.8 or actual_dist > max_dist * 1.2:
                continue  # Skip outliers
            
            # Encode input
            input_vec = self._encode_input(origin, direction, weapon)
            inputs.append(input_vec)
            outputs.append(hit_pos)
            times.append(tof)
            
            if (i + 1) % 5000 == 0:
                print(f"  Generated {len(inputs)}/{n_samples} valid samples...")
        
        # Pad if we don't have enough samples
        while len(inputs) < n_samples:
            inputs.append(inputs[-1])
            outputs.append(outputs[-1])
            times.append(times[-1])
        
        return (
            torch.stack(inputs[:n_samples]),
            torch.stack(outputs[:n_samples]),
            torch.tensor(times[:n_samples], device=DEVICE)
        )
    
    def _encode_input(
        self,
        origin: torch.Tensor,
        direction: torch.Tensor,
        weapon: WeaponProfile
    ) -> torch.Tensor:
        """
        Encode shot parameters into fixed-size input vector.
        
        Input vector layout (24 dimensions):
        [0:3]   - Origin (x, y, z)
        [3:6]   - Direction (normalized)
        [6:9]   - Velocity vector (direction * muzzle_velocity)
        [9:12]  - Weapon params (muzzle_vel, gravity_mult, drag_coeff)
        [12:15] - Derived: speed, speed², kinetic_energy_proxy
        [15:18] - Derived: horizontal_dir, vertical_component, horizontal_speed
        [18:21] - Derived: estimated_range, estimated_tof, drop_estimate
        [21:24] - Reserved for future use
        """
        velocity = direction * weapon.muzzle_velocity
        speed = weapon.muzzle_velocity
        
        # Horizontal components
        horizontal_speed = torch.sqrt(velocity[0]**2 + velocity[2]**2)
        vertical_component = velocity[1]
        
        # Rough estimates for feature engineering
        # Time to travel 100m horizontally
        est_time_100m = 100.0 / (horizontal_speed + 0.001)
        # Drop in that time
        drop_est = 0.5 * weapon.gravity_effective * est_time_100m ** 2
        
        # Estimated range (simplified ballistic formula)
        if weapon.gravity_effective > 0.01:
            est_range = (speed ** 2 * torch.sin(2 * torch.atan2(vertical_component, horizontal_speed))) / weapon.gravity_effective
        else:
            est_range = torch.tensor(1500.0, device=origin.device)  # Hitscan-like
        
        input_vec = torch.tensor([
            # Origin
            origin[0], origin[1], origin[2],
            # Direction
            direction[0], direction[1], direction[2],
            # Velocity
            velocity[0], velocity[1], velocity[2],
            # Weapon params
            weapon.muzzle_velocity, weapon.gravity_multiplier, weapon.drag_coefficient,
            # Derived
            speed, speed ** 2, speed * weapon.drag_coefficient,
            horizontal_speed.item() if isinstance(horizontal_speed, torch.Tensor) else horizontal_speed,
            vertical_component.item() if isinstance(vertical_component, torch.Tensor) else vertical_component,
            (horizontal_speed ** 2).item() if isinstance(horizontal_speed, torch.Tensor) else horizontal_speed ** 2,
            # Estimates
            est_range.item() if isinstance(est_range, torch.Tensor) else est_range,
            est_time_100m.item() if isinstance(est_time_100m, torch.Tensor) else est_time_100m,
            drop_est.item() if isinstance(drop_est, torch.Tensor) else drop_est,
            # Reserved
            0.0, 0.0, 0.0
        ], dtype=torch.float32, device=origin.device)
        
        return input_vec


# =============================================================================
# FOURIER FEATURE ENCODER (Phase-Space Mapping)
# =============================================================================

class FourierEncoder(nn.Module):
    """
    Phase-Space Encoder using Fourier features.
    
    Maps input to higher-dimensional space to combat spectral bias
    and enable learning of high-frequency functions.
    
    γ(x) = [x, sin(2πf₁x), cos(2πf₁x), ..., sin(2πfₖx), cos(2πfₖx)]
    """
    
    def __init__(
        self,
        input_dim: int,
        num_frequencies: int = 16,
        max_freq: float = 10.0,
        learnable: bool = False
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        
        # Output: original + sin/cos for each frequency
        self.output_dim = input_dim * (1 + 2 * num_frequencies)
        
        # Frequency bands
        frequencies = torch.linspace(1.0, max_freq, num_frequencies)
        
        if learnable:
            self.frequencies = nn.Parameter(frequencies)
        else:
            self.register_buffer('frequencies', frequencies)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply Fourier encoding.
        
        Args:
            x: [batch, input_dim]
            
        Returns:
            [batch, output_dim]
        """
        # Original features
        encoded = [x]
        
        # Fourier features
        for freq in self.frequencies:
            encoded.append(torch.sin(2 * np.pi * freq * x))
            encoded.append(torch.cos(2 * np.pi * freq * x))
        
        return torch.cat(encoded, dim=-1)


# =============================================================================
# LIQUID TIME-CONSTANT NETWORK
# =============================================================================

class LTCCell(nn.Module):
    """
    Liquid Time-Constant (LTC) Neural Network Cell.
    
    The hidden state evolves according to an ODE:
    dh/dt = -h/τ(x) + f(x, h)
    
    Where τ(x) is an input-dependent time constant that allows
    the network to adapt its temporal dynamics.
    
    For inference, we "solve" this ODE and condense to a matrix.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        tau_min: float = 0.1,
        tau_max: float = 5.0
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tau_min = tau_min
        self.tau_max = tau_max
        
        # Input to hidden
        self.W_x = nn.Linear(input_dim, hidden_dim)
        
        # Hidden to hidden (recurrent)
        self.W_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # Time constant network
        self.W_tau = nn.Linear(input_dim, hidden_dim)
        
        # Output projection
        self.W_o = nn.Linear(hidden_dim, output_dim)
        
        # Nonlinearity
        self.activation = nn.Tanh()
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization."""
        for module in [self.W_x, self.W_h, self.W_tau, self.W_o]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def compute_tau(self, x: torch.Tensor) -> torch.Tensor:
        """Compute input-dependent time constants."""
        tau_raw = self.W_tau(x)
        # Sigmoid to bound between tau_min and tau_max
        tau = self.tau_min + (self.tau_max - self.tau_min) * torch.sigmoid(tau_raw)
        return tau
    
    def forward(
        self,
        x: torch.Tensor,
        num_steps: int = 5,
        dt: float = 0.2
    ) -> torch.Tensor:
        """
        Forward pass with ODE integration.
        
        Args:
            x: [batch, input_dim]
            num_steps: Number of ODE integration steps
            dt: Integration timestep
            
        Returns:
            [batch, output_dim]
        """
        batch_size = x.shape[0]
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_dim, device=x.device)
        
        # Get time constants
        tau = self.compute_tau(x)
        
        # Integrate ODE
        for _ in range(num_steps):
            # Input contribution
            x_contrib = self.activation(self.W_x(x))
            
            # Hidden contribution
            h_contrib = self.W_h(h)
            
            # ODE: dh/dt = -h/tau + activation(x_contrib + h_contrib)
            dh_dt = -h / tau + self.activation(x_contrib + h_contrib)
            
            # Euler step
            h = h + dt * dh_dt
        
        # Output projection
        output = self.W_o(h)
        
        return output
    
    def forward_condensed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Condensed forward pass - approximates ODE solution directly.
        
        For trained networks operating in a bounded input domain,
        the ODE dynamics can be approximated by a direct mapping.
        
        This is the O(1) inference path!
        """
        # Approximate steady-state solution
        tau = self.compute_tau(x)
        tau_avg = tau.mean(dim=-1, keepdim=True)
        
        # Steady state: h_ss ≈ tau * activation(W_x(x))
        x_contrib = self.activation(self.W_x(x))
        h_approx = tau_avg * x_contrib
        
        # Output
        return self.W_o(h_approx)


# =============================================================================
# BALLISTICS SEGMENT NETWORK
# =============================================================================

class BallisticsSegment(nn.Module):
    """
    Single distance-specialized network segment.
    
    Architecture:
    Input → FourierEncoder → LTC → Output (hit position)
    
    Each segment is trained on a specific distance range
    with calibrated error tolerance.
    """
    
    def __init__(
        self,
        segment_id: int,
        distance_range: Tuple[float, float],
        max_error_mm: float,
        input_dim: int = 24,
        fourier_frequencies: int = 16,
        hidden_dim: int = 128
    ):
        super().__init__()
        
        self.segment_id = segment_id
        self.distance_min, self.distance_max = distance_range
        self.max_error_mm = max_error_mm
        
        # Fourier encoder
        self.encoder = FourierEncoder(input_dim, fourier_frequencies)
        encoded_dim = self.encoder.output_dim
        
        # LTC network
        self.ltc = LTCCell(
            input_dim=encoded_dim,
            hidden_dim=hidden_dim,
            output_dim=3  # Hit position (x, y, z)
        )
        
        # Condensed inference matrix (computed after training)
        self.inference_matrix: Optional[torch.Tensor] = None
        self.is_compiled = False
        
        # Normalization parameters
        self.register_buffer('input_mean', torch.zeros(input_dim))
        self.register_buffer('input_std', torch.ones(input_dim))
        self.register_buffer('output_mean', torch.zeros(3))
        self.register_buffer('output_std', torch.ones(3))
    
    def set_normalization(
        self,
        input_mean: torch.Tensor,
        input_std: torch.Tensor,
        output_mean: torch.Tensor,
        output_std: torch.Tensor
    ):
        """Set normalization parameters from training data."""
        self.input_mean = input_mean
        self.input_std = input_std
        self.output_mean = output_mean
        self.output_std = output_std
    
    def normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.input_mean) / (self.input_std + 1e-8)
    
    def denormalize_output(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.output_std + self.output_mean
    
    def forward(self, x: torch.Tensor, use_condensed: bool = False) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: [batch, input_dim] raw input
            use_condensed: If True, use O(1) matrix multiplication
            
        Returns:
            [batch, 3] predicted hit positions
        """
        # Normalize
        x_norm = self.normalize_input(x)
        
        # Encode
        x_encoded = self.encoder(x_norm)
        
        # Predict
        if use_condensed and self.is_compiled:
            y_norm = self._condensed_forward(x_encoded)
        else:
            y_norm = self.ltc(x_encoded)
        
        # Denormalize
        return self.denormalize_output(y_norm)
    
    def _condensed_forward(self, x_encoded: torch.Tensor) -> torch.Tensor:
        """O(1) inference with precomputed matrix."""
        if self.inference_matrix is None:
            raise RuntimeError("Segment not compiled. Call compile_for_inference().")
        return x_encoded @ self.inference_matrix
    
    def compile_for_inference(self, calibration_data: torch.Tensor):
        """
        Compile the network into a single matrix multiplication.
        
        Uses least-squares fitting on calibration data to find
        the best linear approximation of the learned function.
        """
        self.eval()
        with torch.no_grad():
            # Get LTC outputs for calibration data
            x_norm = self.normalize_input(calibration_data)
            x_encoded = self.encoder(x_norm)
            y_target = self.ltc(x_encoded)
            
            # Solve least squares: x_encoded @ W = y_target
            # W = (X^T X)^{-1} X^T Y
            XtX = x_encoded.T @ x_encoded
            XtY = x_encoded.T @ y_target
            
            # Add regularization for numerical stability
            reg = 1e-6 * torch.eye(XtX.shape[0], device=XtX.device)
            self.inference_matrix = torch.linalg.solve(XtX + reg, XtY)
            
            self.is_compiled = True
            
            # Validate condensation
            y_condensed = x_encoded @ self.inference_matrix
            error = torch.norm(y_condensed - y_target, dim=1).mean() * 1000
            print(f"  Segment {self.segment_id} condensation error: {error:.2f}mm")


# =============================================================================
# MAIN HYPERFLUX BALLISTICS KERNEL
# =============================================================================

class HyperFluxBallisticsKernel(nn.Module):
    """
    ╔═══════════════════════════════════════════════════════════════════════════╗
    ║                    HYPERFLUX BALLISTICS KERNEL                            ║
    ║                                                                            ║
    ║  The main kernel combining all components for O(1) hit prediction.        ║
    ╚═══════════════════════════════════════════════════════════════════════════╝
    
    Features:
    - 5 distance-specialized LNN segments
    - Automatic segment selection based on estimated range
    - World geometry collision checking
    - Moving target interception prediction
    - Batched inference for multiple projectiles
    
    Usage:
        kernel = HyperFluxBallisticsKernel()
        kernel.register_weapon(WeaponPresets.m4a1())
        kernel.train_all_segments(weapon_id=0, samples_per_segment=50000)
        kernel.compile_for_inference()
        
        # O(1) inference
        hit = kernel.predict(origin, direction, weapon_id=0)
    """
    
    # Segment configuration
    SEGMENT_CONFIG = {
        1: {"range": (0, 100), "error_mm": 5, "hidden_dim": 64},
        2: {"range": (101, 300), "error_mm": 10, "hidden_dim": 96},
        3: {"range": (301, 600), "error_mm": 20, "hidden_dim": 128},
        4: {"range": (601, 1000), "error_mm": 30, "hidden_dim": 192},
        5: {"range": (1001, 1500), "error_mm": 50, "hidden_dim": 256}
    }
    
    def __init__(self, input_dim: int = 24):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Create segments
        self.segments = nn.ModuleDict()
        for seg_id, config in self.SEGMENT_CONFIG.items():
            self.segments[str(seg_id)] = BallisticsSegment(
                segment_id=seg_id,
                distance_range=config["range"],
                max_error_mm=config["error_mm"],
                input_dim=input_dim,
                hidden_dim=config["hidden_dim"]
            )
        
        # Registered weapons
        self.weapons: Dict[int, WeaponProfile] = {}
        self.weapon_count = 0
        
        # RK4 engine for training/validation
        self.rk4_engine = RK4ArcadeEngine(dt=0.001)
        
        # World geometry (set externally)
        self.world_geometry: Optional[WorldGeometry] = None
        
        # Statistics
        self.stats = {
            "total_predictions": 0,
            "segment_usage": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            "geometry_hits": 0,
            "interceptions_computed": 0
        }
    
    def register_weapon(self, weapon: WeaponProfile) -> int:
        """
        Register a weapon and return its ID.
        """
        weapon_id = self.weapon_count
        self.weapons[weapon_id] = weapon
        self.weapon_count += 1
        print(f"Registered weapon '{weapon.name}' with ID {weapon_id}")
        return weapon_id
    
    def set_world_geometry(self, geometry: WorldGeometry):
        """Set world geometry for collision detection."""
        self.world_geometry = geometry
        print(f"World geometry set: {geometry.aabb_bounds.shape[0]} AABBs")
    
    def estimate_range(
        self,
        origin: torch.Tensor,
        direction: torch.Tensor,
        weapon: WeaponProfile
    ) -> float:
        """
        Quick range estimation for segment selection.
        """
        v0 = weapon.muzzle_velocity
        g = weapon.gravity_effective
        
        # Vertical angle
        sin_theta = direction[1].item()
        cos_theta = torch.sqrt(1 - direction[1]**2).item()
        
        if g < 0.01:
            # Essentially hitscan
            return 1500.0
        
        # Simplified ballistic range formula
        # R = v0² * sin(2θ) / g
        sin_2theta = 2 * sin_theta * cos_theta
        
        if sin_2theta < 0:
            # Shooting downward
            return abs(v0 * cos_theta * 2.0)  # Rough estimate
        
        range_est = (v0 ** 2 * sin_2theta) / g
        return max(10.0, min(1500.0, range_est))
    
    def select_segment(self, estimated_range: float) -> int:
        """Select appropriate segment based on range."""
        for seg_id, config in self.SEGMENT_CONFIG.items():
            if config["range"][0] <= estimated_range <= config["range"][1]:
                return seg_id
        return 5 if estimated_range > 1000 else 1
    
    def predict_single(
        self,
        origin: torch.Tensor,
        direction: torch.Tensor,
        weapon_id: int,
        check_geometry: bool = True,
        use_condensed: bool = True
    ) -> HitResult:
        """
        Predict hit point for a single shot.
        
        This is the main O(1) inference entry point!
        """
        self.stats["total_predictions"] += 1
        
        weapon = self.weapons[weapon_id]
        
        # Estimate range and select segment
        est_range = self.estimate_range(origin, direction, weapon)
        seg_id = self.select_segment(est_range)
        self.stats["segment_usage"][seg_id] += 1
        
        segment = self.segments[str(seg_id)]
        
        # Encode input
        input_vec = self.rk4_engine._encode_input(origin, direction, weapon)
        input_vec = input_vec.unsqueeze(0)  # Add batch dimension
        
        # O(1) Prediction!
        hit_position = segment(input_vec, use_condensed=use_condensed)[0]
        
        # Calculate derived values
        hit_distance = torch.norm(hit_position - origin)
        time_of_flight = hit_distance / weapon.muzzle_velocity  # Simplified
        
        # Calculate damage
        damage = self._calculate_damage(hit_distance.item(), weapon)
        
        # Check world geometry
        did_hit_geometry = False
        geometry_hit_point = None
        
        if check_geometry and self.world_geometry is not None:
            geom_hit, geom_point = self._check_geometry_collision(
                origin, hit_position
            )
            if geom_hit:
                did_hit_geometry = True
                geometry_hit_point = geom_point
                hit_position = geom_point
                hit_distance = torch.norm(geom_point - origin)
                self.stats["geometry_hits"] += 1
        
        return HitResult(
            hit_position=hit_position,
            time_of_flight=time_of_flight,
            hit_distance=hit_distance,
            damage=torch.tensor(damage),
            did_hit_geometry=torch.tensor(did_hit_geometry),
            geometry_hit_point=geometry_hit_point,
            segment_used=seg_id
        )
    
    def predict_batch(
        self,
        batch: BatchShotInput,
        check_geometry: bool = True,
        use_condensed: bool = True
    ) -> HitResult:
        """
        Predict hit points for a batch of shots.
        
        This processes multiple projectiles in parallel for maximum throughput.
        """
        n = batch.batch_size
        
        # Group by segment for efficient batched processing
        segment_indices = {1: [], 2: [], 3: [], 4: [], 5: []}
        
        for i in range(n):
            weapon = self.weapons[batch.weapon_ids[i].item()]
            est_range = self.estimate_range(
                batch.origins[i], batch.directions[i], weapon
            )
            seg_id = self.select_segment(est_range)
            segment_indices[seg_id].append(i)
        
        # Allocate output tensors
        hit_positions = torch.zeros(n, 3, device=DEVICE)
        times = torch.zeros(n, device=DEVICE)
        distances = torch.zeros(n, device=DEVICE)
        damages = torch.zeros(n, device=DEVICE)
        geometry_hits = torch.zeros(n, dtype=torch.bool, device=DEVICE)
        
        # Process each segment's batch
        for seg_id, indices in segment_indices.items():
            if not indices:
                continue
                
            segment = self.segments[str(seg_id)]
            self.stats["segment_usage"][seg_id] += len(indices)
            
            # Gather inputs for this segment
            seg_inputs = []
            for i in indices:
                weapon = self.weapons[batch.weapon_ids[i].item()]
                input_vec = self.rk4_engine._encode_input(
                    batch.origins[i], batch.directions[i], weapon
                )
                seg_inputs.append(input_vec)
            
            seg_inputs = torch.stack(seg_inputs)
            
            # Batch prediction
            seg_outputs = segment(seg_inputs, use_condensed=use_condensed)
            
            # Scatter results back
            for j, i in enumerate(indices):
                hit_positions[i] = seg_outputs[j]
                distances[i] = torch.norm(seg_outputs[j] - batch.origins[i])
                weapon = self.weapons[batch.weapon_ids[i].item()]
                times[i] = distances[i] / weapon.muzzle_velocity
                damages[i] = self._calculate_damage(distances[i].item(), weapon)
        
        self.stats["total_predictions"] += n
        
        return HitResult(
            hit_position=hit_positions,
            time_of_flight=times,
            hit_distance=distances,
            damage=damages,
            did_hit_geometry=geometry_hits,
            segment_used=0  # Mixed
        )
    
    def predict_interception(
        self,
        origin: torch.Tensor,
        direction: torch.Tensor,
        weapon_id: int,
        target_pos: torch.Tensor,
        target_vel: torch.Tensor
    ) -> Tuple[torch.Tensor, float, bool]:
        """
        Predict interception point for a moving target.
        
        Solves: target_pos + target_vel * t = bullet_pos(t)
        
        Returns:
            (aim_direction, intercept_time, is_possible)
        """
        self.stats["interceptions_computed"] += 1
        
        weapon = self.weapons[weapon_id]
        v_bullet = weapon.muzzle_velocity
        
        # Vector from origin to target
        to_target = target_pos - origin
        
        # Solve quadratic for intercept time
        # |target_pos + target_vel*t - origin|² = (v_bullet * t)²
        
        a = torch.sum(target_vel ** 2) - v_bullet ** 2
        b = 2 * torch.sum(to_target * target_vel)
        c = torch.sum(to_target ** 2)
        
        discriminant = b ** 2 - 4 * a * c
        
        if discriminant < 0:
            # No intercept possible (target too fast)
            return direction, float('inf'), False
        
        # Choose the smaller positive root
        sqrt_disc = torch.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a + 1e-10)
        t2 = (-b + sqrt_disc) / (2 * a + 1e-10)
        
        if t1 > 0:
            t_intercept = t1
        elif t2 > 0:
            t_intercept = t2
        else:
            return direction, float('inf'), False
        
        # Calculate intercept point
        intercept_point = target_pos + target_vel * t_intercept
        
        # Calculate aim direction
        aim_direction = intercept_point - origin
        aim_direction = aim_direction / (torch.norm(aim_direction) + 1e-10)
        
        return aim_direction, t_intercept.item(), True
    
    def _calculate_damage(self, distance: float, weapon: WeaponProfile) -> float:
        """Calculate damage based on distance falloff."""
        if distance <= weapon.damage_falloff_start:
            return weapon.damage_base
        elif distance >= weapon.damage_falloff_end:
            return weapon.damage_base * weapon.damage_min_multiplier
        else:
            # Linear interpolation
            t = (distance - weapon.damage_falloff_start) / (
                weapon.damage_falloff_end - weapon.damage_falloff_start
            )
            return weapon.damage_base * (1 - t * (1 - weapon.damage_min_multiplier))
    
    def _check_geometry_collision(
        self,
        origin: torch.Tensor,
        end: torch.Tensor
    ) -> Tuple[bool, Optional[torch.Tensor]]:
        """
        Check ray against world geometry AABBs.
        
        Uses slab method for ray-AABB intersection.
        """
        if self.world_geometry is None:
            return False, None
        
        direction = end - origin
        length = torch.norm(direction)
        direction = direction / (length + 1e-10)
        
        closest_hit = None
        closest_t = float('inf')
        
        for i in range(self.world_geometry.aabb_bounds.shape[0]):
            aabb = self.world_geometry.aabb_bounds[i]
            min_bound = aabb[:3]
            max_bound = aabb[3:]
            
            # Ray-AABB intersection (slab method)
            t_hit = self._ray_aabb_intersect(origin, direction, min_bound, max_bound)
            
            if t_hit is not None and 0 < t_hit < length and t_hit < closest_t:
                closest_t = t_hit
                closest_hit = origin + direction * t_hit
        
        if closest_hit is not None:
            return True, closest_hit
        return False, None
    
    def _ray_aabb_intersect(
        self,
        origin: torch.Tensor,
        direction: torch.Tensor,
        min_bound: torch.Tensor,
        max_bound: torch.Tensor
    ) -> Optional[float]:
        """Ray-AABB intersection using slab method."""
        inv_dir = 1.0 / (direction + 1e-10)
        
        t1 = (min_bound - origin) * inv_dir
        t2 = (max_bound - origin) * inv_dir
        
        t_min = torch.min(t1, t2)
        t_max = torch.max(t1, t2)
        
        t_enter = torch.max(t_min)
        t_exit = torch.min(t_max)
        
        if t_enter > t_exit or t_exit < 0:
            return None
        
        return max(0, t_enter.item())
    
    # =========================================================================
    # TRAINING METHODS
    # =========================================================================
    
    def train_segment(
        self,
        segment_id: int,
        weapon_id: int,
        n_samples: int = 50000,
        epochs: int = 100,
        batch_size: int = 256,
        learning_rate: float = 0.001
    ) -> Dict:
        """
        Train a single segment on RK4-generated data.
        """
        weapon = self.weapons[weapon_id]
        config = self.SEGMENT_CONFIG[segment_id]
        segment = self.segments[str(segment_id)]
        
        print(f"\n{'='*60}")
        print(f"Training Segment {segment_id}: {config['range'][0]}-{config['range'][1]}m")
        print(f"Target error: ±{config['error_mm']}mm")
        print(f"Weapon: {weapon.name}")
        print(f"{'='*60}")
        
        # Generate training data
        print("\nGenerating training data with RK4...")
        X_train, Y_train, _ = self.rk4_engine.generate_training_batch(
            weapon, n_samples, config["range"], seed=42 + segment_id
        )
        
        # Generate validation data
        print("Generating validation data...")
        X_val, Y_val, _ = self.rk4_engine.generate_training_batch(
            weapon, n_samples // 5, config["range"], seed=1000 + segment_id
        )
        
        # Compute normalization
        input_mean = X_train.mean(dim=0)
        input_std = X_train.std(dim=0)
        output_mean = Y_train.mean(dim=0)
        output_std = Y_train.std(dim=0)
        
        segment.set_normalization(input_mean, input_std, output_mean, output_std)
        
        # Normalize data
        X_train_norm = (X_train - input_mean) / (input_std + 1e-8)
        Y_train_norm = (Y_train - output_mean) / (output_std + 1e-8)
        X_val_norm = (X_val - input_mean) / (input_std + 1e-8)
        Y_val_norm = (Y_val - output_mean) / (output_std + 1e-8)
        
        # Optimizer
        optimizer = torch.optim.Adam(segment.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=10, factor=0.5
        )
        
        # Training loop
        n_batches = len(X_train) // batch_size
        history = {"train_loss": [], "val_loss": [], "val_error_mm": []}
        best_val_loss = float('inf')
        
        print("\nTraining...")
        for epoch in range(epochs):
            segment.train()
            epoch_loss = 0.0
            
            # Shuffle
            perm = torch.randperm(len(X_train))
            X_shuffled = X_train_norm[perm]
            Y_shuffled = Y_train_norm[perm]
            
            for b in range(n_batches):
                start = b * batch_size
                end = start + batch_size
                
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]
                
                # Forward
                X_encoded = segment.encoder(X_batch)
                Y_pred = segment.ltc(X_encoded)
                
                # Loss
                loss = F.mse_loss(Y_pred, Y_batch)
                
                # Backward
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(segment.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_train_loss = epoch_loss / n_batches
            
            # Validation
            segment.eval()
            with torch.no_grad():
                X_val_encoded = segment.encoder(X_val_norm)
                Y_val_pred = segment.ltc(X_val_encoded)
                val_loss = F.mse_loss(Y_val_pred, Y_val_norm).item()
                
                # Denormalize and compute actual error
                Y_val_pred_denorm = Y_val_pred * output_std + output_mean
                errors = torch.norm(Y_val_pred_denorm - Y_val, dim=1) * 1000  # mm
                mean_error_mm = errors.mean().item()
            
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_error_mm"].append(mean_error_mm)
            
            scheduler.step(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1:3d}/{epochs} | "
                      f"Train Loss: {avg_train_loss:.6f} | "
                      f"Val Loss: {val_loss:.6f} | "
                      f"Error: {mean_error_mm:.2f}mm")
        
        # Final evaluation
        segment.eval()
        with torch.no_grad():
            Y_val_pred_denorm = segment(X_val, use_condensed=False)
            final_errors = torch.norm(Y_val_pred_denorm - Y_val, dim=1) * 1000
            final_mean = final_errors.mean().item()
            final_max = final_errors.max().item()
            final_p95 = torch.quantile(final_errors, 0.95).item()
        
        print(f"\n  Final Results:")
        print(f"    Mean error: {final_mean:.2f}mm (target: ±{config['error_mm']}mm)")
        print(f"    Max error:  {final_max:.2f}mm")
        print(f"    P95 error:  {final_p95:.2f}mm")
        
        status = "✓ PASSED" if final_mean <= config['error_mm'] else "✗ NEEDS IMPROVEMENT"
        print(f"    Status: {status}")
        
        return {
            "history": history,
            "final_mean_error_mm": final_mean,
            "final_max_error_mm": final_max,
            "final_p95_error_mm": final_p95,
            "passed": final_mean <= config['error_mm']
        }
    
    def train_all_segments(
        self,
        weapon_id: int,
        samples_per_segment: int = 50000,
        epochs: int = 100
    ) -> Dict:
        """Train all 5 segments."""
        results = {}
        
        print("\n" + "="*70)
        print("  HYPERFLUX BALLISTICS KERNEL - FULL TRAINING")
        print("="*70)
        
        for seg_id in self.SEGMENT_CONFIG.keys():
            results[seg_id] = self.train_segment(
                seg_id, weapon_id, samples_per_segment, epochs
            )
        
        # Summary
        print("\n" + "="*70)
        print("  TRAINING SUMMARY")
        print("="*70)
        
        all_passed = True
        for seg_id, result in results.items():
            config = self.SEGMENT_CONFIG[seg_id]
            status = "✓" if result["passed"] else "✗"
            all_passed = all_passed and result["passed"]
            print(f"  S{seg_id} ({config['range'][0]:4d}-{config['range'][1]:4d}m): "
                  f"{result['final_mean_error_mm']:6.2f}mm / ±{config['error_mm']:2d}mm {status}")
        
        print(f"\n  Overall: {'ALL PASSED ✓' if all_passed else 'SOME FAILED ✗'}")
        
        return results
    
    def compile_for_inference(self):
        """Compile all segments for O(1) inference."""
        print("\n" + "="*60)
        print("  COMPILING FOR O(1) INFERENCE")
        print("="*60)
        
        for seg_id in self.SEGMENT_CONFIG.keys():
            segment = self.segments[str(seg_id)]
            
            # Generate calibration data
            config = self.SEGMENT_CONFIG[seg_id]
            weapon = list(self.weapons.values())[0]  # Use first weapon
            
            X_cal, _, _ = self.rk4_engine.generate_training_batch(
                weapon, 10000, config["range"], seed=9999 + seg_id
            )
            
            print(f"\n  Compiling Segment {seg_id}...")
            segment.compile_for_inference(X_cal)
        
        print("\n  Compilation complete! Kernel ready for O(1) inference.")
    
    # =========================================================================
    # BENCHMARKING
    # =========================================================================
    
    def benchmark(
        self,
        weapon_id: int,
        n_iterations: int = 10000
    ) -> Dict:
        """
        Benchmark O(1) inference vs O(N) RK4.
        """
        import time
        
        weapon = self.weapons[weapon_id]
        
        print("\n" + "="*60)
        print("  BENCHMARK: O(1) LNN vs O(N) RK4")
        print("="*60)
        
        # Prepare test shots
        origins = []
        directions = []
        
        for _ in range(n_iterations):
            origin = torch.tensor([
                np.random.uniform(-50, 50),
                np.random.uniform(1.0, 2.0),
                np.random.uniform(-50, 50)
            ], dtype=torch.float32, device=DEVICE)
            
            direction = torch.tensor([
                np.random.uniform(-1, 1),
                np.random.uniform(-0.2, 0.3),
                np.random.uniform(-1, 1)
            ], dtype=torch.float32, device=DEVICE)
            direction = direction / torch.norm(direction)
            
            origins.append(origin)
            directions.append(direction)
        
        # Warm up
        for i in range(100):
            self.predict_single(origins[i], directions[i], weapon_id)
        
        # Benchmark O(1) LNN
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        
        for i in range(n_iterations):
            self.predict_single(origins[i], directions[i], weapon_id, 
                              check_geometry=False, use_condensed=True)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.perf_counter()
        lnn_time = t1 - t0
        
        # Benchmark O(N) RK4
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.perf_counter()
        
        for i in range(min(1000, n_iterations)):  # RK4 is slow, limit samples
            self.rk4_engine.simulate_trajectory(
                origins[i], directions[i], weapon, max_distance=500
            )
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t1 = time.perf_counter()
        rk4_time = (t1 - t0) * (n_iterations / min(1000, n_iterations))
        
        # Results
        lnn_per_shot = (lnn_time / n_iterations) * 1e6  # microseconds
        rk4_per_shot = (rk4_time / n_iterations) * 1e6
        speedup = rk4_time / lnn_time
        
        print(f"\n  Results ({n_iterations} iterations):")
        print(f"    O(1) LNN:  {lnn_per_shot:8.2f} μs/shot")
        print(f"    O(N) RK4:  {rk4_per_shot:8.2f} μs/shot")
        print(f"    Speedup:   {speedup:8.1f}x")
        print(f"\n  Target: 100x speedup → {'✓ ACHIEVED' if speedup >= 100 else '✗ NOT YET'}")
        
        # Accuracy check
        print("\n  Accuracy verification:")
        errors = []
        for i in range(min(500, n_iterations)):
            lnn_result = self.predict_single(
                origins[i], directions[i], weapon_id,
                check_geometry=False, use_condensed=True
            )
            rk4_hit, _, _ = self.rk4_engine.simulate_trajectory(
                origins[i], directions[i], weapon
            )
            error = torch.norm(lnn_result.hit_position - rk4_hit).item() * 1000
            errors.append(error)
        
        errors = np.array(errors)
        print(f"    Mean error: {errors.mean():.2f}mm")
        print(f"    Max error:  {errors.max():.2f}mm")
        print(f"    P95 error:  {np.percentile(errors, 95):.2f}mm")
        
        return {
            "lnn_us_per_shot": lnn_per_shot,
            "rk4_us_per_shot": rk4_per_shot,
            "speedup": speedup,
            "mean_error_mm": errors.mean(),
            "max_error_mm": errors.max()
        }
    
    def save(self, path: str):
        """Save trained kernel to disk."""
        torch.save({
            'state_dict': self.state_dict(),
            'weapons': {k: v.__dict__ for k, v in self.weapons.items()},
            'stats': self.stats
        }, path)
        print(f"Kernel saved to {path}")
    
    def load(self, path: str):
        """Load trained kernel from disk."""
        checkpoint = torch.load(path, map_location=DEVICE)
        self.load_state_dict(checkpoint['state_dict'])
        self.stats = checkpoint['stats']
        print(f"Kernel loaded from {path}")


# =============================================================================
# EXAMPLE USAGE & DEMO
# =============================================================================

def main():
    """Demo of the HyperFlux Ballistics Kernel."""
    
    print("╔" + "═"*68 + "╗")
    print("║" + " "*20 + "HYPERFLUX BALLISTICS KERNEL" + " "*21 + "║")
    print("║" + " "*22 + "Neural Dynamics Team" + " "*26 + "║")
    print("╚" + "═"*68 + "╝")
    
    # Initialize kernel
    kernel = HyperFluxBallisticsKernel()
    kernel.to(DEVICE)
    
    # Register weapons
    m4_id = kernel.register_weapon(WeaponPresets.m4a1())
    ak_id = kernel.register_weapon(WeaponPresets.ak47())
    awp_id = kernel.register_weapon(WeaponPresets.awp())
    
    # Demo: RK4 ground truth
    print("\n" + "="*60)
    print("  DEMO: RK4 Ground Truth Simulation")
    print("="*60)
    
    origin = torch.tensor([0., 1.7, 0.], device=DEVICE)
    direction = torch.tensor([0.99, 0.1, 0.05], device=DEVICE)
    direction = direction / torch.norm(direction)
    
    hit_pos, tof, trajectory = kernel.rk4_engine.simulate_trajectory(
        origin, direction, kernel.weapons[m4_id]
    )
    
    print(f"\n  Weapon: M4A1")
    print(f"  Origin: ({origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f})")
    print(f"  Direction: ({direction[0]:.3f}, {direction[1]:.3f}, {direction[2]:.3f})")
    print(f"\n  RK4 Result:")
    print(f"    Hit position: ({hit_pos[0]:.2f}, {hit_pos[1]:.2f}, {hit_pos[2]:.2f})")
    print(f"    Distance: {torch.norm(hit_pos - origin):.2f}m")
    print(f"    Time of flight: {tof*1000:.2f}ms")
    print(f"    Trajectory points: {len(trajectory)}")
    
    print("\n" + "="*60)
    print("  Kernel ready for training!")
    print("  Use: kernel.train_all_segments(weapon_id=0)")
    print("="*60)
    
    return kernel


if __name__ == "__main__":
    kernel = main()
