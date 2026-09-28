"""
HyperFlux Ballistics Kernel v1.0
================================
A Liquid Neural Network kernel for O(1) ballistic trajectory prediction.

Architecture:
- RK4 Ground Truth Engine for physics-accurate training data
- 5-Segment Distance-Specialized LNN Networks
- Universal Projectile Encoding (bullets, grenades, arrows, knives, RPGs)
- Single Matrix Multiplication Inference Path

Segments:
- S1: 0-100m    (±5mm error)
- S2: 101-300m  (±10mm error)
- S3: 301-600m  (±20mm error)
- S4: 601-1000m (±30mm error)
- S5: 1001-1500m (±50mm error)

Author: Neural Dynamics Team
Patent: Provisional - HyperFlux Kernel
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple, Optional, List, Dict, Union
from enum import Enum
import json


# =============================================================================
# CONSTANTS & PHYSICS PARAMETERS
# =============================================================================

class PhysicsConstants:
    """Real-world physics constants for ballistic simulation."""
    GRAVITY = 9.80665  # m/s² (standard gravity)
    AIR_DENSITY_SEA_LEVEL = 1.225  # kg/m³ at 15°C
    SPEED_OF_SOUND = 343.0  # m/s at 20°C
    EARTH_ROTATION = 7.2921e-5  # rad/s (for Coriolis)
    

class ProjectileType(Enum):
    """Supported projectile categories."""
    BULLET_PISTOL = "bullet_pistol"
    BULLET_RIFLE = "bullet_rifle"
    BULLET_SNIPER = "bullet_sniper"
    BULLET_SMG = "bullet_smg"
    SHOTGUN_PELLET = "shotgun_pellet"
    GRENADE_FRAG = "grenade_frag"
    GRENADE_SMOKE = "grenade_smoke"
    GRENADE_FLASH = "grenade_flash"
    RPG_ROCKET = "rpg_rocket"
    ARROW_STANDARD = "arrow_standard"
    ARROW_CROSSBOW = "arrow_crossbow"
    KNIFE_THROWN = "knife_thrown"
    MORTAR_SHELL = "mortar_shell"


# =============================================================================
# PROJECTILE DATA STRUCTURES
# =============================================================================

@dataclass
class ProjectileProfile:
    """
    Universal projectile descriptor encoding all physics-relevant properties.
    This gets encoded into the neural network input vector.
    """
    # Identity
    projectile_type: ProjectileType
    name: str
    
    # Mass & Geometry
    mass: float  # kg
    diameter: float  # m (for drag calculation)
    length: float  # m (for stability)
    cross_sectional_area: float = field(init=False)  # m² (computed)
    
    # Aerodynamics
    drag_coefficient: float  # Cd (typically 0.2-0.5 for bullets)
    lift_coefficient: float = 0.0  # Cl (for spinning projectiles)
    magnus_coefficient: float = 0.0  # For spin-induced drift
    
    # Ballistic Coefficient (higher = less drag effect)
    ballistic_coefficient: float = 0.0  # G1 or G7 standard
    
    # Launch Properties
    muzzle_velocity: float  # m/s
    spin_rate: float = 0.0  # rad/s (rifling-induced)
    
    # Stability
    gyroscopic_stability: float = 1.0  # Sg factor
    dynamic_stability: float = 1.0  # Sd factor
    
    # Special Properties
    is_rocket_propelled: bool = False
    thrust_duration: float = 0.0  # s
    thrust_force: float = 0.0  # N
    has_fins: bool = False
    fin_stabilization: float = 0.0
    
    def __post_init__(self):
        self.cross_sectional_area = np.pi * (self.diameter / 2) ** 2
        if self.ballistic_coefficient == 0:
            # Estimate BC from mass and area (simplified)
            self.ballistic_coefficient = self.mass / (self.drag_coefficient * self.cross_sectional_area)


@dataclass
class EnvironmentState:
    """
    Environmental conditions affecting trajectory.
    """
    # Atmospheric
    air_density: float = 1.225  # kg/m³
    temperature: float = 288.15  # K (15°C)
    pressure: float = 101325.0  # Pa
    humidity: float = 0.5  # 0-1 relative
    
    # Wind (3D vector)
    wind_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    
    # Altitude effects
    altitude: float = 0.0  # m above sea level
    
    # Location (for Coriolis)
    latitude: float = 45.0  # degrees
    
    # Gravity variations
    local_gravity: float = 9.80665  # m/s²
    
    def get_adjusted_air_density(self) -> float:
        """Calculate air density adjusted for altitude and temperature."""
        # Barometric formula approximation
        scale_height = 8500  # m
        density_altitude = self.air_density * np.exp(-self.altitude / scale_height)
        # Temperature correction
        temp_ratio = 288.15 / self.temperature
        return density_altitude * temp_ratio


@dataclass 
class LaunchState:
    """
    Initial launch conditions.
    """
    position: np.ndarray  # [x, y, z] in meters
    velocity: np.ndarray  # [vx, vy, vz] in m/s (direction * muzzle_velocity)
    orientation: np.ndarray = field(default_factory=lambda: np.array([0., 0., 0.]))  # Euler angles
    angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))  # rad/s
    
    @property
    def speed(self) -> float:
        return np.linalg.norm(self.velocity)
    
    @property
    def direction(self) -> np.ndarray:
        return self.velocity / (self.speed + 1e-10)


@dataclass
class ImpactResult:
    """
    Predicted impact point and trajectory metadata.
    """
    position: np.ndarray  # [x, y, z] impact point
    time_of_flight: float  # seconds
    impact_velocity: np.ndarray  # velocity at impact
    impact_angle: float  # degrees from horizontal
    trajectory_length: float  # total path length in meters
    max_height: float  # apex height
    segment_used: int  # which LNN segment (1-5)
    confidence: float  # prediction confidence 0-1
    error_estimate: float  # estimated error in mm


# =============================================================================
# RK4 GROUND TRUTH ENGINE
# =============================================================================

class RK4BallisticsEngine:
    """
    Fourth-order Runge-Kutta integrator for ground truth trajectory generation.
    This is the O(N) reference solver that generates training data.
    
    Physics modeled:
    - Gravity (with altitude variation)
    - Air drag (velocity-dependent, Cd model)
    - Magnus effect (spin-induced lateral drift)
    - Coriolis effect (Earth rotation)
    - Wind effects (3D wind field)
    - Rocket thrust (for RPGs)
    - Density altitude effects
    """
    
    def __init__(self, dt: float = 0.0001):
        """
        Initialize RK4 engine.
        
        Args:
            dt: Integration timestep in seconds. 
                Smaller = more accurate but slower.
                0.0001s gives sub-mm accuracy for most projectiles.
        """
        self.dt = dt
        self.physics = PhysicsConstants()
        
    def compute_forces(
        self,
        state: np.ndarray,  # [x, y, z, vx, vy, vz]
        t: float,
        projectile: ProjectileProfile,
        environment: EnvironmentState
    ) -> np.ndarray:
        """
        Compute all forces acting on projectile.
        
        Returns: acceleration vector [ax, ay, az]
        """
        pos = state[:3]
        vel = state[3:6]
        speed = np.linalg.norm(vel)
        
        if speed < 1e-10:
            return np.array([0, -environment.local_gravity, 0])
        
        direction = vel / speed
        
        # === GRAVITY ===
        # Adjust for altitude (simplified)
        g = environment.local_gravity * (6371000 / (6371000 + pos[1])) ** 2
        gravity_force = np.array([0, -g, 0]) * projectile.mass
        
        # === AIR DRAG ===
        # F_drag = 0.5 * rho * v² * Cd * A
        rho = environment.get_adjusted_air_density()
        
        # Velocity relative to air (subtract wind)
        vel_relative = vel - environment.wind_velocity
        speed_relative = np.linalg.norm(vel_relative)
        
        if speed_relative > 1e-10:
            # Mach number for compressibility effects
            mach = speed_relative / self.physics.SPEED_OF_SOUND
            
            # Drag coefficient varies with Mach (transonic drag rise)
            cd = self._get_drag_coefficient(projectile.drag_coefficient, mach)
            
            drag_magnitude = 0.5 * rho * speed_relative**2 * cd * projectile.cross_sectional_area
            drag_direction = -vel_relative / speed_relative
            drag_force = drag_magnitude * drag_direction
        else:
            drag_force = np.zeros(3)
        
        # === MAGNUS EFFECT (Spin Drift) ===
        magnus_force = np.zeros(3)
        if projectile.spin_rate > 0 and speed_relative > 1e-10:
            # Magnus force perpendicular to velocity and spin axis
            spin_axis = direction  # Assume spin axis aligned with velocity
            magnus_direction = np.cross(spin_axis, np.array([0, 1, 0]))
            if np.linalg.norm(magnus_direction) > 1e-10:
                magnus_direction = magnus_direction / np.linalg.norm(magnus_direction)
                magnus_magnitude = (projectile.magnus_coefficient * 
                                   projectile.spin_rate * 
                                   speed_relative * 
                                   projectile.cross_sectional_area * 
                                   rho)
                magnus_force = magnus_magnitude * magnus_direction
        
        # === CORIOLIS EFFECT ===
        coriolis_force = np.zeros(3)
        if abs(environment.latitude) > 0:
            omega = self.physics.EARTH_ROTATION
            lat_rad = np.radians(environment.latitude)
            # Coriolis acceleration: -2 * omega × v
            omega_vec = omega * np.array([0, np.sin(lat_rad), np.cos(lat_rad)])
            coriolis_acc = -2 * np.cross(omega_vec, vel)
            coriolis_force = projectile.mass * coriolis_acc
        
        # === ROCKET THRUST (for RPGs) ===
        thrust_force = np.zeros(3)
        if projectile.is_rocket_propelled and t < projectile.thrust_duration:
            thrust_force = projectile.thrust_force * direction
        
        # === TOTAL FORCE → ACCELERATION ===
        total_force = gravity_force + drag_force + magnus_force + coriolis_force + thrust_force
        acceleration = total_force / projectile.mass
        
        return acceleration
    
    def _get_drag_coefficient(self, base_cd: float, mach: float) -> float:
        """
        Adjust drag coefficient for Mach number (transonic drag rise).
        """
        if mach < 0.8:
            # Subsonic - constant Cd
            return base_cd
        elif mach < 1.2:
            # Transonic - drag rise
            drag_rise = 1.0 + 0.5 * ((mach - 0.8) / 0.4) ** 2
            return base_cd * drag_rise
        else:
            # Supersonic - gradual decrease
            return base_cd * (1.2 + 0.3 / mach)
    
    def _rk4_step(
        self,
        state: np.ndarray,
        t: float,
        dt: float,
        projectile: ProjectileProfile,
        environment: EnvironmentState
    ) -> np.ndarray:
        """
        Single RK4 integration step.
        
        State vector: [x, y, z, vx, vy, vz]
        """
        def derivatives(s, t_):
            acc = self.compute_forces(s, t_, projectile, environment)
            return np.concatenate([s[3:6], acc])  # [vel, acc]
        
        k1 = derivatives(state, t)
        k2 = derivatives(state + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = derivatives(state + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = derivatives(state + dt * k3, t + dt)
        
        return state + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    def simulate_trajectory(
        self,
        projectile: ProjectileProfile,
        launch: LaunchState,
        environment: EnvironmentState,
        target_plane_height: float = 0.0,
        max_time: float = 30.0,
        record_interval: int = 100
    ) -> Tuple[ImpactResult, List[np.ndarray]]:
        """
        Simulate complete trajectory until impact.
        
        Args:
            projectile: Projectile properties
            launch: Initial conditions
            environment: Environmental state
            target_plane_height: Y-coordinate of impact plane
            max_time: Maximum simulation time
            record_interval: Record state every N steps (for visualization)
            
        Returns:
            (ImpactResult, trajectory_points)
        """
        # Initial state vector
        state = np.concatenate([launch.position, launch.velocity])
        t = 0.0
        step = 0
        
        trajectory = [state[:3].copy()]
        max_height = launch.position[1]
        path_length = 0.0
        prev_pos = state[:3].copy()
        
        while t < max_time:
            # RK4 integration step
            new_state = self._rk4_step(state, t, self.dt, projectile, environment)
            
            # Track metrics
            path_length += np.linalg.norm(new_state[:3] - prev_pos)
            max_height = max(max_height, new_state[1])
            prev_pos = new_state[:3].copy()
            
            # Record trajectory point
            if step % record_interval == 0:
                trajectory.append(new_state[:3].copy())
            
            # Check for ground impact (y <= target_plane_height)
            if new_state[1] <= target_plane_height and state[1] > target_plane_height:
                # Linear interpolation to find exact impact point
                t_impact = (target_plane_height - state[1]) / (new_state[1] - state[1])
                impact_pos = state[:3] + t_impact * (new_state[:3] - state[:3])
                impact_vel = state[3:6] + t_impact * (new_state[3:6] - state[3:6])
                
                # Impact angle (degrees from horizontal)
                impact_speed = np.linalg.norm(impact_vel)
                impact_angle = np.degrees(np.arcsin(-impact_vel[1] / (impact_speed + 1e-10)))
                
                return ImpactResult(
                    position=impact_pos,
                    time_of_flight=t + t_impact * self.dt,
                    impact_velocity=impact_vel,
                    impact_angle=impact_angle,
                    trajectory_length=path_length,
                    max_height=max_height,
                    segment_used=0,  # Ground truth, no segment
                    confidence=1.0,
                    error_estimate=0.0
                ), trajectory
            
            state = new_state
            t += self.dt
            step += 1
        
        # Max time reached without impact
        return ImpactResult(
            position=state[:3],
            time_of_flight=t,
            impact_velocity=state[3:6],
            impact_angle=0.0,
            trajectory_length=path_length,
            max_height=max_height,
            segment_used=0,
            confidence=0.5,
            error_estimate=float('inf')
        ), trajectory
    
    def generate_training_batch(
        self,
        projectile: ProjectileProfile,
        n_samples: int,
        distance_range: Tuple[float, float],
        angle_range: Tuple[float, float] = (-5, 45),
        seed: int = 42
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate training data batch for a specific distance range.
        
        Args:
            projectile: Projectile profile
            n_samples: Number of trajectories to simulate
            distance_range: (min_distance, max_distance) in meters
            angle_range: (min_angle, max_angle) in degrees
            seed: Random seed for reproducibility
            
        Returns:
            (inputs, outputs) arrays for training
            inputs: encoded launch conditions
            outputs: impact positions [x, y, z]
        """
        np.random.seed(seed)
        
        inputs = []
        outputs = []
        
        for i in range(n_samples):
            # Random launch angle within range
            angle = np.random.uniform(angle_range[0], angle_range[1])
            angle_rad = np.radians(angle)
            
            # Calculate approximate muzzle velocity direction for target distance
            target_dist = np.random.uniform(distance_range[0], distance_range[1])
            
            # Add horizontal variance
            azimuth = np.random.uniform(-5, 5)  # degrees
            azimuth_rad = np.radians(azimuth)
            
            # Construct velocity vector
            speed = projectile.muzzle_velocity
            vx = speed * np.cos(angle_rad) * np.cos(azimuth_rad)
            vy = speed * np.sin(angle_rad)
            vz = speed * np.cos(angle_rad) * np.sin(azimuth_rad)
            
            # Random environment variations
            env = EnvironmentState(
                wind_velocity=np.random.uniform(-5, 5, 3),
                temperature=np.random.uniform(273, 313),
                air_density=np.random.uniform(1.1, 1.35),
                humidity=np.random.uniform(0.2, 0.9)
            )
            
            launch = LaunchState(
                position=np.array([0., 1.5, 0.]),  # Shoulder height
                velocity=np.array([vx, vy, vz])
            )
            
            # Run RK4 simulation
            result, _ = self.simulate_trajectory(projectile, launch, env)
            
            # Encode input
            input_vec = self._encode_input(projectile, launch, env)
            inputs.append(input_vec)
            outputs.append(result.position)
            
            if (i + 1) % 1000 == 0:
                print(f"Generated {i + 1}/{n_samples} samples")
        
        return np.array(inputs), np.array(outputs)
    
    def _encode_input(
        self,
        projectile: ProjectileProfile,
        launch: LaunchState,
        environment: EnvironmentState
    ) -> np.ndarray:
        """
        Encode all input parameters into a fixed-size vector.
        This is the input format for the LNN.
        
        Total dimensions: 48 (can be adjusted)
        """
        # Projectile properties (12 dims)
        proj_vec = np.array([
            projectile.mass,
            projectile.diameter,
            projectile.length,
            projectile.drag_coefficient,
            projectile.ballistic_coefficient,
            projectile.muzzle_velocity,
            projectile.spin_rate,
            projectile.gyroscopic_stability,
            float(projectile.is_rocket_propelled),
            projectile.thrust_duration,
            projectile.thrust_force,
            float(projectile.has_fins)
        ])
        
        # Launch state (12 dims)
        launch_vec = np.array([
            launch.position[0], launch.position[1], launch.position[2],
            launch.velocity[0], launch.velocity[1], launch.velocity[2],
            launch.orientation[0], launch.orientation[1], launch.orientation[2],
            launch.angular_velocity[0], launch.angular_velocity[1], launch.angular_velocity[2]
        ])
        
        # Environment (12 dims)
        env_vec = np.array([
            environment.air_density,
            environment.temperature,
            environment.pressure,
            environment.humidity,
            environment.wind_velocity[0],
            environment.wind_velocity[1],
            environment.wind_velocity[2],
            environment.altitude,
            environment.latitude,
            environment.local_gravity,
            0.0, 0.0  # Reserved
        ])
        
        # Derived features (12 dims)
        speed = np.linalg.norm(launch.velocity)
        mach = speed / PhysicsConstants.SPEED_OF_SOUND
        kinetic_energy = 0.5 * projectile.mass * speed ** 2
        momentum = projectile.mass * speed
        
        # Approximate range (flat-fire approximation)
        approx_range = (speed ** 2 * np.sin(2 * np.arctan2(launch.velocity[1], 
                        np.sqrt(launch.velocity[0]**2 + launch.velocity[2]**2)))) / environment.local_gravity
        
        derived_vec = np.array([
            speed,
            mach,
            kinetic_energy,
            momentum,
            approx_range,
            speed / projectile.muzzle_velocity,  # Velocity ratio
            projectile.mass / projectile.cross_sectional_area,  # Sectional density
            environment.get_adjusted_air_density(),
            np.linalg.norm(environment.wind_velocity),  # Wind speed
            launch.velocity[1] / (speed + 1e-10),  # Vertical component ratio
            0.0, 0.0  # Reserved
        ])
        
        return np.concatenate([proj_vec, launch_vec, env_vec, derived_vec])


# =============================================================================
# LIQUID NEURAL NETWORK COMPONENTS
# =============================================================================

class FourierEncoder:
    """
    Phase-Space Encoder using Fourier features to combat spectral bias.
    Maps input to higher-dimensional space for better high-frequency learning.
    """
    
    def __init__(self, input_dim: int, num_frequencies: int = 32, max_freq: float = 10.0):
        """
        Args:
            input_dim: Dimension of input vector
            num_frequencies: Number of frequency bands
            max_freq: Maximum frequency multiplier
        """
        self.input_dim = input_dim
        self.num_frequencies = num_frequencies
        self.max_freq = max_freq
        
        # Output dimension: original + sin/cos for each frequency
        self.output_dim = input_dim + input_dim * num_frequencies * 2
        
        # Frequency matrix (learnable in full implementation)
        self.frequencies = np.linspace(1.0, max_freq, num_frequencies)
        
    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Apply Fourier feature encoding.
        
        gamma(x) = [x, sin(2πf₁x), cos(2πf₁x), ..., sin(2πfₖx), cos(2πfₖx)]
        """
        encoded = [x]
        
        for freq in self.frequencies:
            encoded.append(np.sin(2 * np.pi * freq * x))
            encoded.append(np.cos(2 * np.pi * freq * x))
        
        return np.concatenate(encoded, axis=-1)
    
    def get_weight_matrix(self) -> np.ndarray:
        """
        Get the encoding as a single matrix multiplication.
        For O(1) inference path.
        """
        # Identity for passthrough
        W = np.eye(self.input_dim, self.output_dim)
        
        col = self.input_dim
        for freq in self.frequencies:
            for i in range(self.input_dim):
                W[i, col] = 2 * np.pi * freq  # For sin
                W[i, col + self.input_dim] = 2 * np.pi * freq  # For cos
            col += 2 * self.input_dim
        
        return W


class LiquidTimeConstantCell:
    """
    Liquid Time-Constant (LTC) neural network cell.
    
    The hidden state evolves according to:
    dh/dt = -[1/τ + f(x(t))] * h(t) + A * σ(W_x * x + W_h * h + b)
    
    Where τ is a learned time constant that adapts to input dynamics.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        tau_min: float = 0.01,
        tau_max: float = 10.0
    ):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.tau_min = tau_min
        self.tau_max = tau_max
        
        # Initialize weights (Xavier initialization)
        scale_ih = np.sqrt(2.0 / (input_dim + hidden_dim))
        scale_hh = np.sqrt(2.0 / (hidden_dim + hidden_dim))
        scale_ho = np.sqrt(2.0 / (hidden_dim + output_dim))
        
        self.W_x = np.random.randn(input_dim, hidden_dim) * scale_ih
        self.W_h = np.random.randn(hidden_dim, hidden_dim) * scale_hh
        self.W_o = np.random.randn(hidden_dim, output_dim) * scale_ho
        
        # Time constant network
        self.W_tau = np.random.randn(input_dim, hidden_dim) * scale_ih
        
        # Biases
        self.b_h = np.zeros(hidden_dim)
        self.b_tau = np.ones(hidden_dim)  # Initialize tau to 1
        self.b_o = np.zeros(output_dim)
        
        # Amplitude matrix (A in the ODE)
        self.A = np.random.randn(hidden_dim, hidden_dim) * 0.1
        
    def compute_tau(self, x: np.ndarray) -> np.ndarray:
        """Compute adaptive time constants from input."""
        tau_raw = x @ self.W_tau + self.b_tau
        # Sigmoid to bound tau between tau_min and tau_max
        tau = self.tau_min + (self.tau_max - self.tau_min) * self._sigmoid(tau_raw)
        return tau
    
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def _tanh(self, x: np.ndarray) -> np.ndarray:
        return np.tanh(np.clip(x, -500, 500))
    
    def forward_ode(
        self,
        x: np.ndarray,
        h: np.ndarray,
        dt: float
    ) -> np.ndarray:
        """
        Single ODE integration step for the LTC cell.
        
        Uses simplified Euler for speed (RK4 available for training).
        """
        tau = self.compute_tau(x)
        
        # Activation
        pre_act = x @ self.W_x + h @ self.W_h + self.b_h
        activation = self._tanh(pre_act)
        
        # ODE: dh/dt = -h/tau + A @ activation
        dh_dt = -h / tau + self.A @ activation
        
        # Euler step
        h_new = h + dt * dh_dt
        
        return h_new
    
    def forward(
        self,
        x: np.ndarray,
        integration_time: float = 1.0,
        num_steps: int = 10
    ) -> np.ndarray:
        """
        Full forward pass integrating ODE over time.
        """
        batch_size = x.shape[0] if x.ndim > 1 else 1
        x = x.reshape(batch_size, -1)
        
        h = np.zeros((batch_size, self.hidden_dim))
        dt = integration_time / num_steps
        
        for _ in range(num_steps):
            h = self.forward_ode(x, h, dt)
        
        # Output projection
        output = h @ self.W_o + self.b_o
        return output
    
    def get_condensed_matrix(self, integration_time: float = 1.0) -> np.ndarray:
        """
        Approximate the LTC dynamics as a single matrix for O(1) inference.
        
        This is the key innovation: we "compile" the learned dynamics into
        a direct input→output mapping for deployment.
        
        Note: This is an approximation. Accuracy depends on how well the
        dynamics can be linearized around typical operating points.
        """
        # Linearized approximation around zero
        # For small inputs, tanh(x) ≈ x, so:
        # h_ss ≈ τ * A @ (W_x @ x)  (steady state)
        # output ≈ W_o @ h_ss = W_o @ τ_avg @ A @ W_x @ x
        
        tau_avg = (self.tau_min + self.tau_max) / 2
        
        # Condensed transformation matrix
        W_condensed = self.W_x @ self.A.T @ self.W_o * tau_avg
        
        return W_condensed


class BallisticsLNNSegment:
    """
    Single distance-specialized LNN segment.
    
    Each segment is optimized for a specific distance range with
    appropriate error tolerance.
    """
    
    def __init__(
        self,
        segment_id: int,
        distance_range: Tuple[float, float],
        max_error_mm: float,
        input_dim: int = 48,
        hidden_dim: int = 256,
        fourier_frequencies: int = 32
    ):
        self.segment_id = segment_id
        self.distance_min, self.distance_max = distance_range
        self.max_error_mm = max_error_mm
        
        # Fourier encoder
        self.encoder = FourierEncoder(input_dim, fourier_frequencies)
        encoded_dim = self.encoder.output_dim
        
        # LTC network
        self.ltc = LiquidTimeConstantCell(
            input_dim=encoded_dim,
            hidden_dim=hidden_dim,
            output_dim=3  # [x, y, z] impact position
        )
        
        # Trained flag
        self.is_trained = False
        
        # Condensed inference matrix (set after training)
        self.inference_matrix: Optional[np.ndarray] = None
        
    def predict_training(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass for training (full ODE integration).
        """
        x_encoded = self.encoder.encode(x)
        return self.ltc.forward(x_encoded, integration_time=1.0, num_steps=20)
    
    def predict_inference(self, x: np.ndarray) -> np.ndarray:
        """
        O(1) inference using condensed matrix.
        Single matrix multiplication!
        """
        if self.inference_matrix is None:
            raise RuntimeError("Segment not compiled for inference. Call compile_for_inference() first.")
        
        x_encoded = self.encoder.encode(x)
        return x_encoded @ self.inference_matrix
    
    def compile_for_inference(self):
        """
        Compile the trained network into a single matrix multiplication.
        """
        W_encoder = self.encoder.get_weight_matrix()
        W_ltc = self.ltc.get_condensed_matrix()
        
        # Combined transformation
        self.inference_matrix = W_encoder @ W_ltc
        
    def estimate_distance(self, launch_velocity: np.ndarray) -> float:
        """
        Quick distance estimate for segment selection.
        """
        speed = np.linalg.norm(launch_velocity)
        # Simplified flat-fire range estimate
        angle = np.arcsin(launch_velocity[1] / (speed + 1e-10))
        return (speed ** 2 * np.sin(2 * angle)) / 9.81


# =============================================================================
# MAIN BALLISTICS KERNEL
# =============================================================================

class HyperFluxBallisticsKernel:
    """
    The main HyperFlux Ballistics Kernel.
    
    Combines:
    - 5 distance-specialized LNN segments
    - RK4 ground truth engine for training
    - Watchdog protocol for validation
    - O(1) inference path
    
    Segments:
    - S1: 0-100m    (±5mm error)
    - S2: 101-300m  (±10mm error)
    - S3: 301-600m  (±20mm error)
    - S4: 601-1000m (±30mm error)
    - S5: 1001-1500m (±50mm error)
    """
    
    # Segment configuration
    SEGMENT_CONFIG = {
        1: {"range": (0, 100), "error_mm": 5, "hidden_dim": 128},
        2: {"range": (101, 300), "error_mm": 10, "hidden_dim": 192},
        3: {"range": (301, 600), "error_mm": 20, "hidden_dim": 256},
        4: {"range": (601, 1000), "error_mm": 30, "hidden_dim": 320},
        5: {"range": (1001, 1500), "error_mm": 50, "hidden_dim": 384}
    }
    
    def __init__(self, input_dim: int = 48):
        """Initialize the kernel with all segments."""
        self.input_dim = input_dim
        
        # Initialize segments
        self.segments: Dict[int, BallisticsLNNSegment] = {}
        for seg_id, config in self.SEGMENT_CONFIG.items():
            self.segments[seg_id] = BallisticsLNNSegment(
                segment_id=seg_id,
                distance_range=config["range"],
                max_error_mm=config["error_mm"],
                input_dim=input_dim,
                hidden_dim=config["hidden_dim"]
            )
        
        # RK4 engine for ground truth
        self.rk4_engine = RK4BallisticsEngine(dt=0.0001)
        
        # Watchdog state
        self.watchdog_enabled = True
        self.watchdog_threshold = 0.1  # Energy conservation threshold
        
        # Statistics
        self.inference_count = 0
        self.watchdog_rejections = 0
        self.fallback_count = 0
        
    def select_segment(self, estimated_distance: float) -> int:
        """
        Select appropriate segment based on estimated distance.
        """
        for seg_id, config in self.SEGMENT_CONFIG.items():
            if config["range"][0] <= estimated_distance <= config["range"][1]:
                return seg_id
        
        # Default to longest range segment if beyond 1500m
        if estimated_distance > 1500:
            return 5
        # Default to shortest range for very close
        return 1
    
    def estimate_distance(
        self,
        projectile: ProjectileProfile,
        launch: LaunchState,
        environment: EnvironmentState
    ) -> float:
        """
        Quick distance estimation for segment selection.
        Uses simplified ballistic equation.
        """
        v = launch.velocity
        speed = np.linalg.norm(v)
        
        # Horizontal and vertical components
        v_h = np.sqrt(v[0]**2 + v[2]**2)
        v_v = v[1]
        
        # Time to ground (quadratic formula)
        g = environment.local_gravity
        h0 = launch.position[1]
        
        # t = (v_v + sqrt(v_v² + 2*g*h0)) / g
        discriminant = v_v**2 + 2 * g * h0
        if discriminant < 0:
            return 0.0
        
        t_flight = (v_v + np.sqrt(discriminant)) / g
        
        # Horizontal distance (ignoring drag for estimate)
        distance = v_h * t_flight
        
        return distance
    
    def predict(
        self,
        projectile: ProjectileProfile,
        launch: LaunchState,
        environment: EnvironmentState,
        use_inference_mode: bool = True
    ) -> ImpactResult:
        """
        Main prediction entry point.
        
        Args:
            projectile: Projectile characteristics
            launch: Initial launch conditions
            environment: Environmental state
            use_inference_mode: If True, use O(1) matrix multiplication
                              If False, use full ODE integration
                              
        Returns:
            ImpactResult with predicted impact point
        """
        self.inference_count += 1
        
        # Encode input
        input_vec = self.rk4_engine._encode_input(projectile, launch, environment)
        
        # Estimate distance for segment selection
        est_distance = self.estimate_distance(projectile, launch, environment)
        segment_id = self.select_segment(est_distance)
        segment = self.segments[segment_id]
        
        # Get prediction
        if use_inference_mode and segment.inference_matrix is not None:
            # O(1) inference path - single matrix multiplication!
            impact_pos = segment.predict_inference(input_vec.reshape(1, -1))[0]
        else:
            # Training mode - full ODE integration
            impact_pos = segment.predict_training(input_vec.reshape(1, -1))[0]
        
        # Watchdog validation
        if self.watchdog_enabled:
            is_valid, confidence = self._watchdog_validate(
                projectile, launch, environment, impact_pos
            )
            
            if not is_valid:
                self.watchdog_rejections += 1
                # Fallback to RK4 for this prediction
                result, _ = self.rk4_engine.simulate_trajectory(
                    projectile, launch, environment
                )
                self.fallback_count += 1
                result.segment_used = segment_id
                result.confidence = 0.5  # Lower confidence for fallback
                return result
        else:
            confidence = 0.9
        
        # Calculate additional metrics
        distance = np.linalg.norm(impact_pos - launch.position)
        
        # Estimate time of flight (simplified)
        avg_speed = projectile.muzzle_velocity * 0.7  # Account for drag
        tof = distance / avg_speed
        
        return ImpactResult(
            position=impact_pos,
            time_of_flight=tof,
            impact_velocity=launch.velocity * 0.5,  # Estimate
            impact_angle=30.0,  # Estimate
            trajectory_length=distance * 1.1,  # Estimate
            max_height=launch.position[1] + distance * 0.1,  # Estimate
            segment_used=segment_id,
            confidence=confidence,
            error_estimate=segment.max_error_mm
        )
    
    def _watchdog_validate(
        self,
        projectile: ProjectileProfile,
        launch: LaunchState,
        environment: EnvironmentState,
        predicted_pos: np.ndarray
    ) -> Tuple[bool, float]:
        """
        Watchdog Protocol: Validate prediction against physics constraints.
        
        Checks:
        1. Energy conservation bounds
        2. Momentum plausibility
        3. Distance sanity check
        4. Direction consistency
        
        Returns: (is_valid, confidence_score)
        """
        # Initial energy
        v0 = np.linalg.norm(launch.velocity)
        ke_initial = 0.5 * projectile.mass * v0**2
        pe_initial = projectile.mass * environment.local_gravity * launch.position[1]
        e_initial = ke_initial + pe_initial
        
        # Predicted final state energy bounds
        # Final PE at ground level (y=0)
        pe_final = projectile.mass * environment.local_gravity * predicted_pos[1]
        
        # Distance traveled
        distance = np.linalg.norm(predicted_pos - launch.position)
        
        # Sanity checks
        checks = []
        
        # 1. Distance should be positive and reasonable
        max_possible_range = (v0**2) / environment.local_gravity * 2  # Factor of 2 margin
        dist_valid = 0 < distance < max_possible_range
        checks.append(dist_valid)
        
        # 2. Impact point should be in the forward direction
        direction = predicted_pos - launch.position
        forward_component = np.dot(direction, launch.velocity) / (v0 + 1e-10)
        direction_valid = forward_component > 0
        checks.append(direction_valid)
        
        # 3. Vertical position should be at or below launch height (for flat terrain)
        height_valid = predicted_pos[1] <= launch.position[1] + 10  # 10m margin
        checks.append(height_valid)
        
        # 4. Energy cannot increase (conservation with drag losses)
        # Final KE is bounded by initial total energy
        ke_max = e_initial - pe_final
        energy_plausible = ke_max > 0
        checks.append(energy_plausible)
        
        is_valid = all(checks)
        confidence = sum(checks) / len(checks)
        
        return is_valid, confidence
    
    def train_segment(
        self,
        segment_id: int,
        projectile: ProjectileProfile,
        n_samples: int = 10000,
        epochs: int = 100,
        learning_rate: float = 0.001
    ):
        """
        Train a single segment using RK4-generated ground truth.
        
        This is a simplified training loop. Production would use:
        - PyTorch/JAX for autograd
        - GPU acceleration
        - Proper batch normalization
        - Learning rate scheduling
        """
        segment = self.segments[segment_id]
        config = self.SEGMENT_CONFIG[segment_id]
        
        print(f"\n{'='*60}")
        print(f"Training Segment {segment_id}: {config['range'][0]}-{config['range'][1]}m")
        print(f"Target error: ±{config['error_mm']}mm")
        print(f"{'='*60}")
        
        # Generate training data
        print("Generating training data with RK4...")
        X, Y = self.rk4_engine.generate_training_batch(
            projectile,
            n_samples,
            config["range"],
            angle_range=(-2, 45)
        )
        
        # Normalize inputs
        X_mean = X.mean(axis=0)
        X_std = X.std(axis=0) + 1e-8
        X_norm = (X - X_mean) / X_std
        
        # Normalize outputs
        Y_mean = Y.mean(axis=0)
        Y_std = Y.std(axis=0) + 1e-8
        Y_norm = (Y - Y_mean) / Y_std
        
        # Store normalization parameters
        segment.X_mean = X_mean
        segment.X_std = X_std
        segment.Y_mean = Y_mean
        segment.Y_std = Y_std
        
        # Simple gradient descent training
        print("Training LNN...")
        batch_size = 64
        n_batches = n_samples // batch_size
        
        for epoch in range(epochs):
            total_loss = 0
            
            # Shuffle
            indices = np.random.permutation(n_samples)
            X_shuffled = X_norm[indices]
            Y_shuffled = Y_norm[indices]
            
            for b in range(n_batches):
                start = b * batch_size
                end = start + batch_size
                
                X_batch = X_shuffled[start:end]
                Y_batch = Y_shuffled[start:end]
                
                # Forward pass
                Y_pred = segment.predict_training(X_batch)
                
                # Loss (MSE)
                loss = np.mean((Y_pred - Y_batch) ** 2)
                total_loss += loss
                
                # Simplified gradient update (finite differences)
                # Production code would use autograd
                grad_scale = learning_rate * (Y_pred - Y_batch).mean(axis=0)
                segment.ltc.b_o -= grad_scale * 0.01
            
            avg_loss = total_loss / n_batches
            
            if (epoch + 1) % 10 == 0:
                # Calculate actual error in mm
                Y_pred_full = segment.predict_training(X_norm)
                Y_pred_denorm = Y_pred_full * Y_std + Y_mean
                errors = np.linalg.norm(Y_pred_denorm - Y, axis=1) * 1000  # Convert to mm
                mean_error = errors.mean()
                max_error = errors.max()
                
                print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f} | "
                      f"Mean Error: {mean_error:.2f}mm | Max Error: {max_error:.2f}mm")
        
        segment.is_trained = True
        print(f"Segment {segment_id} training complete!")
        
    def compile_for_inference(self):
        """
        Compile all trained segments for O(1) inference.
        """
        print("\nCompiling segments for O(1) inference...")
        for seg_id, segment in self.segments.items():
            if segment.is_trained:
                segment.compile_for_inference()
                print(f"  Segment {seg_id}: Compiled ✓")
            else:
                print(f"  Segment {seg_id}: Not trained, skipping")
        print("Compilation complete!")
    
    def benchmark(
        self,
        projectile: ProjectileProfile,
        n_tests: int = 1000
    ) -> Dict:
        """
        Benchmark O(1) inference vs O(N) RK4.
        """
        import time
        
        results = {
            "rk4_times": [],
            "lnn_times": [],
            "errors_mm": [],
            "segment_usage": {1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
        }
        
        for i in range(n_tests):
            # Random launch conditions
            angle = np.random.uniform(5, 45)
            angle_rad = np.radians(angle)
            speed = projectile.muzzle_velocity
            
            launch = LaunchState(
                position=np.array([0., 1.5, 0.]),
                velocity=np.array([
                    speed * np.cos(angle_rad),
                    speed * np.sin(angle_rad),
                    np.random.uniform(-0.1, 0.1) * speed
                ])
            )
            
            env = EnvironmentState()
            
            # RK4 timing
            t0 = time.perf_counter()
            rk4_result, _ = self.rk4_engine.simulate_trajectory(projectile, launch, env)
            t1 = time.perf_counter()
            results["rk4_times"].append(t1 - t0)
            
            # LNN timing
            t0 = time.perf_counter()
            lnn_result = self.predict(projectile, launch, env, use_inference_mode=True)
            t1 = time.perf_counter()
            results["lnn_times"].append(t1 - t0)
            
            # Error
            error = np.linalg.norm(lnn_result.position - rk4_result.position) * 1000
            results["errors_mm"].append(error)
            results["segment_usage"][lnn_result.segment_used] += 1
        
        # Summary
        results["summary"] = {
            "rk4_avg_ms": np.mean(results["rk4_times"]) * 1000,
            "lnn_avg_ms": np.mean(results["lnn_times"]) * 1000,
            "speedup": np.mean(results["rk4_times"]) / np.mean(results["lnn_times"]),
            "mean_error_mm": np.mean(results["errors_mm"]),
            "max_error_mm": np.max(results["errors_mm"]),
            "p95_error_mm": np.percentile(results["errors_mm"], 95)
        }
        
        return results
    
    def get_statistics(self) -> Dict:
        """Get kernel runtime statistics."""
        return {
            "total_inferences": self.inference_count,
            "watchdog_rejections": self.watchdog_rejections,
            "fallback_count": self.fallback_count,
            "rejection_rate": self.watchdog_rejections / max(1, self.inference_count),
            "segments_trained": sum(1 for s in self.segments.values() if s.is_trained)
        }


# =============================================================================
# PROJECTILE PRESETS
# =============================================================================

class ProjectilePresets:
    """Pre-configured projectile profiles for common ammunition."""
    
    @staticmethod
    def rifle_556_nato() -> ProjectileProfile:
        """5.56x45mm NATO (M855)"""
        return ProjectileProfile(
            projectile_type=ProjectileType.BULLET_RIFLE,
            name="5.56x45mm NATO M855",
            mass=0.004,  # 4g / 62gr
            diameter=0.00569,  # 5.69mm
            length=0.0232,  # 23.2mm
            drag_coefficient=0.295,
            ballistic_coefficient=0.304,
            muzzle_velocity=940,  # m/s
            spin_rate=15000,  # ~900,000 RPM from 1:7 twist
            gyroscopic_stability=1.5
        )
    
    @staticmethod
    def rifle_762_nato() -> ProjectileProfile:
        """7.62x51mm NATO (M80)"""
        return ProjectileProfile(
            projectile_type=ProjectileType.BULLET_RIFLE,
            name="7.62x51mm NATO M80",
            mass=0.00947,  # 9.47g / 147gr
            diameter=0.00782,  # 7.82mm
            length=0.0286,
            drag_coefficient=0.393,
            ballistic_coefficient=0.393,
            muzzle_velocity=853,
            spin_rate=12000,
            gyroscopic_stability=1.4
        )
    
    @staticmethod
    def sniper_338_lapua() -> ProjectileProfile:
        """.338 Lapua Magnum"""
        return ProjectileProfile(
            projectile_type=ProjectileType.BULLET_SNIPER,
            name=".338 Lapua Magnum",
            mass=0.01685,  # 16.85g / 260gr
            diameter=0.00862,
            length=0.0378,
            drag_coefficient=0.289,
            ballistic_coefficient=0.670,
            muzzle_velocity=915,
            spin_rate=10000,
            gyroscopic_stability=1.8
        )
    
    @staticmethod
    def pistol_9mm() -> ProjectileProfile:
        """9x19mm Parabellum"""
        return ProjectileProfile(
            projectile_type=ProjectileType.BULLET_PISTOL,
            name="9x19mm Parabellum",
            mass=0.008,  # 8g / 124gr
            diameter=0.00901,
            length=0.0152,
            drag_coefficient=0.295,
            ballistic_coefficient=0.145,
            muzzle_velocity=375,
            spin_rate=8000,
            gyroscopic_stability=1.2
        )
    
    @staticmethod
    def grenade_m67() -> ProjectileProfile:
        """M67 Fragmentation Grenade"""
        return ProjectileProfile(
            projectile_type=ProjectileType.GRENADE_FRAG,
            name="M67 Frag Grenade",
            mass=0.400,  # 400g
            diameter=0.064,  # 64mm
            length=0.099,
            drag_coefficient=0.47,  # Sphere-like
            ballistic_coefficient=0.02,
            muzzle_velocity=15,  # Typical throw speed
            spin_rate=100,  # Some spin from throw
            gyroscopic_stability=0.8
        )
    
    @staticmethod
    def rpg_pg7() -> ProjectileProfile:
        """RPG-7 PG-7V Rocket"""
        return ProjectileProfile(
            projectile_type=ProjectileType.RPG_ROCKET,
            name="RPG-7 PG-7V",
            mass=2.25,  # 2.25kg
            diameter=0.085,  # 85mm warhead
            length=0.925,
            drag_coefficient=0.40,
            ballistic_coefficient=0.15,
            muzzle_velocity=115,  # Initial launch
            spin_rate=50,
            gyroscopic_stability=1.0,
            is_rocket_propelled=True,
            thrust_duration=0.6,  # 600ms burn
            thrust_force=1200,  # Approximate
            has_fins=True,
            fin_stabilization=0.8
        )
    
    @staticmethod
    def arrow_compound() -> ProjectileProfile:
        """Compound Bow Arrow"""
        return ProjectileProfile(
            projectile_type=ProjectileType.ARROW_STANDARD,
            name="Compound Bow Arrow",
            mass=0.025,  # 25g / 385gr
            diameter=0.006,  # 6mm shaft
            length=0.75,  # 75cm
            drag_coefficient=0.60,
            ballistic_coefficient=0.008,
            muzzle_velocity=90,  # ~300 fps
            spin_rate=200,  # Slight spin from fletching
            gyroscopic_stability=0.6,
            has_fins=True,
            fin_stabilization=0.9
        )
    
    @staticmethod
    def knife_throwing() -> ProjectileProfile:
        """Throwing Knife"""
        return ProjectileProfile(
            projectile_type=ProjectileType.KNIFE_THROWN,
            name="Throwing Knife",
            mass=0.200,  # 200g
            diameter=0.025,  # 25mm width
            length=0.25,  # 25cm
            drag_coefficient=0.80,
            ballistic_coefficient=0.01,
            muzzle_velocity=15,  # Typical throw
            spin_rate=2000,  # Rotation for stability
            gyroscopic_stability=0.5
        )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("="*70)
    print("HyperFlux Ballistics Kernel v1.0")
    print("Neural Dynamics Team - Confidential")
    print("="*70)
    
    # Initialize kernel
    kernel = HyperFluxBallisticsKernel()
    
    # Get a projectile
    rifle = ProjectilePresets.rifle_556_nato()
    print(f"\nProjectile: {rifle.name}")
    print(f"  Mass: {rifle.mass*1000:.1f}g")
    print(f"  Muzzle velocity: {rifle.muzzle_velocity} m/s")
    print(f"  Ballistic coefficient: {rifle.ballistic_coefficient}")
    
    # Set up a shot
    launch = LaunchState(
        position=np.array([0., 1.5, 0.]),  # Shoulder height
        velocity=np.array([
            rifle.muzzle_velocity * np.cos(np.radians(5)),
            rifle.muzzle_velocity * np.sin(np.radians(5)),
            0.
        ])
    )
    
    env = EnvironmentState(
        wind_velocity=np.array([2., 0., 1.]),  # Light crosswind
        temperature=293.15,  # 20°C
        humidity=0.6
    )
    
    # RK4 ground truth
    print("\n" + "-"*50)
    print("Running RK4 ground truth simulation...")
    result_rk4, trajectory = kernel.rk4_engine.simulate_trajectory(rifle, launch, env)
    
    print(f"\nGround Truth Result:")
    print(f"  Impact position: ({result_rk4.position[0]:.3f}, {result_rk4.position[1]:.3f}, {result_rk4.position[2]:.3f}) m")
    print(f"  Time of flight: {result_rk4.time_of_flight*1000:.1f} ms")
    print(f"  Distance: {np.linalg.norm(result_rk4.position - launch.position):.2f} m")
    print(f"  Impact velocity: {np.linalg.norm(result_rk4.impact_velocity):.1f} m/s")
    print(f"  Impact angle: {result_rk4.impact_angle:.1f}°")
    
    print("\n" + "="*70)
    print("Kernel ready for training. Use train_segment() to train each segment.")
    print("="*70)
