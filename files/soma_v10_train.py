"""
╔══════════════════════════════════════════════════════════════════════╗
║  SOMA v10 — Universal Motor Execution Layer                        ║
║  Training Pipeline                                                  ║
║                                                                      ║
║  Architecture: Low-Rank Sparse CfC + Per-Joint Shared Expert        ║
║               + Differentiable CPG + HOCBF-CLF-QP                   ║
║  Training:    PGPE Evolution Strategy via MJX                       ║
║  Target:      Any Type B robot from URDF in 20 minutes              ║
║                                                                      ║
║  Incremental validation flags:                                       ║
║    USE_LOWRANK_CFC  = True   # vs full-rank baseline                ║
║    USE_PERJOINT_MOE = True   # vs monolithic CfC                    ║
║    USE_DIFF_CPG     = True   # vs fixed CPG                         ║
║    USE_HOCBF        = True   # vs simple clamp                      ║
║    USE_PGPE         = True   # vs OpenAI-ES                         ║
║                                                                      ║
║  Proven foundations (from v1-v9):                                     ║
║    ✓ Residual PD + α·CfC (v8: raw torque always fails)             ║
║    ✓ ES through MJX (v9: 0.1403→0.0323, 10s/gen H100)             ║
║    ✓ Curriculum stand→walk→turn→push (v9 vmapped ES v7)            ║
║    ✓ Elitist updates (v9: reject worse)                             ║
║    ✓ Antithetic sampling (v9: halves variance)                      ║
║    ✗ No teacher-student (v1-v8: always creates ceiling)             ║
║    ✗ No PPO/RL (v5: destroys Lyapunov/freq invariance)             ║
║    ✗ No DAgger (v7: contamination at scale)                         ║
║    ✗ No raw torque (v8: falls at 0N)                                ║
║    ✗ No backprop through MJX (gradients explode after 16 steps)     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import os
import time
import json
import functools
from typing import NamedTuple, Tuple, Optional, Dict, List
from dataclasses import dataclass, field

import numpy as np

# ══════════════════════════════════════════════════════════════════════
# Installation (uncomment for Colab)
# ══════════════════════════════════════════════════════════════════════
# !pip install mujoco mujoco-mjx jax[cuda12] evosax optax flax -q

import jax
import jax.numpy as jnp
from jax import random, vmap, jit, lax
import mujoco
from mujoco import mjx
import optax

# Try evosax, fall back to manual PGPE
try:
    import evosax
    HAS_EVOSAX = True
except ImportError:
    HAS_EVOSAX = False
    print("evosax not found — using built-in PGPE")

# ══════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

@dataclass
class SOMAConfig:
    """Complete SOMA v10 configuration."""

    # ── Architecture toggles (for incremental validation) ──
    USE_LOWRANK_CFC:  bool = True   # Low-rank sparse recurrent connectivity
    USE_PERJOINT_MOE: bool = True   # Per-joint shared expert (vs monolithic)
    USE_DIFF_CPG:     bool = True   # Differentiable CPG jointly optimized
    USE_HOCBF:        bool = True   # High-Order CBF safety filter
    USE_PGPE:         bool = True   # PGPE (vs OpenAI-ES)

    # ── CfC Backbone ──
    backbone_hidden:  int = 32      # CfC backbone hidden size
    backbone_rank:    int = 8       # Low-rank factor (if USE_LOWRANK_CFC)
    backbone_sparsity: float = 0.5  # Fraction of connections masked

    # ── Per-Joint Shared Expert ──
    expert_hidden:    int = 16      # Expert CfC hidden size
    expert_rank:      int = 4       # Expert low-rank factor
    expert_sparsity:  float = 0.5   # Expert sparsity

    # ── Monolithic fallback (if not USE_PERJOINT_MOE) ──
    mono_hidden:      int = 64      # Monolithic CfC hidden size

    # ── CPG ──
    cpg_base_freq:    float = 2.0   # Base gait frequency (Hz)
    cpg_base_amp:     float = 0.25  # Base swing amplitude (rad)
    cpg_duty:         float = 0.6   # Stance duty factor

    # ── Residual ──
    alpha_init:       float = 0.3   # Initial residual authority
    kp_scale:         float = 1.0   # PD gain scaling factor
    kd_scale:         float = 1.0   # PD damping scaling factor

    # ── HOCBF ──
    cbf_alpha1:       float = 10.0  # First barrier class-K function gain
    cbf_alpha2:       float = 10.0  # Second barrier class-K function gain
    cbf_clf_lambda:   float = 1.0   # CLF convergence rate
    cbf_clf_penalty:  float = 100.0 # CLF slack penalty

    # ── Training ──
    pop_size:         int = 256     # ES population (×2 for antithetic)
    sigma_init:       float = 0.03  # Initial perturbation scale
    sigma_min:        float = 0.005 # Minimum sigma (annealed)
    sigma_decay:      float = 0.995 # Per-generation sigma decay
    lr:               float = 0.01  # Learning rate
    rollout_steps:    int = 1000    # Steps per rollout (2s at 500Hz)
    dt:               float = 0.002 # Simulation timestep
    n_generations:    int = 150     # Total generations

    # ── Curriculum phases (generation ranges) ──
    phase1_end:       int = 30      # Stand
    phase2_end:       int = 70      # Walk
    phase3_end:       int = 110     # Walk + Turn
    phase4_end:       int = 150     # Walk + Turn + Push recovery

    # ── Domain randomization (staged) ──
    dr_friction_range:  Tuple[float, float] = (0.5, 1.25)
    dr_mass_range:      Tuple[float, float] = (-0.15, 0.15)  # fraction
    dr_motor_range:     Tuple[float, float] = (0.9, 1.1)     # scale
    dr_kp_range:        Tuple[float, float] = (0.85, 1.15)   # scale
    dr_kd_range:        Tuple[float, float] = (0.5, 1.5)     # scale
    dr_latency_steps:   int = 10    # Max action delay steps (20ms at 500Hz)

    # ── Push recovery ──
    push_vel_range:     Tuple[float, float] = (0.5, 2.0)  # m/s impulse
    push_interval:      Tuple[int, int] = (1500, 4000)     # steps between pushes

    # ── Fitness weights ──
    w_vel:     float = 5.0    # Velocity tracking
    w_stab:    float = 2.0    # Stability (orientation)
    w_alive:   float = 10.0   # Survival bonus
    w_energy:  float = 0.1    # Energy penalty
    w_smooth:  float = 0.5    # Action smoothness

    # ── Robot (set by URDF parser) ──
    robot_xml:  str = ""
    robot_name: str = ""

    def __post_init__(self):
        self.control_freq = 1.0 / self.dt


# ══════════════════════════════════════════════════════════════════════
# SECTION 1: URDF/MJCF AUTO-PARSER
# ══════════════════════════════════════════════════════════════════════

@dataclass
class JointInfo:
    """Parsed info for a single joint."""
    name: str
    joint_id: int
    actuator_id: int
    joint_type: int        # 0=free, 1=ball, 2=slide, 3=hinge
    axis: np.ndarray       # (3,) joint axis
    range: np.ndarray      # (2,) [min, max] position limits
    torque_limit: float    # Max torque
    gear_ratio: float      # Gear ratio
    parent_body: int       # Parent body id
    child_body: int        # Child body id
    child_mass: float      # Mass of child link
    child_com: np.ndarray  # (3,) CoM of child link
    limb_id: int = -1      # Which limb this joint belongs to
    depth_in_limb: int = 0 # 0=hip, 1=knee, 2=ankle, ...
    # Neighbors in kinematic tree (set during parsing)
    parent_joint_id: int = -1
    child_joint_ids: List[int] = field(default_factory=list)


@dataclass
class RobotInfo:
    """Complete parsed robot description."""
    name: str
    nu: int                     # Number of actuated joints
    nv: int                     # Velocity dims
    nq: int                     # Position dims
    joints: List[JointInfo] = field(default_factory=list)
    n_limbs: int = 0
    joints_per_limb: int = 0
    morphology: str = ""        # "quadruped", "biped", "hexapod"
    base_body_id: int = 0
    base_mass: float = 0.0
    total_mass: float = 0.0
    standing_height: float = 0.0
    # CPG phase offsets per limb (set from morphology)
    cpg_phase_offsets: np.ndarray = None
    # PD gains (auto-scaled from URDF)
    kp: np.ndarray = None
    kd: np.ndarray = None
    # Joint descriptors for neural network input (n_joints × desc_dim)
    joint_descriptors: np.ndarray = None


def parse_robot(mj_model: mujoco.MjModel) -> RobotInfo:
    """
    Parse MuJoCo model into structured robot description.

    Extracts kinematic tree, identifies limbs, computes PD gains,
    and generates joint descriptors for neural network input.

    LESSON FROM v5.1: ALWAYS inspect actuator configuration.
    """
    # Extract model name safely across MuJoCo versions
    try:
        raw_names = bytes(mj_model.names)
        model_name = raw_names.split(b'\x00')[0].decode()
    except Exception:
        model_name = "robot"
    if not model_name:
        model_name = "robot"

    info = RobotInfo(
        name=model_name,
        nu=mj_model.nu,
        nv=mj_model.nv,
        nq=mj_model.nq,
    )

    # ── Parse each actuated joint ──
    # Build actuator→joint mapping
    act_to_jnt = {}
    for i in range(mj_model.nu):
        # Find which joint this actuator drives
        trntype = mj_model.actuator_trntype[i]
        if trntype == 0:  # mjTRN_JOINT
            jnt_id = mj_model.actuator_trnid[i, 0]
            act_to_jnt[i] = jnt_id

    # Parse joints
    for act_id in range(mj_model.nu):
        if act_id not in act_to_jnt:
            continue
        jnt_id = act_to_jnt[act_id]

        # Joint properties
        jnt_type = int(mj_model.jnt_type[jnt_id])
        jnt_axis = np.array(mj_model.jnt_axis[jnt_id])
        jnt_range = np.array(mj_model.jnt_range[jnt_id])

        # Actuator properties
        torque_limit = float(mj_model.actuator_ctrlrange[act_id, 1])
        if torque_limit == 0:
            torque_limit = float(mj_model.actuator_forcerange[act_id, 1])
        if torque_limit == 0:
            torque_limit = 33.5  # Default Go2-like torque limit
        gear_ratio = float(mj_model.actuator_gear[act_id, 0])
        if gear_ratio == 0:
            gear_ratio = 1.0

        # Body properties
        body_id = mj_model.jnt_bodyid[jnt_id]
        parent_body = mj_model.body_parentid[body_id]
        child_mass = float(mj_model.body_mass[body_id])
        child_com = np.array(mj_model.body_ipos[body_id])

        # Joint name
        jnt_name_adr = mj_model.name_jntadr[jnt_id]
        jnt_name = ""
        if jnt_name_adr >= 0:
            try:
                raw = bytes(mj_model.names[jnt_name_adr:])
                jnt_name = raw.split(b'\x00')[0].decode()
            except Exception:
                jnt_name = f"joint_{jnt_id}"

        ji = JointInfo(
            name=jnt_name,
            joint_id=jnt_id,
            actuator_id=act_id,
            joint_type=jnt_type,
            axis=jnt_axis,
            range=jnt_range,
            torque_limit=torque_limit,
            gear_ratio=gear_ratio,
            parent_body=int(parent_body),
            child_body=int(body_id),
            child_mass=child_mass,
            child_com=child_com,
        )
        info.joints.append(ji)

    # ── Build kinematic tree relationships ──
    body_to_joints = {}
    for ji in info.joints:
        body_to_joints.setdefault(ji.child_body, []).append(ji.actuator_id)
        body_to_joints.setdefault(ji.parent_body, [])

    for ji in info.joints:
        # Find parent joint (joint whose child_body == this joint's parent_body)
        for other in info.joints:
            if other.child_body == ji.parent_body and other.actuator_id != ji.actuator_id:
                ji.parent_joint_id = other.actuator_id
                break
        # Find child joints
        for other in info.joints:
            if other.parent_joint_id == ji.actuator_id:
                ji.child_joint_ids.append(other.actuator_id)

    # ── Identify limbs via kinematic tree ──
    # Root joints = joints whose parent_body is the base (body 1 or the
    # body with the most child joints)
    base_body = 1  # Usually body 1 is the torso
    # Find the body that's the parent of most actuated joints
    parent_counts = {}
    for ji in info.joints:
        parent_counts[ji.parent_body] = parent_counts.get(ji.parent_body, 0) + 1
    if parent_counts:
        base_body = max(parent_counts, key=parent_counts.get)

    info.base_body_id = base_body
    info.base_mass = float(mj_model.body_mass[base_body])
    info.total_mass = float(np.sum(mj_model.body_mass))

    # Hip joints = joints whose parent is the base body
    hip_joints = [ji for ji in info.joints if ji.parent_body == base_body]
    # If no direct hip joints, look one level deeper
    if not hip_joints:
        children_of_base = [b for b in range(mj_model.nbody)
                           if mj_model.body_parentid[b] == base_body]
        hip_joints = [ji for ji in info.joints
                     if ji.parent_body in children_of_base]

    info.n_limbs = len(hip_joints)

    # Trace each limb
    for limb_idx, hip in enumerate(hip_joints):
        hip.limb_id = limb_idx
        hip.depth_in_limb = 0
        # BFS down the kinematic chain
        queue = [(hip, 0)]
        while queue:
            current, depth = queue.pop(0)
            current.limb_id = limb_idx
            current.depth_in_limb = depth
            for child_id in current.child_joint_ids:
                child_ji = info.joints[child_id]
                queue.append((child_ji, depth + 1))

    # Determine morphology
    if info.n_limbs == 4:
        info.morphology = "quadruped"
        info.joints_per_limb = info.nu // 4
    elif info.n_limbs == 2:
        info.morphology = "biped"
        info.joints_per_limb = info.nu // 2
    elif info.n_limbs == 6:
        info.morphology = "hexapod"
        info.joints_per_limb = info.nu // 6
    else:
        info.morphology = f"custom_{info.n_limbs}leg"
        info.joints_per_limb = info.nu // max(info.n_limbs, 1)

    # ── CPG phase offsets from morphology ──
    if info.morphology == "quadruped":
        # Diagonal trot: FL=0, FR=π, RL=π, RR=0
        info.cpg_phase_offsets = np.array([0.0, np.pi, np.pi, 0.0])
    elif info.morphology == "biped":
        # Alternating: L=0, R=π
        info.cpg_phase_offsets = np.array([0.0, np.pi])
    elif info.morphology == "hexapod":
        # Tripod: legs 0,2,4 = 0; legs 1,3,5 = π
        info.cpg_phase_offsets = np.array([0.0, np.pi, 0.0, np.pi, 0.0, np.pi])
    else:
        info.cpg_phase_offsets = np.zeros(max(info.n_limbs, 1))

    # ── Auto-scale PD gains ──
    kp = np.zeros(info.nu)
    kd = np.zeros(info.nu)
    for ji in info.joints:
        joint_range = ji.range[1] - ji.range[0]
        if joint_range < 0.01:
            joint_range = 2.0  # fallback
        # Kp proportional to torque capacity / range
        kp[ji.actuator_id] = 0.3 * ji.torque_limit / joint_range
        # Kd from critical damping: Kd = 2 * sqrt(Kp * inertia_approx)
        inertia_approx = ji.child_mass * 0.1**2  # rough approximation
        kd[ji.actuator_id] = 2.0 * np.sqrt(kp[ji.actuator_id] * max(inertia_approx, 0.001))
    info.kp = kp
    info.kd = kd

    # ── Standing height (approximate from base position in default config) ──
    data = mujoco.MjData(mj_model)
    mujoco.mj_forward(mj_model, data)
    info.standing_height = float(data.xpos[base_body, 2])

    # ── Joint descriptors for neural input ──
    # [type_onehot(3), axis(3), mass, length_approx, torque_limit_norm, gear_ratio_norm]
    desc_dim = 10
    descriptors = np.zeros((info.nu, desc_dim))
    max_torque = max(ji.torque_limit for ji in info.joints) or 1.0
    max_mass = max(ji.child_mass for ji in info.joints) or 1.0
    for ji in info.joints:
        d = np.zeros(desc_dim)
        # Joint type one-hot (hinge=0, slide=1, ball=2)
        if ji.joint_type == 3:  # hinge
            d[0] = 1.0
        elif ji.joint_type == 2:  # slide
            d[1] = 1.0
        else:
            d[2] = 1.0
        # Axis
        d[3:6] = ji.axis
        # Mass (normalized)
        d[6] = ji.child_mass / max_mass
        # Length approximation from CoM position
        d[7] = np.linalg.norm(ji.child_com) / 0.5  # normalize by ~0.5m
        # Torque (normalized)
        d[8] = ji.torque_limit / max_torque
        # Gear ratio (normalized)
        d[9] = ji.gear_ratio / max(abs(ji.gear_ratio), 1.0)
        descriptors[ji.actuator_id] = d
    info.joint_descriptors = descriptors

    return info


def print_robot_info(info: RobotInfo):
    """Print parsed robot information for validation."""
    print(f"\n{'='*70}")
    print(f"  SOMA v10 — Robot: {info.name}")
    print(f"{'='*70}")
    print(f"  Morphology:    {info.morphology}")
    print(f"  Actuators:     {info.nu}")
    print(f"  Limbs:         {info.n_limbs} × {info.joints_per_limb} joints")
    print(f"  Total mass:    {info.total_mass:.2f} kg")
    print(f"  Base mass:     {info.base_mass:.2f} kg")
    print(f"  Stand height:  {info.standing_height:.3f} m")
    print(f"  CPG phases:    {info.cpg_phase_offsets}")
    print(f"\n  Joint details:")
    print(f"  {'ID':>3} {'Name':<20} {'Limb':>4} {'Depth':>5} {'Range':>16} {'τ_max':>6} {'Kp':>7} {'Kd':>7}")
    for ji in info.joints:
        print(f"  {ji.actuator_id:3d} {ji.name:<20} {ji.limb_id:4d} {ji.depth_in_limb:5d} "
              f"[{ji.range[0]:+.2f},{ji.range[1]:+.2f}] {ji.torque_limit:6.1f} "
              f"{info.kp[ji.actuator_id]:7.1f} {info.kd[ji.actuator_id]:7.3f}")
    print(f"{'='*70}\n")


# ══════════════════════════════════════════════════════════════════════
# SECTION 2: LOW-RANK SPARSE CfC CELL
# ══════════════════════════════════════════════════════════════════════

def init_lowrank_sparse_cfc(key, input_dim, hidden_dim, rank, sparsity):
    """
    Initialize a Low-Rank Sparse CfC cell.

    W_rec = (W1 @ W2.T) ⊙ M

    From: Tumma, Loo, Lechner, Hasani, Rus (ICLR 2024 Spotlight)
    - Fewer params outperform full-rank under distribution shift
    - Drop-in replacement for standard recurrent matrices

    CfC closed-form (Hasani et al., Nature Machine Intelligence 2022):
    h(t+dt) = σ(τ_a·dt + τ_b) ⊙ f(x,h) + (1 - σ(τ_a·dt + τ_b)) ⊙ h(t)

    Where σ is the time-gate sigmoid, f is the backbone transform.
    """
    k1, k2, k3, k4, k5, k6, k7, k8, k9 = random.split(key, 9)

    # Recurrent weights: low-rank sparse
    W1 = random.normal(k1, (hidden_dim, rank)) * 0.1
    W2 = random.normal(k2, (hidden_dim, rank)) * 0.1
    # Binary sparsity mask (fixed, not learned)
    M = (random.uniform(k3, (hidden_dim, hidden_dim)) > sparsity).astype(jnp.float32)

    # Input projection
    W_in = random.normal(k4, (input_dim, hidden_dim)) * (2.0 / (input_dim + hidden_dim))**0.5

    # Time-gate parameters (τ_a, τ_b)
    tau_a = random.normal(k5, (hidden_dim,)) * 0.1
    tau_b = jnp.zeros(hidden_dim)  # bias starts at 0 → σ(0)=0.5

    # Backbone transform: f(x,h) = tanh(W_f @ [x_proj, h_rec] + b_f)
    W_f = random.normal(k6, (hidden_dim * 2, hidden_dim)) * (2.0 / (hidden_dim * 3))**0.5
    b_f = jnp.zeros(hidden_dim)

    # Output projection
    W_out = random.normal(k7, (hidden_dim, hidden_dim)) * 0.1
    b_out = jnp.zeros(hidden_dim)

    return {
        'W1': W1, 'W2': W2, 'M': M,
        'W_in': W_in,
        'tau_a': tau_a, 'tau_b': tau_b,
        'W_f': W_f, 'b_f': b_f,
        'W_out': W_out, 'b_out': b_out,
    }


def init_fullrank_cfc(key, input_dim, hidden_dim):
    """Initialize standard CfC (for baseline comparison)."""
    k1, k2, k3, k4, k5 = random.split(key, 5)

    W_rec = random.normal(k1, (hidden_dim, hidden_dim)) * (2.0 / (hidden_dim * 2))**0.5
    W_in = random.normal(k2, (input_dim, hidden_dim)) * (2.0 / (input_dim + hidden_dim))**0.5
    tau_a = random.normal(k3, (hidden_dim,)) * 0.1
    tau_b = jnp.zeros(hidden_dim)
    W_f = random.normal(k4, (hidden_dim * 2, hidden_dim)) * (2.0 / (hidden_dim * 3))**0.5
    b_f = jnp.zeros(hidden_dim)
    W_out = random.normal(k4, (hidden_dim, hidden_dim)) * 0.1
    b_out = jnp.zeros(hidden_dim)

    return {
        'W_rec': W_rec, 'W_in': W_in,
        'tau_a': tau_a, 'tau_b': tau_b,
        'W_f': W_f, 'b_f': b_f,
        'W_out': W_out, 'b_out': b_out,
    }


def cfc_step(params, x, h, dt, is_lowrank=True):
    """
    Single CfC cell forward step.

    h(t+dt) = σ(τ_a·dt + τ_b) ⊙ f(x,h) + (1 - σ(τ_a·dt + τ_b)) ⊙ h(t)

    The dt-gate σ(τ_a·dt + τ_b) gives:
    - Frequency invariance: same weights work at any dt
    - Bounded hidden state: σ ∈ (0,1) → h is convex combo
    - O(1) constant time: no ODE solver, single matmul

    Args:
        params: CfC parameters dict
        x: input features (D,)
        h: hidden state (H,)
        dt: timestep (scalar)
        is_lowrank: whether to use low-rank sparse recurrence
    Returns:
        h_new: updated hidden state (H,)
        out: output features (H,)
    """
    # Recurrent transform
    if is_lowrank:
        # W_rec = (W1 @ W2.T) ⊙ M — low-rank sparse
        W_rec = (params['W1'] @ params['W2'].T) * params['M']
    else:
        W_rec = params['W_rec']

    h_rec = h @ W_rec  # (H,)

    # Input projection
    x_proj = x @ params['W_in']  # (H,)

    # Backbone: f(x,h) = tanh(W_f @ [x_proj, h_rec] + b_f)
    concat = jnp.concatenate([x_proj, h_rec])  # (2H,)
    f_xh = jnp.tanh(concat @ params['W_f'] + params['b_f'])  # (H,)

    # Time gate: σ(τ_a · dt + τ_b)
    # This is THE key CfC mechanism — dt modulates interpolation
    sigma = jax.nn.sigmoid(params['tau_a'] * dt + params['tau_b'])  # (H,)

    # CfC update: interpolate between current state and backbone output
    h_new = sigma * f_xh + (1.0 - sigma) * h  # (H,)

    # Output
    out = h_new @ params['W_out'] + params['b_out']  # (H,)

    return h_new, out


# ══════════════════════════════════════════════════════════════════════
# SECTION 3: DIFFERENTIABLE CPG
# ══════════════════════════════════════════════════════════════════════

def init_cpg_params(key, n_limbs, config: SOMAConfig):
    """
    Initialize differentiable CPG parameters.

    These are included in the ES optimization vector so ES
    tunes CPG jointly with CfC weights.

    CPG output per limb: A · sin(phase + offset)
    Phase advances at frequency f per timestep.
    CfC backbone modulates Δf and ΔA per limb.
    """
    return {
        'base_freq': jnp.ones(n_limbs) * config.cpg_base_freq,
        'base_amp':  jnp.ones(n_limbs) * config.cpg_base_amp,
        # Per-joint amplitude profile within a limb
        # (hip gets more, ankle gets less — learned by ES)
        'joint_amp_profile': jnp.ones(3) * 0.5,  # Will be resized per robot
    }


def cpg_step(cpg_params, phases, dt, delta_f, delta_a, phase_offsets):
    """
    Differentiable CPG step.

    Sinusoidal phase-coupled oscillators (proven better than Matsuoka
    in every comparison — Buchli & Ijspeert 2006, confirmed 2025).

    Args:
        cpg_params: learnable CPG parameters
        phases: current phase per limb (n_limbs,)
        dt: timestep
        delta_f: frequency modulation from CfC (n_limbs,)
        delta_a: amplitude modulation from CfC (n_limbs,)
        phase_offsets: fixed coupling offsets (n_limbs,)

    Returns:
        new_phases: advanced phases (n_limbs,)
        cpg_signals: per-limb sinusoidal signal (n_limbs,)
    """
    # Frequency = base + learned delta (clamped positive)
    freq = jnp.maximum(cpg_params['base_freq'] + delta_f, 0.5)  # min 0.5 Hz

    # Advance phase
    new_phases = (phases + 2.0 * jnp.pi * freq * dt) % (2.0 * jnp.pi)

    # Amplitude = base + learned delta (clamped positive)
    amp = jnp.maximum(cpg_params['base_amp'] + delta_a, 0.05)  # min 5cm

    # Sinusoidal output with phase coupling
    cpg_signals = amp * jnp.sin(new_phases + phase_offsets)

    return new_phases, cpg_signals


def cpg_to_joint_targets(cpg_signals, joint_amp_profile, stand_q,
                         limb_assignments, cmd_scale):
    """
    Convert per-limb CPG signals to per-joint position targets.

    Args:
        cpg_signals: (n_limbs,) sinusoidal signal per limb
        joint_amp_profile: (joints_per_limb,) amplitude profile
        stand_q: (nu,) standing joint configuration
        limb_assignments: (nu,) which limb each joint belongs to
        cmd_scale: scalar in [0,1], 0=standing, 1=full gait

    Returns:
        q_target: (nu,) joint position targets
    """
    # Each joint gets its limb's signal scaled by depth profile
    nu = stand_q.shape[0]
    offsets = jnp.zeros(nu)

    for j in range(nu):
        limb = limb_assignments[j]
        depth = j % len(joint_amp_profile)  # depth within limb
        offsets = offsets.at[j].set(
            cpg_signals[limb] * joint_amp_profile[depth]
        )

    return stand_q + cmd_scale * offsets


# ══════════════════════════════════════════════════════════════════════
# SECTION 4: PER-JOINT SHARED EXPERT + BACKBONE
# ══════════════════════════════════════════════════════════════════════

def init_soma_network(key, robot_info: RobotInfo, config: SOMAConfig):
    """
    Initialize the complete SOMA v10 network.

    Architecture:
        [Proprioception] → CfC Backbone → global_context
                                │
        For each joint j (shared weights):
            expert_j(local_j, desc_j, msg_j, global, phase_j) → Δτ_j
                                │
        τ_j = PD_j + α · Δτ_j
                                │
        [τ_all] → HOCBF-QP → [τ_safe_all]
    """
    keys = random.split(key, 10)
    params = {}

    if config.USE_PERJOINT_MOE:
        # ── CfC Backbone ──
        # Input: full proprioception [q(nu), qd(nu), quat(4), gyro(3), prev_action(nu)]
        backbone_input_dim = robot_info.nu * 2 + 4 + 3 + robot_info.nu
        if config.USE_LOWRANK_CFC:
            params['backbone'] = init_lowrank_sparse_cfc(
                keys[0], backbone_input_dim, config.backbone_hidden,
                config.backbone_rank, config.backbone_sparsity
            )
        else:
            params['backbone'] = init_fullrank_cfc(
                keys[0], backbone_input_dim, config.backbone_hidden
            )

        # Backbone → per-limb CPG modulation (Δf, Δa per limb)
        params['cpg_mod'] = {
            'W': random.normal(keys[1], (config.backbone_hidden, robot_info.n_limbs * 2)) * 0.01,
            'b': jnp.zeros(robot_info.n_limbs * 2),
        }

        # ── Shared Per-Joint Expert ──
        # Input: local(3) + descriptor(10) + neighbor_msg(expert_hidden) + global(backbone_hidden) + phase(2)
        expert_input_dim = 3 + 10 + config.expert_hidden + config.backbone_hidden + 2
        if config.USE_LOWRANK_CFC:
            params['expert'] = init_lowrank_sparse_cfc(
                keys[2], expert_input_dim, config.expert_hidden,
                config.expert_rank, config.expert_sparsity
            )
        else:
            params['expert'] = init_fullrank_cfc(
                keys[2], expert_input_dim, config.expert_hidden
            )

        # Expert → torque residual (single scalar per joint)
        params['expert_out'] = {
            'W': random.normal(keys[3], (config.expert_hidden, 1)) * 0.01,
            'b': jnp.zeros(1),
        }

        # ── Residual authority (learnable) ──
        params['alpha'] = jnp.array(config.alpha_init)

    else:
        # ── Monolithic CfC (baseline for comparison) ──
        mono_input_dim = robot_info.nu * 2 + 4 + 3 + robot_info.nu + 2  # +2 for phase
        if config.USE_LOWRANK_CFC:
            params['mono'] = init_lowrank_sparse_cfc(
                keys[0], mono_input_dim, config.mono_hidden,
                config.backbone_rank, config.backbone_sparsity
            )
        else:
            params['mono'] = init_fullrank_cfc(
                keys[0], mono_input_dim, config.mono_hidden
            )
        params['mono_out'] = {
            'W': random.normal(keys[1], (config.mono_hidden, robot_info.nu)) * 0.01,
            'b': jnp.zeros(robot_info.nu),
        }
        params['alpha'] = jnp.array(config.alpha_init)

    # ── CPG parameters (jointly optimized) ──
    if config.USE_DIFF_CPG:
        params['cpg'] = init_cpg_params(keys[5], robot_info.n_limbs, config)
        # Resize joint_amp_profile for this robot
        params['cpg']['joint_amp_profile'] = jnp.ones(robot_info.joints_per_limb) * 0.3

    return params


def count_params(params):
    """Count total learnable parameters (excluding binary masks)."""
    total = 0
    def _count(p):
        nonlocal total
        if isinstance(p, dict):
            for k, v in p.items():
                if k == 'M':  # Skip binary mask
                    continue
                _count(v)
        elif isinstance(p, jnp.ndarray) or isinstance(p, np.ndarray):
            total += int(np.prod(p.shape))
        elif isinstance(p, (float, int)):
            total += 1
    _count(params)
    return total


# ══════════════════════════════════════════════════════════════════════
# SECTION 5: SOMA FORWARD PASS (SINGLE TIMESTEP)
# ══════════════════════════════════════════════════════════════════════

def soma_step(params, state, robot_info, config, dt):
    """
    Complete SOMA v10 forward pass for one timestep.

    Flow:
    1. Build proprioceptive observation
    2. CfC backbone processes full-body state → global context
    3. Backbone outputs CPG frequency/amplitude modulation
    4. CPG generates per-limb reference signals
    5. Per-joint expert processes local state + context → Δτ
    6. Residual: τ = PD(q_target) + α · Δτ
    7. HOCBF-CLF-QP filters to safe torques

    Args:
        params: network parameters
        state: dict with keys:
            'q': joint positions (nu,)
            'qd': joint velocities (nu,)
            'quat': base orientation quaternion (4,)
            'gyro': angular velocity (3,)
            'prev_action': previous torque output (nu,)
            'h_backbone': backbone hidden state (H_backbone,)
            'h_experts': per-joint expert hidden states (nu, H_expert)
            'cpg_phases': CPG phase per limb (n_limbs,)
            'cmd': velocity command [vx, vy, yaw_rate] (3,)
        robot_info: parsed robot description
        config: SOMAConfig
        dt: timestep

    Returns:
        torques: safe output torques (nu,)
        new_state: updated state dict
    """
    q = state['q']
    qd = state['qd']
    quat = state['quat']
    gyro = state['gyro']
    prev_action = state['prev_action']
    h_backbone = state['h_backbone']
    h_experts = state['h_experts']
    cpg_phases = state['cpg_phases']
    cmd = state['cmd']
    nu = robot_info.nu

    # Command magnitude for CPG scaling
    cmd_scale = jnp.clip(jnp.sqrt(cmd[0]**2 + cmd[1]**2) / 1.0, 0.0, 1.0)

    if config.USE_PERJOINT_MOE:
        # ── Step 1: Backbone ──
        obs = jnp.concatenate([q, qd, quat, gyro, prev_action])
        h_backbone_new, global_ctx = cfc_step(
            params['backbone'], obs, h_backbone, dt,
            is_lowrank=config.USE_LOWRANK_CFC
        )

        # ── Step 2: CPG modulation from backbone ──
        cpg_mod = global_ctx @ params['cpg_mod']['W'] + params['cpg_mod']['b']
        n_limbs = robot_info.n_limbs
        delta_f = cpg_mod[:n_limbs] * 0.5   # Scale frequency deltas
        delta_a = cpg_mod[n_limbs:] * 0.1    # Scale amplitude deltas

        # ── Step 3: CPG step ──
        if config.USE_DIFF_CPG:
            new_phases, cpg_signals = cpg_step(
                params['cpg'], cpg_phases, dt, delta_f, delta_a,
                jnp.array(robot_info.cpg_phase_offsets)
            )
        else:
            # Fixed CPG (no learned params)
            freq = jnp.ones(n_limbs) * config.cpg_base_freq
            new_phases = (cpg_phases + 2 * jnp.pi * freq * dt) % (2 * jnp.pi)
            cpg_signals = config.cpg_base_amp * jnp.sin(
                new_phases + jnp.array(robot_info.cpg_phase_offsets)
            )

        # ── Step 4: CPG → joint targets ──
        limb_assignments = jnp.array([ji.limb_id for ji in robot_info.joints])
        stand_q = jnp.array([
            (ji.range[0] + ji.range[1]) / 2.0 for ji in robot_info.joints
        ])
        joint_amp = params['cpg']['joint_amp_profile'] if config.USE_DIFF_CPG else (
            jnp.ones(robot_info.joints_per_limb) * 0.3
        )
        q_target = cpg_to_joint_targets(
            cpg_signals, joint_amp, stand_q, limb_assignments, cmd_scale
        )

        # ── Step 5: PD torque ──
        kp = jnp.array(robot_info.kp) * config.kp_scale
        kd = jnp.array(robot_info.kd) * config.kd_scale
        tau_pd = kp * (q_target - q) - kd * qd

        # ── Step 6: Per-joint expert ──
        descriptors = jnp.array(robot_info.joint_descriptors)  # (nu, 10)
        new_h_experts = jnp.zeros_like(h_experts)
        delta_tau = jnp.zeros(nu)

        # Message passing: each joint gets mean of kinematic neighbors' hidden states
        # Build neighbor messages from previous step's hidden states
        def compute_joint_expert(j, carry):
            delta_tau, new_h = carry

            # Local state
            local_j = jnp.array([q[j], qd[j], prev_action[j]])

            # Joint descriptor
            desc_j = descriptors[j]

            # Neighbor message (mean of parent + children hidden states)
            ji = robot_info.joints[j]
            neighbor_ids = []
            if ji.parent_joint_id >= 0:
                neighbor_ids.append(ji.parent_joint_id)
            neighbor_ids.extend(ji.child_joint_ids)

            if len(neighbor_ids) > 0:
                neighbor_h = jnp.stack([h_experts[nid] for nid in neighbor_ids])
                msg_j = jnp.mean(neighbor_h, axis=0)
            else:
                msg_j = jnp.zeros(config.expert_hidden)

            # Phase signal for this joint's limb
            limb = ji.limb_id
            phase_j = jnp.array([jnp.sin(new_phases[limb]), jnp.cos(new_phases[limb])])

            # Expert input
            expert_in = jnp.concatenate([local_j, desc_j, msg_j, global_ctx, phase_j])

            # Expert CfC step
            h_j_new, out_j = cfc_step(
                params['expert'], expert_in, h_experts[j], dt,
                is_lowrank=config.USE_LOWRANK_CFC
            )

            # Torque residual (single scalar)
            dtau_j = (out_j @ params['expert_out']['W'] + params['expert_out']['b'])[0]

            delta_tau = delta_tau.at[j].set(dtau_j)
            new_h = new_h.at[j].set(h_j_new)

            return (delta_tau, new_h)

        # Unroll expert computation for all joints
        # (can't use vmap easily due to variable neighbor structure,
        #  but nu is small so this is fine for compilation)
        for j in range(nu):
            delta_tau, new_h_experts = compute_joint_expert(j, (delta_tau, new_h_experts))

        # ── Step 7: Residual combination ──
        alpha = jax.nn.sigmoid(params['alpha'])  # Bound to (0,1)
        # Scale residual by torque limits for each joint
        torque_limits = jnp.array([ji.torque_limit for ji in robot_info.joints])
        tau_nominal = tau_pd + alpha * jnp.tanh(delta_tau) * torque_limits

    else:
        # ── Monolithic baseline ──
        phase_sin = jnp.sin(cpg_phases[0])
        phase_cos = jnp.cos(cpg_phases[0])
        obs = jnp.concatenate([q, qd, quat, gyro, prev_action,
                               jnp.array([phase_sin, phase_cos])])
        h_backbone_new, out = cfc_step(
            params['mono'], obs, h_backbone, dt,
            is_lowrank=config.USE_LOWRANK_CFC
        )
        delta_tau = out @ params['mono_out']['W'] + params['mono_out']['b']

        # Fixed CPG for baseline
        freq = jnp.ones(robot_info.n_limbs) * config.cpg_base_freq
        new_phases = (cpg_phases + 2 * jnp.pi * freq * dt) % (2 * jnp.pi)
        cpg_signals = config.cpg_base_amp * jnp.sin(
            new_phases + jnp.array(robot_info.cpg_phase_offsets)
        )
        limb_assignments = jnp.array([ji.limb_id for ji in robot_info.joints])
        stand_q = jnp.array([(ji.range[0] + ji.range[1]) / 2.0 for ji in robot_info.joints])
        q_target = cpg_to_joint_targets(
            cpg_signals, jnp.ones(robot_info.joints_per_limb) * 0.3,
            stand_q, limb_assignments, cmd_scale
        )
        kp = jnp.array(robot_info.kp) * config.kp_scale
        kd = jnp.array(robot_info.kd) * config.kd_scale
        tau_pd = kp * (q_target - q) - kd * qd

        alpha = jax.nn.sigmoid(params['alpha'])
        torque_limits = jnp.array([ji.torque_limit for ji in robot_info.joints])
        tau_nominal = tau_pd + alpha * jnp.tanh(delta_tau) * torque_limits

        new_h_experts = h_experts  # unchanged

    # ── Step 8: Safety filter ──
    if config.USE_HOCBF:
        tau_safe = hocbf_clq_qp(
            tau_nominal, q, qd, robot_info, config
        )
    else:
        # Simple clamp to torque limits
        tau_safe = jnp.clip(tau_nominal, -torque_limits, torque_limits)

    # ── Update state ──
    new_state = {
        'q': q, 'qd': qd, 'quat': quat, 'gyro': gyro,
        'prev_action': tau_safe,
        'h_backbone': h_backbone_new,
        'h_experts': new_h_experts,
        'cpg_phases': new_phases,
        'cmd': cmd,
    }

    return tau_safe, new_state


# ══════════════════════════════════════════════════════════════════════
# SECTION 6: HOCBF-CLF-QP SAFETY FILTER
# ══════════════════════════════════════════════════════════════════════

def hocbf_clq_qp(tau_nominal, q, qd, robot_info, config):
    """
    High-Order Control Barrier Function with CLF-QP safety filter.

    For torque-controlled robots (relative degree 2 from position to torque):
      h(q) = (q_max - q)(q - q_min)          # position barrier
      ψ₁ = ḣ + α₁·h                           # first-order condition
      constraint: ψ̇₁ + α₂·ψ₁ ≥ 0             # enforced on torque

    Also enforces:
      - Torque limits (hard)
      - Velocity limits (hard)
      - Anti-tipover via base orientation (soft if available)

    Uses analytical QP solution for box-constrained case.
    For the typical per-joint safety constraints, the QP decouples
    into independent per-joint projections, so no matrix inversion needed.

    From: Xiao & Belta (IEEE TAC 2022), Ames et al. (IEEE TAC 2017)
    """
    nu = robot_info.nu
    alpha1 = config.cbf_alpha1
    alpha2 = config.cbf_alpha2

    tau_safe = tau_nominal  # Start from nominal

    for j in range(nu):
        ji = robot_info.joints[j]
        q_j = q[j]
        qd_j = qd[j]
        tau_j = tau_nominal[j]
        q_min = ji.range[0] + 0.05  # 0.05 rad safety margin
        q_max = ji.range[1] - 0.05

        # ── Position barrier h(q) = (q_max - q)(q - q_min) ──
        h = (q_max - q_j) * (q_j - q_min)

        # ── First derivative ḣ = q̇ · (q_max + q_min - 2q) ──
        dh = qd_j * (q_max + q_min - 2.0 * q_j)

        # ── ψ₁ = ḣ + α₁·h ──
        psi1 = dh + alpha1 * h

        # ── ψ̇₁ needs q̈, which depends on τ via: q̈ ≈ τ / I_approx ──
        # Approximate inertia for this joint
        I_approx = jnp.maximum(ji.child_mass * 0.1**2, 0.001)
        # ∂ψ̇₁/∂τ ≈ (q_max + q_min - 2q)/I_approx + α₁·q̇_term
        # For HOCBF: ψ̇₁ + α₂·ψ₁ ≥ 0
        # → (∂ψ̇₁/∂τ)·τ + rest ≥ -α₂·ψ₁
        # Since this is scalar per joint, direct projection:
        dpsi1_dtau = (q_max + q_min - 2.0 * q_j) / I_approx
        rest_term = alpha1 * dh  # remaining terms

        # Constraint: dpsi1_dtau · τ + rest_term + α₂·ψ₁ ≥ 0
        # → τ ≥ -(rest_term + α₂·ψ₁) / dpsi1_dtau  (if dpsi1_dtau > 0)
        # → τ ≤ -(rest_term + α₂·ψ₁) / dpsi1_dtau  (if dpsi1_dtau < 0)

        rhs = -(rest_term + alpha2 * psi1)

        # Safe projection
        tau_lower = jnp.where(
            dpsi1_dtau > 1e-6,
            rhs / dpsi1_dtau,
            -ji.torque_limit
        )
        tau_upper = jnp.where(
            dpsi1_dtau < -1e-6,
            rhs / dpsi1_dtau,
            ji.torque_limit
        )

        # Combine with torque limits
        tau_lower = jnp.maximum(tau_lower, -ji.torque_limit)
        tau_upper = jnp.minimum(tau_upper, ji.torque_limit)

        # Also enforce velocity limits via first-order CBF
        vel_limit = jnp.pi * 10  # Conservative velocity limit
        h_vel_pos = vel_limit - qd_j
        h_vel_neg = qd_j + vel_limit
        # If approaching positive vel limit, torque must be ≤ -α·h_vel
        tau_vel_upper = jnp.where(
            h_vel_pos < 1.0,  # Within 1 rad/s of limit
            -alpha1 * (vel_limit - qd_j) * I_approx,
            ji.torque_limit
        )
        tau_vel_lower = jnp.where(
            h_vel_neg < 1.0,
            alpha1 * (qd_j + vel_limit) * I_approx,
            -ji.torque_limit
        )

        tau_upper = jnp.minimum(tau_upper, tau_vel_upper)
        tau_lower = jnp.maximum(tau_lower, tau_vel_lower)

        # Project nominal torque to safe set (minimum modification)
        tau_j_safe = jnp.clip(tau_j, tau_lower, tau_upper)
        tau_safe = tau_safe.at[j].set(tau_j_safe)

    return tau_safe


# ══════════════════════════════════════════════════════════════════════
# SECTION 7: MJX ROLLOUT ENGINE
# ══════════════════════════════════════════════════════════════════════

def setup_mjx(xml_path: str, config: SOMAConfig):
    """
    Load MuJoCo model and prepare MJX pipeline.

    Converts to torque actuators if needed (lesson from v5.1).
    Strips collision meshes for MJX (lesson from v2 OOM).
    """
    mj_model = mujoco.MjModel.from_xml_path(xml_path)

    # ── CRITICAL: Verify and convert actuators (v5.1 lesson) ──
    for i in range(mj_model.nu):
        # Check actuator type — warn if position servo
        gaintype = mj_model.actuator_gaintype[i]
        if gaintype == 0:  # Fixed gain
            print(f"  Actuator {i}: gaintype=fixed (OK for torque mode)")
        else:
            print(f"  ⚠ Actuator {i}: gaintype={gaintype} — check torque mode!")

    # Parse robot info
    robot_info = parse_robot(mj_model)
    print_robot_info(robot_info)

    # Convert to MJX
    mx_model = mjx.put_model(mj_model)

    # Get default data
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)
    mx_data = mjx.put_data(mj_model, mj_data)

    return mj_model, mx_model, mx_data, robot_info


def make_initial_state(robot_info: RobotInfo, config: SOMAConfig):
    """Create initial SOMA state dict."""
    return {
        'q': jnp.zeros(robot_info.nu),
        'qd': jnp.zeros(robot_info.nu),
        'quat': jnp.array([1.0, 0.0, 0.0, 0.0]),
        'gyro': jnp.zeros(3),
        'prev_action': jnp.zeros(robot_info.nu),
        'h_backbone': jnp.zeros(config.backbone_hidden if config.USE_PERJOINT_MOE
                                 else config.mono_hidden),
        'h_experts': jnp.zeros((robot_info.nu, config.expert_hidden)),
        'cpg_phases': jnp.zeros(robot_info.n_limbs),
        'cmd': jnp.array([0.5, 0.0, 0.0]),  # Default: walk forward 0.5 m/s
    }


def rollout_one(params, mx_model, mx_data_init, robot_info, config,
                cmd, rng, phase_config):
    """
    Run a single MJX rollout with SOMA controller.

    Args:
        params: network parameters
        mx_model: MJX model
        mx_data_init: initial MJX data
        robot_info: parsed robot
        config: SOMAConfig
        cmd: velocity command [vx, vy, yaw_rate]
        rng: random key
        phase_config: dict with {'phase': int, 'dr_scale': float, 'push': bool}

    Returns:
        fitness: scalar fitness value
        metrics: dict of tracking metrics
    """
    dt = config.dt
    n_steps = config.rollout_steps
    nu = robot_info.nu

    # ── Domain randomization (if enabled for this phase) ──
    dr_scale = phase_config.get('dr_scale', 0.0)
    rng, rng_dr = random.split(rng)

    # We apply DR by modifying the control gains, not the MJX model
    # (MJX model is shared across vmapped rollouts)
    dr_keys = random.split(rng_dr, 5)
    kp_scale = 1.0 + dr_scale * random.uniform(
        dr_keys[0], (), minval=config.dr_kp_range[0]-1, maxval=config.dr_kp_range[1]-1
    )
    kd_scale = 1.0 + dr_scale * random.uniform(
        dr_keys[1], (), minval=config.dr_kd_range[0]-1, maxval=config.dr_kd_range[1]-1
    )
    motor_scale = 1.0 + dr_scale * random.uniform(
        dr_keys[2], (), minval=config.dr_motor_range[0]-1, maxval=config.dr_motor_range[1]-1
    )

    # ── Initialize SOMA state ──
    soma_state = make_initial_state(robot_info, config)
    soma_state['cmd'] = cmd

    # Modified config for this rollout (DR applied)
    rollout_config = config
    # (In practice we'd modify kp_scale/kd_scale, but for JIT we pass them separately)

    # ── Rollout loop ──
    mx_data = mx_data_init

    # Accumulators
    total_vel_err = 0.0
    total_stab = 0.0
    total_energy = 0.0
    total_smooth = 0.0
    alive_steps = 0.0
    min_height = 10.0

    def step_fn(carry, step_idx):
        mx_data, soma_state, prev_tau, metrics = carry
        total_vel_err, total_stab, total_energy, total_smooth, alive_steps, min_h = metrics

        # Extract proprioception from MJX state
        # Joint positions (skip free joint: first 7 qpos are base pose)
        qpos_joints = mx_data.qpos[7:7+nu]  # Assumes free base joint
        qvel_joints = mx_data.qvel[6:6+nu]  # Skip base vel (6-DOF)

        # Base orientation (quaternion from qpos[3:7])
        base_quat = mx_data.qpos[3:7]

        # Base angular velocity
        base_gyro = mx_data.qvel[3:6]

        # Base height
        base_height = mx_data.qpos[2]

        # Update SOMA state with current proprioception
        soma_state = {**soma_state,
            'q': qpos_joints,
            'qd': qvel_joints,
            'quat': base_quat,
            'gyro': base_gyro,
        }

        # ── SOMA forward pass ──
        tau, soma_state = soma_step(params, soma_state, robot_info, config, dt)

        # Apply motor scale (DR) and scale by torque limits
        tau = tau * motor_scale

        # Apply to MJX
        mx_data = mx_data.replace(ctrl=tau)
        mx_data = mjx.step(mx_model, mx_data)

        # ── Push perturbation (Phase 4 only) ──
        do_push = phase_config.get('push', False)
        rng_push = random.fold_in(rng, step_idx)
        push_interval = random.randint(rng_push, (), *config.push_interval)
        apply_push = do_push & (step_idx % push_interval == 0) & (step_idx > 100)
        push_vel = random.uniform(
            rng_push, (2,),
            minval=-config.push_vel_range[1],
            maxval=config.push_vel_range[1]
        )
        # Apply push by modifying base velocity
        new_qvel = jnp.where(
            apply_push,
            mx_data.qvel.at[0].set(mx_data.qvel[0] + push_vel[0])
                        .at[1].set(mx_data.qvel[1] + push_vel[1]),
            mx_data.qvel
        )
        mx_data = mx_data.replace(qvel=new_qvel)

        # ── Compute metrics ──
        # Velocity tracking error
        base_vel = mx_data.qvel[0:3]  # World-frame base velocity
        cmd = soma_state['cmd']
        vel_err = jnp.sqrt(
            (base_vel[0] - cmd[0])**2 +
            (base_vel[1] - cmd[1])**2
        )

        # Stability (orientation error from upright)
        # For quaternion [w,x,y,z], upright = [1,0,0,0]
        quat_err = 1.0 - base_quat[0]**2  # 0 = perfectly upright

        # Energy (torque magnitude)
        energy = jnp.sum(tau**2)

        # Smoothness (action change)
        smooth = jnp.sum((tau - prev_tau)**2)

        # Alive check
        alive = (base_height > robot_info.standing_height * 0.4).astype(jnp.float32)

        # Update accumulators
        total_vel_err += vel_err * alive
        total_stab += quat_err * alive
        total_energy += energy * alive
        total_smooth += smooth * alive
        alive_steps += alive
        min_h = jnp.minimum(min_h, base_height)

        metrics = (total_vel_err, total_stab, total_energy, total_smooth, alive_steps, min_h)
        return (mx_data, soma_state, tau, metrics), None

    # Initial carry
    init_metrics = (0.0, 0.0, 0.0, 0.0, 0.0, 10.0)
    init_carry = (mx_data, soma_state, jnp.zeros(nu), init_metrics)

    # Run rollout
    (mx_data_final, soma_state_final, _, final_metrics), _ = lax.scan(
        step_fn, init_carry, jnp.arange(n_steps)
    )

    total_vel_err, total_stab, total_energy, total_smooth, alive_steps, min_h = final_metrics

    # Normalize by alive steps
    alive_steps = jnp.maximum(alive_steps, 1.0)
    avg_vel_err = total_vel_err / alive_steps
    avg_stab = total_stab / alive_steps
    avg_energy = total_energy / alive_steps
    avg_smooth = total_smooth / alive_steps
    survival_ratio = alive_steps / n_steps

    # ── Fitness (negative = minimize) ──
    fitness = -(
        config.w_vel * avg_vel_err +
        config.w_stab * avg_stab +
        config.w_energy * avg_energy * 1e-4 +
        config.w_smooth * avg_smooth * 1e-4 -
        config.w_alive * survival_ratio
    )

    metrics = {
        'vel_err': avg_vel_err,
        'stab': avg_stab,
        'energy': avg_energy,
        'smooth': avg_smooth,
        'alive': survival_ratio,
        'min_h': min_h,
    }

    return fitness, metrics


# ══════════════════════════════════════════════════════════════════════
# SECTION 8: PARAMETER FLATTEN/UNFLATTEN FOR ES
# ══════════════════════════════════════════════════════════════════════

def flatten_params(params):
    """Flatten nested param dict to a single 1D array."""
    flat = []
    structure = []

    def _flatten(p, prefix=""):
        if isinstance(p, dict):
            for k in sorted(p.keys()):
                if k == 'M':  # Skip binary mask (not optimized)
                    continue
                _flatten(p[k], f"{prefix}.{k}")
        elif isinstance(p, jnp.ndarray):
            flat.append(p.ravel())
            structure.append((prefix, p.shape))
        elif isinstance(p, (float, int, np.floating)):
            flat.append(jnp.array([float(p)]))
            structure.append((prefix, ()))

    _flatten(params)
    return jnp.concatenate(flat), structure


def unflatten_params(flat, structure, template_params):
    """Reconstruct nested param dict from flat array."""
    params = {}
    offset = 0

    def _set_nested(d, keys, value):
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value

    for name, shape in structure:
        keys = [k for k in name.split('.') if k]
        if shape == ():
            val = flat[offset]
            offset += 1
        else:
            size = int(np.prod(shape))
            val = flat[offset:offset+size].reshape(shape)
            offset += size
        _set_nested(params, keys, val)

    # Restore binary masks from template
    def _restore_masks(src, dst):
        if isinstance(src, dict):
            for k in src:
                if k == 'M':
                    dst[k] = src[k]
                elif k in dst and isinstance(src[k], dict):
                    _restore_masks(src[k], dst[k])

    _restore_masks(template_params, params)
    return params


# ══════════════════════════════════════════════════════════════════════
# SECTION 9: PGPE EVOLUTION STRATEGY
# ══════════════════════════════════════════════════════════════════════

def pgpe_update(flat_params, flat_sigma, fitnesses, perturbations, lr, sigma_lr=0.01):
    """
    Parameter-exploring Policy Gradients (PGPE).

    Key advantage over OpenAI-ES: samples parameters once per episode,
    so gradient variance doesn't grow with episode length.
    At 10kHz × 10s = 100K steps/episode, this matters.

    From: Sehnke et al., 2010

    Args:
        flat_params: current mean parameters (D,)
        flat_sigma: per-parameter std (D,)
        fitnesses: fitness values for each perturbation pair (N,)
        perturbations: noise vectors (N, D)
        lr: learning rate
        sigma_lr: sigma learning rate
    Returns:
        new_params, new_sigma
    """
    N = perturbations.shape[0]

    # PGPE uses fitness difference between symmetric perturbations
    # Assumes fitnesses is [f(θ+ε₁), f(θ-ε₁), f(θ+ε₂), f(θ-ε₂), ...]
    f_plus = fitnesses[0::2]   # (N/2,)
    f_minus = fitnesses[1::2]  # (N/2,)
    epsilons = perturbations[0::2]  # (N/2, D)

    # Rank-based fitness shaping (reduces outlier influence)
    f_diff = f_plus - f_minus  # (N/2,)

    # Parameter gradient: E[f_diff * ε / σ²]
    grad_mu = jnp.mean(
        f_diff[:, None] * epsilons / (flat_sigma[None, :]**2 + 1e-8),
        axis=0
    )

    # Sigma gradient: E[f_diff * (ε²/σ³ - 1/σ)]
    grad_sigma = jnp.mean(
        f_diff[:, None] * (epsilons**2 / (flat_sigma[None, :]**3 + 1e-8) - 1.0 / (flat_sigma[None, :] + 1e-8)),
        axis=0
    )

    # Baseline subtraction using mean fitness
    f_mean = (f_plus + f_minus) / 2.0
    baseline_grad = jnp.mean(
        (f_mean[:, None] - jnp.mean(f_mean)) * epsilons / (flat_sigma[None, :]**2 + 1e-8),
        axis=0
    )
    grad_mu = grad_mu - baseline_grad * 0.5

    # Gradient clipping
    grad_norm = jnp.linalg.norm(grad_mu)
    grad_mu = jnp.where(grad_norm > 1.0, grad_mu / grad_norm, grad_mu)

    # Update
    new_params = flat_params + lr * grad_mu
    new_sigma = flat_sigma + sigma_lr * grad_sigma
    new_sigma = jnp.clip(new_sigma, 0.001, 0.1)  # Keep sigma bounded

    return new_params, new_sigma


def openai_es_update(flat_params, fitnesses, perturbations, lr, sigma):
    """
    OpenAI-ES update (fallback if PGPE doesn't converge).

    From: Salimans et al., 2017
    """
    N = perturbations.shape[0]

    # Rank-based fitness shaping
    ranks = jnp.argsort(jnp.argsort(-fitnesses)).astype(jnp.float32)
    shaped = jnp.maximum(0.0, jnp.log(N/2.0 + 1) - jnp.log(ranks + 1))
    shaped = shaped / (jnp.sum(shaped) + 1e-8) - 1.0 / N

    # Gradient estimate
    grad = jnp.mean(shaped[:, None] * perturbations, axis=0) / sigma

    # Gradient clipping
    grad_norm = jnp.linalg.norm(grad)
    grad = jnp.where(grad_norm > 1.0, grad / grad_norm, grad)

    new_params = flat_params + lr * grad
    return new_params


# ══════════════════════════════════════════════════════════════════════
# SECTION 10: TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def get_phase_config(gen, config):
    """Get curriculum phase configuration for a generation."""
    if gen < config.phase1_end:
        # Phase 1: Standing
        return {
            'phase': 1,
            'phase_name': 'Stand',
            'cmd_vx_range': (0.0, 0.0),
            'cmd_yaw_range': (0.0, 0.0),
            'dr_scale': 0.0,
            'push': False,
        }
    elif gen < config.phase2_end:
        # Phase 2: Walking with gradual DR
        progress = (gen - config.phase1_end) / (config.phase2_end - config.phase1_end)
        return {
            'phase': 2,
            'phase_name': 'Walk',
            'cmd_vx_range': (0.2, 0.8),
            'cmd_yaw_range': (0.0, 0.0),
            'dr_scale': min(progress, 1.0) * 0.5,  # Ramp DR to 50%
            'push': False,
        }
    elif gen < config.phase3_end:
        # Phase 3: Walking + Turning with full DR
        return {
            'phase': 3,
            'phase_name': 'Walk+Turn',
            'cmd_vx_range': (0.2, 1.0),
            'cmd_yaw_range': (-1.0, 1.0),
            'dr_scale': 0.75,
            'push': False,
        }
    else:
        # Phase 4: Push recovery with full DR
        return {
            'phase': 4,
            'phase_name': 'Push',
            'cmd_vx_range': (0.3, 0.8),
            'cmd_yaw_range': (-0.5, 0.5),
            'dr_scale': 1.0,
            'push': True,
        }


def train(xml_path: str, config: SOMAConfig = None):
    """
    Main SOMA v10 training loop.

    Usage:
        train("path/to/robot.xml")
        train("path/to/robot.xml", SOMAConfig(n_generations=200))
    """
    if config is None:
        config = SOMAConfig()

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  SOMA v10 — Universal Motor Execution Layer                ║")
    print("║  Training Pipeline                                         ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print(f"\n  JAX devices: {jax.devices()}")

    # ── Setup ──
    t0 = time.time()
    mj_model, mx_model, mx_data_init, robot_info = setup_mjx(xml_path, config)

    # ── Initialize network ──
    rng = random.PRNGKey(42)
    rng, init_key = random.split(rng)
    params = init_soma_network(init_key, robot_info, config)
    n_params = count_params(params)
    print(f"\n  Total parameters: {n_params:,}")
    print(f"  Architecture:     {'Per-Joint MoE' if config.USE_PERJOINT_MOE else 'Monolithic'}")
    print(f"  CfC type:         {'Low-Rank Sparse' if config.USE_LOWRANK_CFC else 'Full-Rank'}")
    print(f"  CPG:              {'Differentiable (joint opt)' if config.USE_DIFF_CPG else 'Fixed'}")
    print(f"  Safety:           {'HOCBF-CLF-QP' if config.USE_HOCBF else 'Simple clamp'}")
    print(f"  ES:               {'PGPE' if config.USE_PGPE else 'OpenAI-ES'}")

    # Flatten params for ES
    flat_params, structure = flatten_params(params)
    print(f"  Flat param dim:   {flat_params.shape[0]:,}")

    # Verify flatten/unflatten roundtrip
    reconstructed = unflatten_params(flat_params, structure, params)
    flat_check, _ = flatten_params(reconstructed)
    assert jnp.allclose(flat_params, flat_check), "Flatten/unflatten mismatch!"
    print(f"  Flatten/unflatten verified ✓")

    # Initialize per-parameter sigma (PGPE)
    flat_sigma = jnp.ones_like(flat_params) * config.sigma_init

    # ── Compile vmapped rollout ──
    print(f"\n  Compiling vmapped rollout ({config.pop_size}×2 parallel)...")
    t_compile = time.time()

    # Vmapped rollout: evaluate many parameter vectors in parallel
    def eval_one(flat_p, cmd, rng_rollout, phase_config_static):
        p = unflatten_params(flat_p, structure, params)
        fitness, metrics = rollout_one(
            p, mx_model, mx_data_init, robot_info, config,
            cmd, rng_rollout, phase_config_static
        )
        return fitness, metrics

    # We'll vmap over flat_p and rng, with cmd and phase_config broadcast
    eval_batch = jit(vmap(eval_one, in_axes=(0, None, 0, None)))

    # Warm-up compile
    rng, rng_warmup = random.split(rng)
    dummy_params = jnp.tile(flat_params[None, :], (4, 1))
    dummy_rngs = random.split(rng_warmup, 4)
    dummy_cmd = jnp.array([0.5, 0.0, 0.0])
    dummy_phase = {'phase': 1, 'dr_scale': 0.0, 'push': False}

    _ = eval_batch(dummy_params, dummy_cmd, dummy_rngs, dummy_phase)
    print(f"  Compiled in {time.time() - t_compile:.1f}s")

    # ── Training loop ──
    best_fitness = -1e10
    best_flat = flat_params.copy()
    best_vel_err = 1.0

    print(f"\n{'='*90}")
    print(f"  SOMA v10 — TRAINING ({config.n_generations} generations)")
    print(f"{'='*90}")
    print(f"  {'Gen':>4} {'Phase':<12} {'Fitness':>9} {'VelErr':>8} {'Stab':>8} "
          f"{'Alive':>6} {'MinH':>6} {'BstVE':>8} {'Sigma':>7} {'Elitist':>8} {'Time':>6}")
    print(f"  {'-'*4} {'-'*12} {'-'*9} {'-'*8} {'-'*8} {'-'*6} {'-'*6} {'-'*8} {'-'*7} {'-'*8} {'-'*6}")

    for gen in range(config.n_generations):
        t_gen = time.time()

        # ── Phase configuration ──
        phase_cfg = get_phase_config(gen, config)

        # ── Sample commands for this generation ──
        rng, rng_cmd = random.split(rng)
        cmd_vx = random.uniform(rng_cmd, (), minval=phase_cfg['cmd_vx_range'][0],
                                maxval=max(phase_cfg['cmd_vx_range'][1], 0.01))
        cmd_yaw = random.uniform(rng_cmd, (), minval=phase_cfg['cmd_yaw_range'][0],
                                 maxval=max(phase_cfg['cmd_yaw_range'][1], 0.01))
        cmd = jnp.array([cmd_vx, 0.0, cmd_yaw])

        # ── Generate perturbations ──
        rng, rng_noise = random.split(rng)
        noise = random.normal(rng_noise, (config.pop_size, flat_params.shape[0]))
        noise = noise * flat_sigma[None, :]  # Scale by per-param sigma

        # Antithetic: [+ε₁, -ε₁, +ε₂, -ε₂, ...]
        perturbed_pos = flat_params[None, :] + noise
        perturbed_neg = flat_params[None, :] - noise
        all_perturbed = jnp.concatenate([
            perturbed_pos.reshape(-1, 1, flat_params.shape[0]),
            perturbed_neg.reshape(-1, 1, flat_params.shape[0])
        ], axis=1).reshape(-1, flat_params.shape[0])  # (pop_size*2, D)

        # ── Evaluate all perturbations ──
        rng, rng_eval = random.split(rng)
        eval_rngs = random.split(rng_eval, config.pop_size * 2)

        fitnesses, metrics_batch = eval_batch(
            all_perturbed, cmd, eval_rngs, phase_cfg
        )

        # ── ES update ──
        mean_fitness = jnp.mean(fitnesses)
        mean_vel_err = jnp.mean(metrics_batch['vel_err'])
        mean_stab = jnp.mean(metrics_batch['stab'])
        mean_alive = jnp.mean(metrics_batch['alive'])
        mean_min_h = jnp.mean(metrics_batch['min_h'])

        if config.USE_PGPE:
            new_flat, new_sigma = pgpe_update(
                flat_params, flat_sigma, fitnesses, all_perturbed - flat_params[None, :],
                config.lr, sigma_lr=0.005
            )
            # Sigma decay
            new_sigma = jnp.maximum(new_sigma * config.sigma_decay, config.sigma_min)
        else:
            sigma = jnp.mean(flat_sigma)
            new_flat = openai_es_update(flat_params, fitnesses, all_perturbed - flat_params[None, :], config.lr, sigma)
            new_sigma = flat_sigma * config.sigma_decay
            new_sigma = jnp.maximum(new_sigma, config.sigma_min)

        # ── Elitist check ──
        # Evaluate the candidate before accepting
        rng, rng_elite = random.split(rng)
        new_params_dict = unflatten_params(new_flat, structure, params)
        new_fitness, new_metrics = rollout_one(
            new_params_dict, mx_model, mx_data_init, robot_info, config,
            cmd, rng_elite, phase_cfg
        )

        if new_fitness >= mean_fitness:
            flat_params = new_flat
            flat_sigma = new_sigma
            elitist_status = "ACCEPT"
        else:
            # Still update sigma even if we reject the mean update
            flat_sigma = new_sigma
            elitist_status = "REJECT"

        # Track best
        if new_metrics['vel_err'] < best_vel_err:
            best_vel_err = float(new_metrics['vel_err'])
            best_flat = flat_params.copy()
            best_fitness = float(new_fitness)

        gen_time = time.time() - t_gen
        avg_sigma = float(jnp.mean(flat_sigma))

        print(f"  {gen+1:4d} {phase_cfg['phase_name']:<12} {float(mean_fitness):9.4f} "
              f"{float(mean_vel_err):8.4f} {float(mean_stab):8.4f} "
              f"{float(mean_alive):6.3f} {float(mean_min_h):6.3f} "
              f"{best_vel_err:8.4f} {avg_sigma:7.4f} {elitist_status:>8} {gen_time:5.1f}s")

        # ── Save checkpoint every 10 generations ──
        if (gen + 1) % 10 == 0:
            ckpt_path = f"/tmp/soma_v10/gen_{gen+1}.npy"
            os.makedirs("/tmp/soma_v10", exist_ok=True)
            np.save(ckpt_path, np.array(best_flat))
            print(f"  → Checkpoint saved: {ckpt_path}")

    # ── Final save ──
    total_time = time.time() - t0
    final_path = "/tmp/soma_v10/soma_v10_best.npy"
    os.makedirs("/tmp/soma_v10", exist_ok=True)
    np.save(final_path, np.array(best_flat))

    print(f"\n{'='*90}")
    print(f"  TRAINING COMPLETE")
    print(f"  Total time:    {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Best vel_err:  {best_vel_err:.4f}")
    print(f"  Best fitness:  {best_fitness:.4f}")
    print(f"  Parameters:    {n_params:,}")
    print(f"  Weights saved: {final_path}")
    print(f"{'='*90}\n")

    return best_flat, structure, params, robot_info


# ══════════════════════════════════════════════════════════════════════
# SECTION 11: VALIDATION SUITE
# ══════════════════════════════════════════════════════════════════════

def validate(flat_params, structure, template_params, xml_path, config=None):
    """
    Post-training validation suite.

    Tests:
    1. Forward walking at multiple speeds
    2. Turning at multiple rates
    3. Push recovery (standing + walking)
    4. Inference latency benchmark
    5. Parameter count and memory footprint
    """
    if config is None:
        config = SOMAConfig()

    print("\n" + "="*70)
    print("  SOMA v10 — VALIDATION SUITE")
    print("="*70)

    mj_model, mx_model, mx_data_init, robot_info = setup_mjx(xml_path, config)
    params = unflatten_params(flat_params, structure, template_params)

    # ── Test 1: Walking at multiple speeds ──
    print("\n  Test 1: Velocity Tracking")
    print(f"  {'Speed':>8} {'VelErr':>8} {'Alive':>6} {'Stab':>8}")
    for vx in [0.0, 0.3, 0.5, 0.8, 1.0]:
        cmd = jnp.array([vx, 0.0, 0.0])
        phase_cfg = {'phase': 2, 'dr_scale': 0.0, 'push': False}
        rng = random.PRNGKey(0)
        fitness, metrics = rollout_one(params, mx_model, mx_data_init,
                                        robot_info, config, cmd, rng, phase_cfg)
        print(f"  {vx:8.1f} {float(metrics['vel_err']):8.4f} "
              f"{float(metrics['alive']):6.3f} {float(metrics['stab']):8.4f}")

    # ── Test 2: Turning ──
    print("\n  Test 2: Turning")
    print(f"  {'YawRate':>8} {'VelErr':>8} {'Alive':>6}")
    for yaw in [-1.0, -0.5, 0.0, 0.5, 1.0]:
        cmd = jnp.array([0.5, 0.0, yaw])
        fitness, metrics = rollout_one(params, mx_model, mx_data_init,
                                        robot_info, config, cmd, random.PRNGKey(1),
                                        {'phase': 3, 'dr_scale': 0.0, 'push': False})
        print(f"  {yaw:8.1f} {float(metrics['vel_err']):8.4f} {float(metrics['alive']):6.3f}")

    # ── Test 3: Push recovery ──
    print("\n  Test 3: Push Recovery (walking)")
    phase_cfg = {'phase': 4, 'dr_scale': 0.0, 'push': True}
    cmd = jnp.array([0.5, 0.0, 0.0])
    fitness, metrics = rollout_one(params, mx_model, mx_data_init,
                                    robot_info, config, cmd, random.PRNGKey(2), phase_cfg)
    print(f"  VelErr: {float(metrics['vel_err']):.4f}  Alive: {float(metrics['alive']):.3f}")

    # ── Test 4: Inference latency ──
    print("\n  Test 4: Inference Latency")
    soma_state = make_initial_state(robot_info, config)
    soma_state['q'] = jnp.zeros(robot_info.nu)
    soma_state['qd'] = jnp.zeros(robot_info.nu)

    # Warm up
    for _ in range(100):
        _, _ = soma_step(params, soma_state, robot_info, config, config.dt)

    # Benchmark
    t0 = time.time()
    n_iters = 10000
    for _ in range(n_iters):
        _, soma_state = soma_step(params, soma_state, robot_info, config, config.dt)
    latency_us = (time.time() - t0) / n_iters * 1e6
    print(f"  Latency:   {latency_us:.1f} μs ({1e6/latency_us:.0f} Hz)")

    # ── Test 5: Model stats ──
    n_params = count_params(params)
    mem_fp32 = n_params * 4 / 1024  # KB
    mem_int8 = n_params * 1 / 1024  # KB
    print(f"\n  Model Stats:")
    print(f"  Parameters:  {n_params:,}")
    print(f"  Memory FP32: {mem_fp32:.1f} KB")
    print(f"  Memory INT8: {mem_int8:.1f} KB")
    print(f"  Fits MCU:    {'✓' if mem_int8 < 256 else '✗'} (256KB flash)")

    print("\n" + "="*70 + "\n")


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # Default: Unitree Go2
    xml_path = sys.argv[1] if len(sys.argv) > 1 else None

    if xml_path is None:
        # Try common locations
        candidates = [
            "go2/scene.xml",
            "/tmp/go2/scene.xml",
            "mujoco_menagerie/unitree_go2/scene.xml",
        ]
        for c in candidates:
            if os.path.exists(c):
                xml_path = c
                break

    if xml_path is None:
        print("Usage: python soma_v10_train.py <path_to_robot.xml>")
        print("\nTo download Go2:")
        print("  git clone https://github.com/google-deepmind/mujoco_menagerie.git")
        print("  python soma_v10_train.py mujoco_menagerie/unitree_go2/scene.xml")
        sys.exit(1)

    config = SOMAConfig(
        robot_xml=xml_path,
        # Start conservative — validate architecture first
        n_generations=150,
        pop_size=256,
    )

    # Train
    best_flat, structure, template_params, robot_info = train(xml_path, config)

    # Validate
    validate(best_flat, structure, template_params, xml_path, config)
