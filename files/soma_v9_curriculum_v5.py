"""
SOMA v9 — Curriculum Trainer v5 (vmap scenarios + linearize + vmap dirs)
=========================================================================
Full spinal cord: CPG → Coord CfC → MoE(6) → Limb CfCs(4) → PD → CBF → motors

Root cause analysis of v3/v4 failures:
  Problem A — Gradient noise: ES finite-difference noisy → JVP exact (fixed in v3)
  Problem B — Scenario alignment:
    Fixed scenarios → memorization in 20 gens (v4 failure)
    Random scenarios → gradient/gate see different objectives, all rejected (v3 failure)
  These are INDEPENDENT problems. v5 fixes both simultaneously.

v5 solution:
  - Fresh stratified scenarios sampled ONCE per gen
  - vmap rollout over all 8 scenarios → single batched forward pass
  - linearize over batched loss → ONE trace, exact JVP
  - vmap 200 directions over linear trace → parallel, basically free
  - Gradient AND elitist gate use SAME scenarios for that gen
    → No memorization (changes every gen)
    → No misalignment (consistent within gen)

Three nested parallelisms in one JIT call:
  vmap(8 scenarios) → linearize → vmap(200 directions)
  XLA sees whole graph, optimizes everything together.

Memory estimate: ~250MB (safe on 80GB H100)
Speed estimate: ~20s/gen → 150 gens in ~50 minutes

3 Curriculum Phases:
  Phase 1 (gen 0-49):    Multi-speed forward walking
  Phase 2 (gen 50-99):   + turning (vy, omega)
  Phase 3 (gen 100-149): + push recovery (100-300N)
"""
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
from jax import random
import mujoco
from mujoco import mjx
import numpy as np
import time
import json

print(f"JAX: {jax.devices()}")

# ============================================================
# Robot
# ============================================================
GO2_XML = """
<mujoco model="go2">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" solver="Newton" iterations="4" ls_iterations="8">
    <flag contact="enable"/>
  </option>
  <default>
    <joint armature="0.01" damping="0.5" frictionloss="0.1"/>
    <geom friction="1.0 0.005 0.001" contype="1" conaffinity="1" condim="3"/>
    <motor ctrllimited="true"/>
  </default>
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1" diffuse="0.8 0.8 0.8"/>
    <geom name="floor" type="plane" size="10 10 0.1" rgba="0.8 0.9 0.8 1" contype="1" conaffinity="1"/>
    <body name="base" pos="0 0 0.35">
      <freejoint name="root"/>
      <geom type="box" size="0.2 0.05 0.04" mass="6.0" rgba="0.2 0.3 0.6 1" contype="0" conaffinity="0"/>
      <body name="FL_hip" pos="0.15 0.07 0"><joint name="FL_hip_joint" type="hinge" axis="1 0 0" range="-0.8 0.8"/><geom type="capsule" fromto="0 0 0 0 0.05 0" size="0.015" mass="0.5" contype="0" conaffinity="0"/>
        <body name="FL_thigh" pos="0 0.05 0"><joint name="FL_thigh_joint" type="hinge" axis="0 1 0" range="-1.5 3.0"/><geom type="capsule" fromto="0 0 0 0 0 -0.20" size="0.018" mass="1.0" contype="0" conaffinity="0"/>
          <body name="FL_calf" pos="0 0 -0.20"><joint name="FL_calf_joint" type="hinge" axis="0 1 0" range="-2.8 -0.5"/><geom type="capsule" fromto="0 0 0 0 0 -0.20" size="0.015" mass="0.3" contype="1" conaffinity="1"/>
      </body></body></body>
      <body name="FR_hip" pos="0.15 -0.07 0"><joint name="FR_hip_joint" type="hinge" axis="1 0 0" range="-0.8 0.8"/><geom type="capsule" fromto="0 0 0 0 -0.05 0" size="0.015" mass="0.5" contype="0" conaffinity="0"/>
        <body name="FR_thigh" pos="0 -0.05 0"><joint name="FR_thigh_joint" type="hinge" axis="0 1 0" range="-1.5 3.0"/><geom type="capsule" fromto="0 0 0 0 0 -0.20" size="0.018" mass="1.0" contype="0" conaffinity="0"/>
          <body name="FR_calf" pos="0 0 -0.20"><joint name="FR_calf_joint" type="hinge" axis="0 1 0" range="-2.8 -0.5"/><geom type="capsule" fromto="0 0 0 0 0 -0.20" size="0.015" mass="0.3" contype="1" conaffinity="1"/>
      </body></body></body>
      <body name="RL_hip" pos="-0.15 0.07 0"><joint name="RL_hip_joint" type="hinge" axis="1 0 0" range="-0.8 0.8"/><geom type="capsule" fromto="0 0 0 0 0.05 0" size="0.015" mass="0.5" contype="0" conaffinity="0"/>
        <body name="RL_thigh" pos="0 0.05 0"><joint name="RL_thigh_joint" type="hinge" axis="0 1 0" range="-1.5 3.0"/><geom type="capsule" fromto="0 0 0 0 0 -0.20" size="0.018" mass="1.0" contype="0" conaffinity="0"/>
          <body name="RL_calf" pos="0 0 -0.20"><joint name="RL_calf_joint" type="hinge" axis="0 1 0" range="-2.8 -0.5"/><geom type="capsule" fromto="0 0 0 0 0 -0.20" size="0.015" mass="0.3" contype="1" conaffinity="1"/>
      </body></body></body>
      <body name="RR_hip" pos="-0.15 -0.07 0"><joint name="RR_hip_joint" type="hinge" axis="1 0 0" range="-0.8 0.8"/><geom type="capsule" fromto="0 0 0 0 -0.05 0" size="0.015" mass="0.5" contype="0" conaffinity="0"/>
        <body name="RR_thigh" pos="0 -0.05 0"><joint name="RR_thigh_joint" type="hinge" axis="0 1 0" range="-1.5 3.0"/><geom type="capsule" fromto="0 0 0 0 0 -0.20" size="0.018" mass="1.0" contype="0" conaffinity="0"/>
          <body name="RR_calf" pos="0 0 -0.20"><joint name="RR_calf_joint" type="hinge" axis="0 1 0" range="-2.8 -0.5"/><geom type="capsule" fromto="0 0 0 0 0 -0.20" size="0.015" mass="0.3" contype="1" conaffinity="1"/>
      </body></body></body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="FL_hip_joint" ctrlrange="-25 25" gear="1"/><motor joint="FL_thigh_joint" ctrlrange="-25 25" gear="1"/><motor joint="FL_calf_joint" ctrlrange="-40 40" gear="1"/>
    <motor joint="FR_hip_joint" ctrlrange="-25 25" gear="1"/><motor joint="FR_thigh_joint" ctrlrange="-25 25" gear="1"/><motor joint="FR_calf_joint" ctrlrange="-40 40" gear="1"/>
    <motor joint="RL_hip_joint" ctrlrange="-25 25" gear="1"/><motor joint="RL_thigh_joint" ctrlrange="-25 25" gear="1"/><motor joint="RL_calf_joint" ctrlrange="-40 40" gear="1"/>
    <motor joint="RR_hip_joint" ctrlrange="-25 25" gear="1"/><motor joint="RR_thigh_joint" ctrlrange="-25 25" gear="1"/><motor joint="RR_calf_joint" ctrlrange="-40 40" gear="1"/>
  </actuator>
  <keyframe>
    <key name="stand" qpos="0 0 0.35  1 0 0 0  0 0.8 -1.5  0 0.8 -1.5  0 0.8 -1.5  0 0.8 -1.5"/>
  </keyframe>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(GO2_XML)
mj_data = mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model, mj_data, 0)
mujoco.mj_forward(model, mj_data)
mx_model = mjx.put_model(model)
mx_data_init = mjx.put_data(model, mj_data)

NU = model.nu
NV = model.nv
q_stand = jnp.array(mj_data.qpos[7:7+NU])
q_min = jnp.array(model.jnt_range[1:NU+1, 0])
q_max = jnp.array(model.jnt_range[1:NU+1, 1])
t_min = jnp.array(model.actuator_ctrlrange[:, 0])
t_max = jnp.array(model.actuator_ctrlrange[:, 1])
GAIT_OFFSETS = jnp.array([0.0, jnp.pi, jnp.pi, 0.0])

print(f"Go2: NU={NU}, NV={NV}")

# ============================================================
# Config
# ============================================================
H_c, H_e, H_l = 32, 16, 8
n_exp, n_limb = 6, 4
STATE_DIM = 58
DT = 0.002

alpha = 1.0
gain_range = 0.5
KP, KD = 80.0, 8.0
cbf_apos, qd_max, cbf_tr = 10.0, 20.0, 500.0
cpg_freq, cpg_amp, cpg_calf = 2.0, 0.5, 0.3

# v5 config
N_DIRS = 200         # JVP directions (vmap over linear trace — basically free)
N_STEPS = 1000       # full 2s rollout
N_GENS = 150
N_SCENARIOS = 8      # vmapped scenarios — fresh stratified every gen
SAVE_EVERY = 10
SAVE_DIR = '/tmp/soma_curriculum_v5'

# Adam
LR = 0.001
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8

# Grad clipping
MAX_GRAD_NORM = 1.0

# Fitness weights
W_VEL = 8.0
W_LAT = 4.0
W_YAW = 4.0
W_STAB = 1.0
W_ENERGY = 0.05
W_SMOOTH = 0.05

# Phases
PHASE1_END = 50
PHASE2_END = 100
PHASE3_END = 150

# ============================================================
# CfC
# ============================================================
def init_cfc(key, ind, hd, od):
    k1, k2, k3 = random.split(key, 3)
    si = 0.1/jnp.sqrt(float(ind)); sh = 0.1/jnp.sqrt(float(hd))
    return {'W_in': random.normal(k1,(ind,4*hd))*si, 'W_hh': random.normal(k2,(hd,4*hd))*sh,
            'b': jnp.zeros(4*hd), 'W_out': random.normal(k3,(hd,od))*0.01, 'b_out': jnp.zeros(od)}

def cfc_step(p, x, h, dt):
    H = h.shape[-1]; pr = x@p['W_in']+h@p['W_hh']+p['b']
    f1,f2 = pr[...,:H],pr[...,H:2*H]; ta,tb = pr[...,2*H:3*H],pr[...,3*H:]
    s = jax.nn.sigmoid(ta*dt+tb); hn = jnp.tanh(f1)*(1-s)+s*jnp.tanh(f2)
    return hn@p['W_out']+p['b_out'], hn

def init_params(key):
    ks = random.split(key, 20)
    p = {'coord': init_cfc(ks[0], STATE_DIM, H_c, H_c+n_exp+1)}
    for i in range(n_exp): p[f'expert_{i}'] = init_cfc(ks[1+i], STATE_DIM+H_c, H_e, 3*NU)
    for i in range(n_limb): p[f'limb_{i}'] = init_cfc(ks[7+i], 23, H_l, 12)
    return p

def flatten(p):
    l = []
    for k in sorted(p.keys()):
        for sk in sorted(p[k].keys()): l.append(p[k][sk].reshape(-1))
    return jnp.concatenate(l)

def unflatten(f, t):
    p = {}; i = 0
    for k in sorted(t.keys()):
        p[k] = {}
        for sk in sorted(t[k].keys()):
            s,n = t[k][sk].shape, t[k][sk].size; p[k][sk] = f[i:i+n].reshape(s); i += n
    return p

tmpl = init_params(random.PRNGKey(42))
flat0 = flatten(tmpl)
N = flat0.shape[0]
print(f"Params: {N:,}")

def qri(q, v):
    w = q[...,0:1]; xyz = q[...,1:4]; t = 2*jnp.cross(xyz, v)
    return v + w*t + jnp.cross(xyz, t)

# ============================================================
# Load best available weights
# ============================================================
# ============================================================
# Load weights — direct load, no ambiguity
# ============================================================
_WEIGHT_FILE = '/tmp/soma_v5_init.npy'
if not os.path.exists(_WEIGHT_FILE):
    # fallback: copy from v3 if available
    import shutil
    for _src in ['/tmp/soma_curriculum_v3/soma_v9_v3_best_vel.npy',
                 '/tmp/soma_curriculum_v2/soma_v9_curv2_best_vel.npy',
                 'soma_v9_best_weights.npy']:
        if os.path.exists(_src):
            shutil.copy(_src, _WEIGHT_FILE)
            print(f"Copied {_src} → {_WEIGHT_FILE}")
            break

if os.path.exists(_WEIGHT_FILE):
    init_weights = jnp.array(np.load(_WEIGHT_FILE))
    print(f"Loaded weights: {_WEIGHT_FILE} ({init_weights.shape[0]:,} params)")
    assert init_weights.shape[0] == N, f"Shape mismatch: {init_weights.shape[0]} != {N}"
else:
    print("WARNING: No weights found. Starting from random init.")
    init_weights = flat0

# ============================================================
# CPG
# ============================================================
def cpg_ref(phase, vx_cmd, vy_cmd, omega_cmd):
    lp = (phase + GAIT_OFFSETS) % (2*jnp.pi)
    speed_mag = jnp.maximum(jnp.abs(vx_cmd),
                jnp.maximum(jnp.abs(vy_cmd), jnp.abs(omega_cmd)*0.3))
    any_move = jnp.abs(vx_cmd)+jnp.abs(vy_cmd)+jnp.abs(omega_cmd)
    speed_mag = jnp.where(any_move > 0.01, jnp.maximum(speed_mag, 0.15), 0.0)
    qr = q_stand
    for leg in range(4):
        th = q_stand[leg*3+1]+cpg_amp*speed_mag*jnp.sin(lp[leg])
        ca = q_stand[leg*3+2]-cpg_calf*jnp.maximum(0.0,-jnp.sin(lp[leg]))
        qr = qr.at[leg*3+1].set(jnp.clip(th, q_min[leg*3+1], q_max[leg*3+1]))
        qr = qr.at[leg*3+2].set(jnp.clip(ca, q_min[leg*3+2], q_max[leg*3+2]))
    return qr, speed_mag

# ============================================================
# CBF
# ============================================================
def cbf_filter(ctrl, q, qd, prev_ctrl):
    margin_hi = q_max-q; margin_lo = q-q_min
    brake = jnp.where(qd > cbf_apos*margin_hi, -KD*2*(qd-cbf_apos*margin_hi), 0.0)
    brake += jnp.where(-qd > cbf_apos*margin_lo, KD*2*(-qd-cbf_apos*margin_lo), 0.0)
    ctrl = ctrl + brake
    vs = jnp.clip(1-(qd/qd_max)**2, 0.1, 1.0)
    ctrl = jnp.where((qd>0)&(ctrl>0)&(jnp.abs(qd)>qd_max*0.8), ctrl*vs, ctrl)
    ctrl = jnp.where((qd<0)&(ctrl<0)&(jnp.abs(qd)>qd_max*0.8), ctrl*vs, ctrl)
    ctrl = jnp.clip(ctrl, prev_ctrl-cbf_tr*DT, prev_ctrl+cbf_tr*DT)
    return jnp.clip(ctrl, t_min, t_max)

# ============================================================
# Rollout — returns LOSS (minimize this)
# ============================================================
def rollout_loss(flat_params, vx_cmd, vy_cmd, omega_cmd, push_force, push_start):
    params = unflatten(flat_params, tmpl)
    hc = jnp.zeros(H_c); he = jnp.zeros((n_exp,H_e)); hl = jnp.zeros((n_limb,H_l))
    push_end = push_start + 50

    def step(carry, step_idx):
        d, ph, hc, he, hl, pc = carry
        q = d.qpos[7:7+NU]; qd_j = d.qvel[6:6+NU]

        qr, speed_mag = cpg_ref(ph, vx_cmd, vy_cmd, omega_cmd)
        nph = (ph + 2*jnp.pi*(cpg_freq+0.5*speed_mag)*DT) % (2*jnp.pi)

        quat = d.qpos[3:7]; angvel = d.qvel[3:6]
        grav = qri(quat, jnp.array([0.,0.,-9.81]))
        lp = (nph+GAIT_OFFSETS) % (2*jnp.pi)
        ps = jnp.concatenate([jnp.sin(lp), jnp.cos(lp)])
        s = jnp.concatenate([q, qd_j/5.0, qr, quat, angvel, grav/9.81,
                             ps, jnp.array([vx_cmd, vy_cmd, omega_cmd]),
                             jnp.array([d.qpos[2]-0.35])])[None]

        co, hcn = cfc_step(params['coord'], s, hc[None], DT)
        co = co[0]; hcn = hcn[0]
        csig = co[:H_c]; gl = co[H_c:H_c+n_exp]
        tmp = jax.nn.softplus(co[-1:])+0.1; gw = jax.nn.softmax(gl/tmp)

        ei = jnp.concatenate([s[0], csig])[None]
        cb = jnp.zeros(3*NU); hen = jnp.zeros_like(he)
        for i in range(n_exp):
            eo, hei = cfc_step(params[f'expert_{i}'], ei, he[i:i+1], DT)
            cb = cb + gw[i]*eo[0]; hen = hen.at[i].set(hei[0])
        edq = jnp.tanh(cb[:NU]); ekp = jnp.tanh(cb[NU:2*NU]); ekd = jnp.tanh(cb[2*NU:])

        fdq = jnp.zeros(NU); fkp = jnp.zeros(NU)
        fkd = jnp.zeros(NU); frs = jnp.zeros(NU)
        hln = jnp.zeros_like(hl)
        for lb in range(n_limb):
            j0,j1 = lb*3,(lb+1)*3
            psl = jnp.array([ps[lb], ps[4+lb]])
            le = jnp.concatenate([edq[j0:j1], ekp[j0:j1], ekd[j0:j1]])
            li = jnp.concatenate([q[j0:j1], qd_j[j0:j1]/5.0, qr[j0:j1], psl, grav/9.81, le])[None]
            lo, hli = cfc_step(params[f'limb_{lb}'], li, hl[lb:lb+1], DT)
            lo = lo[0]; hli = hli[0]; hln = hln.at[lb].set(hli)
            fdq = fdq.at[j0:j1].set(0.7*edq[j0:j1]+0.3*jnp.tanh(lo[:3]))
            fkp = fkp.at[j0:j1].set(0.7*ekp[j0:j1]+0.3*jnp.tanh(lo[3:6]))
            fkd = fkd.at[j0:j1].set(0.7*ekd[j0:j1]+0.3*jnp.tanh(lo[6:9]))
            frs = frs.at[j0:j1].set(jnp.tanh(lo[9:12]))

        qtg = qr + alpha*fdq
        kp_ = KP*(1+gain_range*fkp); kd_ = KD*(1+gain_range*fkd)
        ctrl = kp_*(qtg-q) - kd_*qd_j + alpha*frs*10.0
        ctrl = cbf_filter(ctrl, q, qd_j, pc)

        # Push
        qfrc = jnp.zeros(NV)
        in_push = (step_idx >= push_start) & (step_idx < push_end)
        qfrc = qfrc.at[0].set(jnp.where(in_push, push_force[0], 0.0))
        qfrc = qfrc.at[1].set(jnp.where(in_push, push_force[1], 0.0))
        qfrc = qfrc.at[2].set(jnp.where(in_push, push_force[2], 0.0))

        nd = d.replace(ctrl=ctrl, qfrc_applied=qfrc)
        nd = mjx.step(mx_model, nd)

        bv = qri(nd.qpos[3:7], nd.qvel[0:3])
        metrics = jnp.array([bv[0], bv[1], nd.qvel[5], nd.qpos[2],
                             1.0-nd.qpos[3]**2, jnp.mean(ctrl**2),
                             jnp.mean((ctrl-pc)**2)])
        return (nd, nph, hcn, hen, hln, ctrl), metrics

    init = (mx_data_init, 0.0, hc, he, hl, jnp.zeros(NU))
    _, met = jax.lax.scan(step, init, jnp.arange(N_STEPS))

    half = N_STEPS // 2
    vex = jnp.mean((met[half:,0]-vx_cmd)**2)
    vey = jnp.mean((met[half:,1]-vy_cmd)**2)
    ye = jnp.mean((met[half:,2]-omega_cmd)**2)
    stab = jnp.mean(met[half:,4])
    energy = jnp.mean(met[half:,5])
    smooth = jnp.mean(met[half:,6])
    min_h = jnp.min(met[:,3])
    alive = jnp.where(min_h > 0.15, 0.0, 100.0)  # penalty if fell

    loss = alive + W_VEL*vex + W_LAT*vey + W_YAW*ye + W_STAB*stab + W_ENERGY*energy + W_SMOOTH*smooth
    return loss

def multi_scenario_loss(flat_params, scenarios):
    """Average loss across N_SCENARIOS. This is what we minimize."""
    total = 0.0
    for i in range(N_SCENARIOS):
        sc = scenarios[i]
        total = total + rollout_loss(
            flat_params, sc[0], sc[1], sc[2],
            jnp.array([sc[3], sc[4], sc[5]]),
            sc[6].astype(jnp.int32))
    return total / N_SCENARIOS

def eval_metrics(flat_params, scenarios):
    """Get individual metrics for logging (separate from loss)."""
    total_vex = 0.0; total_vey = 0.0; total_ye = 0.0; worst_h = 1.0
    total_loss = 0.0
    for i in range(N_SCENARIOS):
        sc = scenarios[i]
        params = unflatten(flat_params, tmpl)
        hc = jnp.zeros(H_c); he = jnp.zeros((n_exp,H_e)); hl = jnp.zeros((n_limb,H_l))

        def step(carry, step_idx):
            d, ph, hc, he, hl, pc = carry
            q = d.qpos[7:7+NU]; qd_j = d.qvel[6:6+NU]
            qr, speed_mag = cpg_ref(ph, sc[0], sc[1], sc[2])
            nph = (ph + 2*jnp.pi*(cpg_freq+0.5*speed_mag)*DT) % (2*jnp.pi)
            quat = d.qpos[3:7]; angvel = d.qvel[3:6]
            grav = qri(quat, jnp.array([0.,0.,-9.81]))
            lp = (nph+GAIT_OFFSETS) % (2*jnp.pi)
            ps = jnp.concatenate([jnp.sin(lp), jnp.cos(lp)])
            s = jnp.concatenate([q, qd_j/5.0, qr, quat, angvel, grav/9.81,
                                 ps, jnp.array([sc[0], sc[1], sc[2]]),
                                 jnp.array([d.qpos[2]-0.35])])[None]
            co, hcn = cfc_step(params['coord'], s, hc[None], DT); co=co[0]; hcn=hcn[0]
            csig = co[:H_c]; gl = co[H_c:H_c+n_exp]
            tmp = jax.nn.softplus(co[-1:])+0.1; gw = jax.nn.softmax(gl/tmp)
            ei = jnp.concatenate([s[0], csig])[None]
            cb = jnp.zeros(3*NU); hen = jnp.zeros_like(he)
            for ii in range(n_exp):
                eo, hei = cfc_step(params[f'expert_{ii}'], ei, he[ii:ii+1], DT)
                cb = cb + gw[ii]*eo[0]; hen = hen.at[ii].set(hei[0])
            edq=jnp.tanh(cb[:NU]); ekp=jnp.tanh(cb[NU:2*NU]); ekd=jnp.tanh(cb[2*NU:])
            fdq=jnp.zeros(NU); fkp=jnp.zeros(NU); fkd=jnp.zeros(NU); frs=jnp.zeros(NU)
            hln = jnp.zeros_like(hl)
            for lb in range(n_limb):
                j0,j1=lb*3,(lb+1)*3
                psl=jnp.array([ps[lb],ps[4+lb]])
                le=jnp.concatenate([edq[j0:j1],ekp[j0:j1],ekd[j0:j1]])
                li=jnp.concatenate([q[j0:j1],qd_j[j0:j1]/5.0,qr[j0:j1],psl,grav/9.81,le])[None]
                lo,hli=cfc_step(params[f'limb_{lb}'],li,hl[lb:lb+1],DT); lo=lo[0]; hli=hli[0]
                hln=hln.at[lb].set(hli)
                fdq=fdq.at[j0:j1].set(0.7*edq[j0:j1]+0.3*jnp.tanh(lo[:3]))
                fkp=fkp.at[j0:j1].set(0.7*ekp[j0:j1]+0.3*jnp.tanh(lo[3:6]))
                fkd=fkd.at[j0:j1].set(0.7*ekd[j0:j1]+0.3*jnp.tanh(lo[6:9]))
                frs=frs.at[j0:j1].set(jnp.tanh(lo[9:12]))
            qtg=qr+alpha*fdq; kp_=KP*(1+gain_range*fkp); kd_=KD*(1+gain_range*fkd)
            ctrl=kp_*(qtg-q)-kd_*qd_j+alpha*frs*10.0; ctrl=cbf_filter(ctrl,q,qd_j,pc)
            qfrc=jnp.zeros(NV)
            in_push=(step_idx>=sc[6].astype(jnp.int32))&(step_idx<sc[6].astype(jnp.int32)+50)
            qfrc=qfrc.at[0].set(jnp.where(in_push,sc[3],0.0))
            qfrc=qfrc.at[1].set(jnp.where(in_push,sc[4],0.0))
            qfrc=qfrc.at[2].set(jnp.where(in_push,sc[5],0.0))
            nd=d.replace(ctrl=ctrl,qfrc_applied=qfrc); nd=mjx.step(mx_model,nd)
            bv=qri(nd.qpos[3:7],nd.qvel[0:3])
            return (nd,nph,hcn,hen,hln,ctrl), jnp.array([bv[0],bv[1],nd.qvel[5],nd.qpos[2]])

        init = (mx_data_init, 0.0, hc, he, hl, jnp.zeros(NU))
        _, met = jax.lax.scan(step, init, jnp.arange(N_STEPS))
        half = N_STEPS//2
        total_vex += jnp.mean((met[half:,0]-sc[0])**2)
        total_vey += jnp.mean((met[half:,1]-sc[1])**2)
        total_ye += jnp.mean((met[half:,2]-sc[2])**2)
        worst_h = jnp.minimum(worst_h, jnp.min(met[:,3]))

    n = N_SCENARIOS
    return total_vex/n, total_vey/n, total_ye/n, worst_h

# ============================================================
# STRATIFIED SCENARIO GENERATION (from v2)
# ============================================================
def make_scenarios_stratified(gen, rng):
    scenarios = np.zeros((N_SCENARIOS, 7))

    if gen < PHASE1_END:
        # 8 scenarios: 2 per speed band [0-0.2, 0.2-0.4, 0.4-0.6, 0.6-0.8]
        for i, (lo, hi) in enumerate([(0.0,0.2),(0.0,0.2),(0.2,0.4),(0.2,0.4),
                                       (0.4,0.6),(0.4,0.6),(0.6,0.8),(0.6,0.8)]):
            scenarios[i,0] = rng.uniform(lo, hi)

    elif gen < PHASE2_END:
        # 8 scenarios: mix of straight + turning
        scenarios[0,0] = rng.uniform(0.2, 0.5)
        scenarios[1,0] = rng.uniform(0.4, 0.7)
        scenarios[2,0] = rng.uniform(0.3, 0.6)
        scenarios[2,2] = rng.uniform(-0.5, 0.5)
        if abs(scenarios[2,2]) < 0.1: scenarios[2,2] = 0.3*rng.choice([-1,1])
        scenarios[3,0] = rng.uniform(0.3, 0.6)
        scenarios[3,2] = rng.uniform(-0.5, 0.5)
        if abs(scenarios[3,2]) < 0.1: scenarios[3,2] = 0.3*rng.choice([-1,1])
        scenarios[4,0] = rng.uniform(0.3, 0.7)
        scenarios[4,1] = rng.uniform(-0.2, 0.2)
        if abs(scenarios[4,1]) < 0.05: scenarios[4,1] = 0.15*rng.choice([-1,1])
        scenarios[5,0] = rng.uniform(0.2, 0.5)
        scenarios[5,1] = rng.uniform(-0.2, 0.2)
        if abs(scenarios[5,1]) < 0.05: scenarios[5,1] = 0.15*rng.choice([-1,1])
        scenarios[6,0] = rng.uniform(0.2, 0.6)
        scenarios[6,1] = rng.uniform(-0.15, 0.15)
        scenarios[6,2] = rng.uniform(-0.4, 0.4)
        scenarios[7,0] = rng.uniform(0.2, 0.6)
        scenarios[7,1] = rng.uniform(-0.15, 0.15)
        scenarios[7,2] = rng.uniform(-0.4, 0.4)

    else:
        # 8 scenarios: mix of walking + push recovery
        scenarios[0,0] = rng.uniform(0.3, 0.7)
        scenarios[0,1] = rng.uniform(-0.1, 0.1)
        scenarios[0,2] = rng.uniform(-0.3, 0.3)
        scenarios[1,0] = rng.uniform(0.3, 0.7)
        scenarios[1,1] = rng.uniform(-0.1, 0.1)
        scenarios[1,2] = rng.uniform(-0.3, 0.3)
        for i in range(2, 8):
            scenarios[i,0] = rng.uniform(0.0, 0.6)
            pm = rng.uniform(100, 300)
            angle = rng.uniform(0, 2*np.pi)
            scenarios[i,3] = pm*np.cos(angle)
            scenarios[i,4] = pm*np.sin(angle)
            scenarios[i,6] = rng.integers(200, 700)

    return scenarios

def phase_name(gen):
    if gen < PHASE1_END: return "Phase1:Walk"
    elif gen < PHASE2_END: return "Phase2:Turn"
    else: return "Phase3:Push"

# Fixed eval scenarios
EVAL_FORWARD = np.array([
    [0.0,0,0,0,0,0,9999],[0.3,0,0,0,0,0,9999],
    [0.5,0,0,0,0,0,9999],[0.8,0,0,0,0,0,9999]], dtype=np.float32)
EVAL_TURN = np.array([
    [0.3,0,0,0,0,0,9999],[0.5,0,0.3,0,0,0,9999],
    [0.5,0.15,0,0,0,0,9999],[0.3,-0.1,-0.4,0,0,0,9999]], dtype=np.float32)
EVAL_PUSH = np.array([
    [0.5,0,0,0,0,0,9999],[0.5,0,0,200,0,0,300],
    [0.3,0,0.3,0,200,0,400],[0.0,0,0,250,0,0,300]], dtype=np.float32)

# ============================================================
# Save
# ============================================================
def save_weights(theta, gen, metrics, tag="checkpoint"):
    os.makedirs(SAVE_DIR, exist_ok=True)
    try:
        np.save(f"{SAVE_DIR}/soma_v9_v5_{tag}.npy", np.array(theta))
        meta = {'gen': gen, 'phase': phase_name(gen), **metrics,
                'n_params': int(N), 'optimizer': 'linearize+vmap+Adam', 'n_dirs': N_DIRS,
                'lr': LR, 'sampling': 'stratified', 'elitist': True}
        with open(f"{SAVE_DIR}/soma_v9_v3_{tag}_meta.json", 'w') as f:
            json.dump(meta, f, indent=2)
        np.save(f"{SAVE_DIR}/soma_v9_v5_latest.npy", np.array(theta))
    except Exception as e:
        print(f"    >> Save failed: {e}")

# ============================================================
# GRADIENT COMPUTATION v5 — three nested parallelisms
# ============================================================
# Fix for v4's scenario alignment problem:
#   v4: gradient on fixed scenarios → memorization
#   v4: gradient on random, gate on fixed → misalignment, all rejected
#
# v5: fresh stratified scenarios sampled once per gen
#     vmap rollout over ALL scenarios → batched forward pass
#     linearize the BATCHED loss → one trace
#     vmap 200 directions → parallel, basically free
#     gradient AND elitist gate use SAME scenarios for that gen
#
# XLA sees: vmap(8) → linearize → vmap(200) as one graph
# All three levels optimized together

@jax.jit
def compute_gradient_v5(flat_params, scenarios, directions):
    """
    Three nested parallelisms in one JIT call.
    scenarios: (N_SCENARIOS, 7) — fresh stratified for this gen
    directions: (N_DIRS, N) — random unit vectors
    Returns: (loss_val, grad_est)
    """
    def batched_loss(fp):
        # vmap rollout over all scenarios simultaneously
        losses = jax.vmap(
            lambda sc: rollout_loss(
                fp, sc[0], sc[1], sc[2],
                jnp.array([sc[3], sc[4], sc[5]]),
                sc[6].astype(jnp.int32)
            )
        )(scenarios)
        return jnp.mean(losses)

    # ONE linearize call traces through ALL vmapped scenarios
    loss_val, jvp_fn = jax.linearize(batched_loss, flat_params)

    # vmap 200 directions over cheap linear trace
    dloss_vals = jax.vmap(jvp_fn)(directions)  # (N_DIRS,)
    dloss_vals = jnp.clip(dloss_vals, -10.0, 10.0)

    # Gradient reconstruction
    grad_est = jnp.sum(dloss_vals[:, None] * directions, axis=0) * (N / N_DIRS)
    return loss_val, grad_est

# ============================================================
# Training
# ============================================================
def train():
    print(f"\n{'='*70}")
    print(f"SOMA v9 — CURRICULUM TRAINER v5 (vmap scenarios + linearize + vmap dirs)")
    print(f"{'='*70}")
    print(f"  Params:        {N:,}")
    print(f"  JVP dirs:      {N_DIRS} (exact directional derivatives)")
    print(f"  Scenarios:     {N_SCENARIOS} (stratified)")
    print(f"  Rollout:       {N_STEPS} steps ({N_STEPS*DT:.1f}s)")
    print(f"  Generations:   {N_GENS}")
    print(f"    Phase 1:     Gen 0-{PHASE1_END-1}  (multi-speed forward)")
    print(f"    Phase 2:     Gen {PHASE1_END}-{PHASE2_END-1}  (+ turning)")
    print(f"    Phase 3:     Gen {PHASE2_END}-{PHASE3_END-1} (+ push recovery)")
    print(f"  Optimizer:     Adam (LR={LR}, grad_clip={MAX_GRAD_NORM})")
    print(f"  Gradient:      vmap({N_SCENARIOS} scenarios) → linearize → vmap({N_DIRS} dirs)")
    print(f"  Elitist:       YES (reject worse, momentum reset after 3 consec)")
    print(f"  State norm:    grav/9.81 qd/5.0, alpha={alpha}")
    print()

    # Compile
    print("  Compiling v5 gradient (vmap+linearize+vmap)...")
    t0 = time.time()

    @jax.jit
    def eval_fn(fp, scen):
        return eval_metrics(fp, scen)

    @jax.jit
    def loss_fn_jit(fp, scen):
        return multi_scenario_loss(fp, scen)

    # Warmup compile — use actual N_SCENARIOS size
    np_rng0 = np.random.default_rng(0)
    scen0_np = make_scenarios_stratified(0, np_rng0)
    scen0 = jnp.array(scen0_np, dtype=jnp.float32)
    rng0 = random.PRNGKey(0)
    dirs0 = random.normal(rng0, (N_DIRS, N))
    dirs0 = dirs0 / jnp.linalg.norm(dirs0, axis=1, keepdims=True)

    loss0, grad0 = compute_gradient_v5(init_weights, scen0, dirs0)
    ct = time.time() - t0
    print(f"  Compiled in {ct:.1f}s")
    print(f"  Baseline loss: {float(loss0):.4f}, grad_norm: {float(jnp.linalg.norm(grad0)):.4f}")

    # Eval baseline on fixed eval
    eval_scen0 = jnp.array(EVAL_FORWARD)
    vex0, vey0, ye0, mh0 = eval_fn(init_weights, eval_scen0)
    print(f"  Baseline vel_err_x={float(vex0):.4f}, min_h={float(mh0):.3f}")

    # Timing
    t1 = time.time()
    for _ in range(3):
        _, _ = compute_gradient_v5(init_weights, scen0, dirs0)
    t_grad = (time.time() - t1) / 3
    gen_time = t_grad + 8  # gradient + elitist + eval overhead
    print(f"  Grad (v5): {t_grad:.2f}s")
    print(f"  Est per gen: {gen_time:.0f}s ({gen_time/60:.1f} min)")
    print(f"  Est total:   {gen_time*N_GENS/3600:.1f} hours ({gen_time*N_GENS/60:.0f} min)")
    print()

    # Adam state
    theta = init_weights.copy()
    m_adam = jnp.zeros(N); v_adam = jnp.zeros(N)

    best_vex = float('inf'); best_theta = theta.copy()
    best_loss = float('inf'); best_loss_theta = theta.copy()

    np_rng = np.random.default_rng(42)
    jax_rng = random.PRNGKey(7)

    # Track elitist stats
    accepted = 0; rejected = 0; consecutive_rejects = 0

    print(f"  {'Gen':>4} {'Phase':<12} {'Loss':>8} {'VelX':>8} {'VelY':>8} {'Yaw':>8} {'MinH':>6} {'BstVX':>8} {'Elitist':>8} {'Time':>5}")
    print(f"  {'-'*4} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*5}")

    for gen in range(N_GENS):
        tg = time.time()
        phase = phase_name(gen)

        # Fresh stratified scenarios — sampled once, used for EVERYTHING this gen
        # gradient, elitist gate, and eval all see the same batch
        # → no memorization (changes every gen)
        # → no misalignment (consistent within gen)
        scenarios_np = make_scenarios_stratified(gen, np_rng)
        scenarios = jnp.array(scenarios_np, dtype=jnp.float32)

        # Generate random unit directions
        jax_rng, dk = random.split(jax_rng)
        directions = random.normal(dk, (N_DIRS, N))
        directions = directions / jnp.linalg.norm(directions, axis=1, keepdims=True)

        # v5 gradient: vmap(8 scenarios) → linearize → vmap(200 dirs)
        _, grad_est = compute_gradient_v5(theta, scenarios, directions)

        # Gradient clipping
        grad_norm = jnp.linalg.norm(grad_est)
        grad_est = jnp.where(grad_norm > MAX_GRAD_NORM,
                             grad_est * MAX_GRAD_NORM / grad_norm,
                             grad_est)

        # Adam update
        m_adam = ADAM_B1*m_adam + (1-ADAM_B1)*grad_est
        v_adam = ADAM_B2*v_adam + (1-ADAM_B2)*grad_est**2
        mhat = m_adam / (1 - ADAM_B1**(gen+1))
        vhat = v_adam / (1 - ADAM_B2**(gen+1))
        theta_candidate = theta - LR * mhat / (jnp.sqrt(vhat) + ADAM_EPS)

        # Elitist check — same fresh scenarios as gradient (aligned)
        current_loss_fixed   = float(loss_fn_jit(theta,           scenarios))
        candidate_loss_fixed = float(loss_fn_jit(theta_candidate, scenarios))

        if candidate_loss_fixed < current_loss_fixed:
            theta = theta_candidate
            accepted += 1
            elite_str = "accept"
            consecutive_rejects = 0
        else:
            rejected += 1
            elite_str = "REJECT"
            consecutive_rejects += 1
            # Reset Adam momentum after 3 consecutive rejections
            # Stale momentum from rejected steps points the wrong direction
            if consecutive_rejects >= 3:
                m_adam = jnp.zeros(N)
                v_adam = jnp.zeros(N)
                consecutive_rejects = 0
                elite_str = "REJECT*"  # * = momentum reset

        # Evaluate on fixed eval set for consistent BstVX tracking across gens
        if gen < PHASE1_END:
            eval_scen = jnp.array(EVAL_FORWARD)
        elif gen < PHASE2_END:
            eval_scen = jnp.array(EVAL_TURN)
        else:
            eval_scen = jnp.array(EVAL_PUSH)

        vexc, veyc, yec, mhc = eval_fn(theta, eval_scen)
        vexc, veyc, yec, mhc = float(vexc), float(veyc), float(yec), float(mhc)
        eval_loss = float(loss_fn_jit(theta, eval_scen))

        if vexc < best_vex:
            best_vex = vexc; best_theta = theta.copy()
        if eval_loss < best_loss:
            best_loss = eval_loss; best_loss_theta = theta.copy()

        gt = time.time() - tg
        print(f"  {gen+1:>4} {phase:<12} {eval_loss:>8.3f} {vexc:>8.4f} {veyc:>8.4f} {yec:>8.4f} {mhc:>6.3f} {best_vex:>8.4f} {elite_str:>8} {gt:>4.0f}s")

        if gen+1 == PHASE1_END:
            print(f"\n  >>> PHASE 1 COMPLETE — switching to turning <<<")
            print(f"  >>> Accepted: {accepted}, Rejected: {rejected} ({100*rejected/max(1,accepted+rejected):.0f}% rejected) <<<\n")
        if gen+1 == PHASE2_END:
            print(f"\n  >>> PHASE 2 COMPLETE — switching to push recovery <<<")
            print(f"  >>> Accepted: {accepted}, Rejected: {rejected} ({100*rejected/max(1,accepted+rejected):.0f}% rejected) <<<\n")

        if (gen+1) % SAVE_EVERY == 0:
            save_weights(theta, gen+1,
                        {'vel_err_x': vexc, 'vel_err_y': veyc, 'yaw_err': yec, 'min_h': mhc},
                        f"gen{gen+1:03d}")
            save_weights(best_theta, gen+1, {'best_vel_err_x': best_vex}, "best_vel")
            print(f"    >> Saved gen{gen+1:03d} + best_vel | Accept rate: {100*accepted/max(1,accepted+rejected):.0f}%")

    # ============================================================
    # Final save
    # ============================================================
    save_weights(best_theta, N_GENS, {'best_vel_err': best_vex}, "final_best_vel")
    save_weights(best_loss_theta, N_GENS, {'best_loss': best_loss}, "final_best_loss")

    try:
        np.save('/mnt/user-data/outputs/soma_v9_v5_weights.npy', np.array(best_theta))
    except: pass

    # ============================================================
    # Final eval
    # ============================================================
    print(f"\n{'='*70}")
    print(f"FINAL EVALUATION")
    print(f"{'='*70}")

    @jax.jit
    def eval_single(fp, vx, vy, om, pf, ps):
        params = unflatten(fp, tmpl)
        hc=jnp.zeros(H_c); he=jnp.zeros((n_exp,H_e)); hl=jnp.zeros((n_limb,H_l))
        def step(carry, step_idx):
            d,ph,hc,he,hl,pc = carry
            q=d.qpos[7:7+NU]; qd_j=d.qvel[6:6+NU]
            qr,sm = cpg_ref(ph, vx, vy, om)
            nph=(ph+2*jnp.pi*(cpg_freq+0.5*sm)*DT)%(2*jnp.pi)
            quat=d.qpos[3:7]; angvel=d.qvel[3:6]
            grav=qri(quat,jnp.array([0.,0.,-9.81]))
            lp=(nph+GAIT_OFFSETS)%(2*jnp.pi); ps_=jnp.concatenate([jnp.sin(lp),jnp.cos(lp)])
            s=jnp.concatenate([q,qd_j/5.0,qr,quat,angvel,grav/9.81,ps_,
                               jnp.array([vx,vy,om]),jnp.array([d.qpos[2]-0.35])])[None]
            co,hcn=cfc_step(params['coord'],s,hc[None],DT); co=co[0]; hcn=hcn[0]
            csig=co[:H_c]; gl=co[H_c:H_c+n_exp]
            tmp=jax.nn.softplus(co[-1:])+0.1; gw=jax.nn.softmax(gl/tmp)
            ei=jnp.concatenate([s[0],csig])[None]
            cb=jnp.zeros(3*NU); hen=jnp.zeros_like(he)
            for ii in range(n_exp):
                eo,hei=cfc_step(params[f'expert_{ii}'],ei,he[ii:ii+1],DT)
                cb=cb+gw[ii]*eo[0]; hen=hen.at[ii].set(hei[0])
            edq=jnp.tanh(cb[:NU]); ekp=jnp.tanh(cb[NU:2*NU]); ekd=jnp.tanh(cb[2*NU:])
            fdq=jnp.zeros(NU);fkp=jnp.zeros(NU);fkd=jnp.zeros(NU);frs=jnp.zeros(NU)
            hln=jnp.zeros_like(hl)
            for lb in range(n_limb):
                j0,j1=lb*3,(lb+1)*3
                psl=jnp.array([ps_[lb],ps_[4+lb]])
                le=jnp.concatenate([edq[j0:j1],ekp[j0:j1],ekd[j0:j1]])
                li=jnp.concatenate([q[j0:j1],qd_j[j0:j1]/5.0,qr[j0:j1],psl,grav/9.81,le])[None]
                lo,hli=cfc_step(params[f'limb_{lb}'],li,hl[lb:lb+1],DT); lo=lo[0];hli=hli[0]
                hln=hln.at[lb].set(hli)
                fdq=fdq.at[j0:j1].set(0.7*edq[j0:j1]+0.3*jnp.tanh(lo[:3]))
                fkp=fkp.at[j0:j1].set(0.7*ekp[j0:j1]+0.3*jnp.tanh(lo[3:6]))
                fkd=fkd.at[j0:j1].set(0.7*ekd[j0:j1]+0.3*jnp.tanh(lo[6:9]))
                frs=frs.at[j0:j1].set(jnp.tanh(lo[9:12]))
            qtg=qr+alpha*fdq; kp_=KP*(1+gain_range*fkp); kd_=KD*(1+gain_range*fkd)
            ctrl=kp_*(qtg-q)-kd_*qd_j+alpha*frs*10.0; ctrl=cbf_filter(ctrl,q,qd_j,pc)
            qfrc=jnp.zeros(NV)
            in_push=(step_idx>=ps)&(step_idx<ps+50)
            qfrc=qfrc.at[0].set(jnp.where(in_push,pf[0],0.0))
            qfrc=qfrc.at[1].set(jnp.where(in_push,pf[1],0.0))
            nd=d.replace(ctrl=ctrl,qfrc_applied=qfrc); nd=mjx.step(mx_model,nd)
            bv=qri(nd.qpos[3:7],nd.qvel[0:3])
            return (nd,nph,hcn,hen,hln,ctrl), jnp.array([bv[0],bv[1],nd.qvel[5],nd.qpos[2]])
        init=(mx_data_init,0.0,hc,he,hl,jnp.zeros(NU))
        _,met=jax.lax.scan(step,init,jnp.arange(N_STEPS))
        half=N_STEPS//2
        return (jnp.mean(met[half:,0]),jnp.mean(met[half:,1]),jnp.mean(met[half:,2]),
                jnp.min(met[:,3]),jnp.mean((met[half:,0]-vx)**2))

    bw = best_theta

    print(f"\n  FORWARD WALKING (includes untrained speeds *):")
    print(f"  {'cmd':>6} {'actual':>8} {'vel_err':>8} {'min_h':>6} {'st'}")
    for vx in [0.0, 0.1, 0.15, 0.2, 0.3, 0.37, 0.4, 0.5, 0.55, 0.6, 0.71, 0.8, 1.0]:
        avx,_,_,mh,ve = eval_single(bw,jnp.float32(vx),0.0,0.0,jnp.zeros(3),jnp.int32(9999))
        st="OK" if float(mh)>0.15 else "FELL"
        tr=" " if vx in [0.0,0.2,0.4,0.6,0.8] else "*"
        print(f"  {vx:>5.2f}{tr} {float(avx):>8.4f} {float(ve):>8.4f} {float(mh):>6.3f} {st}")

    print(f"\n  TURNING:")
    for vx,vy,om in [(0.3,0,0.3),(0.3,0,-0.3),(0.5,0.15,0),(0.5,-0.15,0),
                      (0.5,0,0.5),(0.0,0,0.3),(0.4,0.1,0.2)]:
        avx,avy,ayr,mh,_ = eval_single(bw,jnp.float32(vx),jnp.float32(vy),
                                          jnp.float32(om),jnp.zeros(3),jnp.int32(9999))
        st="OK" if float(mh)>0.15 else "FELL"
        print(f"  vx={vx:.1f} vy={vy:.2f} om={om:.2f} → avx={float(avx):.3f} avy={float(avy):.3f} ayr={float(ayr):.3f} mh={float(mh):.3f} {st}")

    print(f"\n  PUSH RECOVERY:")
    for vx,pfx,pfy,lab in [(0.5,100,0,"fwd100"),(0.5,200,0,"fwd200"),
                            (0.5,300,0,"fwd300"),(0.5,0,200,"lat200"),
                            (0.0,200,0,"stand+fwd"),(0.0,0,250,"stand+lat")]:
        _,_,_,mh,_ = eval_single(bw,jnp.float32(vx),0.0,0.0,
                                    jnp.array([pfx,pfy,0],dtype=jnp.float32),jnp.int32(300))
        st="OK" if float(mh)>0.15 else "FELL"
        print(f"  vx={vx:.1f} {lab:>12} mh={float(mh):.3f} {st}")

    print(f"\n{'='*70}")
    print(f"COMPLETE")
    print(f"  Best vel_err_x: {best_vex:.4f}")
    print(f"  Accepted: {accepted}, Rejected: {rejected}")
    print(f"  Accept rate: {100*accepted/max(1,accepted+rejected):.0f}%")
    print(f"{'='*70}")

    return best_theta

best = train()
