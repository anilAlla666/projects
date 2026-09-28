"""
SOMA v12 — Step 1.5: Train MLP PPO Walker + Collect Walking Data
==================================================================
Trains a THROWAWAY MLP policy via PPO to walk Go2 forward.
Then collects 3M transitions of actual walking for world model training.

This MLP is NOT SOMA. It's a data collection tool. We throw it away.

REQUIRES: mujoco==3.2.7, mujoco-mjx==3.2.7, jax==0.4.38
Run in same Colab as Step 1 (MJX env already compiled).
"""

import os, sys, time
import numpy as np

os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

import jax
import jax.numpy as jnp
from jax import random, vmap, jit, lax
import mujoco
from mujoco import mjx
from functools import partial

print("=" * 60)
print("  SOMA v12 — Step 1.5: MLP PPO Walker")
print("  Train throwaway policy → collect walking data")
print("=" * 60)
print(f"  JAX: {jax.__version__} | Backend: {jax.default_backend()}")

# ═══════════════════ CONFIG ═══════════════════

N_ENVS = 2048               # Parallel envs for PPO
PPO_STEPS = 64               # Steps per rollout
PPO_EPOCHS = 4               # Epochs per PPO update
PPO_MINIBATCHES = 4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
LR = 3e-4
ENT_COEF = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TOTAL_UPDATES = 300          # ~300 updates × 2048 envs × 64 steps = ~39M steps
N_SUBSTEPS = 10              # 50Hz policy
KP = 25.0
KD = 0.5

# Data collection after training
COLLECT_TRANSITIONS = 3_000_000
SAVE_PATH = "go2_walking_data.npz"
MERGED_PATH = "go2_merged_data.npz"

# ═══════════════════ GO2 MJX ENV ═══════════════════

print("\n[ENV] Setting up Go2 MJX environment...")

GO2_XML = """<mujoco model="go2_mjx">
  <option timestep="0.002" gravity="0 0 -9.81"/>
  <default>
    <joint damping="0.5" armature="0.01"/>
    <geom friction="0.8 0.02 0.01" condim="3" contype="1" conaffinity="1"/>
  </default>
  <worldbody>
    <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="50 50 0.1" rgba=".9 .9 .9 1"/>
    <body name="base" pos="0 0 0.35">
      <freejoint name="root"/>
      <geom type="capsule" size="0.08" fromto="-0.2 0 0 0.2 0 0" mass="6.0" rgba="0.2 0.2 0.8 1"/>
      <body name="FR_hip" pos="0.15 -0.08 0">
        <joint name="FR_hip_j" type="hinge" axis="1 0 0" range="-0.86 0.86"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 -0.04 0" mass="0.5"/>
        <body name="FR_thigh" pos="0 -0.04 0">
          <joint name="FR_thigh_j" type="hinge" axis="0 1 0" range="-1.05 2.27"/>
          <geom type="capsule" size="0.015" fromto="0 0 0 0 0 -0.213" mass="0.6"/>
          <body name="FR_calf" pos="0 0 -0.213">
            <joint name="FR_calf_j" type="hinge" axis="0 1 0" range="-2.72 -0.92"/>
            <geom type="capsule" size="0.012" fromto="0 0 0 0 0 -0.213" mass="0.15"/>
            <body name="FR_foot" pos="0 0 -0.213">
              <geom type="sphere" size="0.02" mass="0.05"/>
            </body></body></body></body>
      <body name="FL_hip" pos="0.15 0.08 0">
        <joint name="FL_hip_j" type="hinge" axis="1 0 0" range="-0.86 0.86"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0.04 0" mass="0.5"/>
        <body name="FL_thigh" pos="0 0.04 0">
          <joint name="FL_thigh_j" type="hinge" axis="0 1 0" range="-1.05 2.27"/>
          <geom type="capsule" size="0.015" fromto="0 0 0 0 0 -0.213" mass="0.6"/>
          <body name="FL_calf" pos="0 0 -0.213">
            <joint name="FL_calf_j" type="hinge" axis="0 1 0" range="-2.72 -0.92"/>
            <geom type="capsule" size="0.012" fromto="0 0 0 0 0 -0.213" mass="0.15"/>
            <body name="FL_foot" pos="0 0 -0.213">
              <geom type="sphere" size="0.02" mass="0.05"/>
            </body></body></body></body>
      <body name="RR_hip" pos="-0.15 -0.08 0">
        <joint name="RR_hip_j" type="hinge" axis="1 0 0" range="-0.86 0.86"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 -0.04 0" mass="0.5"/>
        <body name="RR_thigh" pos="0 -0.04 0">
          <joint name="RR_thigh_j" type="hinge" axis="0 1 0" range="-1.05 2.27"/>
          <geom type="capsule" size="0.015" fromto="0 0 0 0 0 -0.213" mass="0.6"/>
          <body name="RR_calf" pos="0 0 -0.213">
            <joint name="RR_calf_j" type="hinge" axis="0 1 0" range="-2.72 -0.92"/>
            <geom type="capsule" size="0.012" fromto="0 0 0 0 0 -0.213" mass="0.15"/>
            <body name="RR_foot" pos="0 0 -0.213">
              <geom type="sphere" size="0.02" mass="0.05"/>
            </body></body></body></body>
      <body name="RL_hip" pos="-0.15 0.08 0">
        <joint name="RL_hip_j" type="hinge" axis="1 0 0" range="-0.86 0.86"/>
        <geom type="capsule" size="0.02" fromto="0 0 0 0 0.04 0" mass="0.5"/>
        <body name="RL_thigh" pos="0 0.04 0">
          <joint name="RL_thigh_j" type="hinge" axis="0 1 0" range="-1.05 2.27"/>
          <geom type="capsule" size="0.015" fromto="0 0 0 0 0 -0.213" mass="0.6"/>
          <body name="RL_calf" pos="0 0 -0.213">
            <joint name="RL_calf_j" type="hinge" axis="0 1 0" range="-2.72 -0.92"/>
            <geom type="capsule" size="0.012" fromto="0 0 0 0 0 -0.213" mass="0.15"/>
            <body name="RL_foot" pos="0 0 -0.213">
              <geom type="sphere" size="0.02" mass="0.05"/>
            </body></body></body></body>
    </body>
  </worldbody>
  <actuator>
    <motor joint="FR_hip_j" ctrlrange="-23.7 23.7"/>
    <motor joint="FR_thigh_j" ctrlrange="-23.7 23.7"/>
    <motor joint="FR_calf_j" ctrlrange="-35.5 35.5"/>
    <motor joint="FL_hip_j" ctrlrange="-23.7 23.7"/>
    <motor joint="FL_thigh_j" ctrlrange="-23.7 23.7"/>
    <motor joint="FL_calf_j" ctrlrange="-35.5 35.5"/>
    <motor joint="RR_hip_j" ctrlrange="-23.7 23.7"/>
    <motor joint="RR_thigh_j" ctrlrange="-23.7 23.7"/>
    <motor joint="RR_calf_j" ctrlrange="-35.5 35.5"/>
    <motor joint="RL_hip_j" ctrlrange="-23.7 23.7"/>
    <motor joint="RL_thigh_j" ctrlrange="-23.7 23.7"/>
    <motor joint="RL_calf_j" ctrlrange="-35.5 35.5"/>
  </actuator>
</mujoco>"""

os.makedirs("/tmp/go2", exist_ok=True)
with open("/tmp/go2/go2_mjx.xml", "w") as f:
    f.write(GO2_XML)

mj_model = mujoco.MjModel.from_xml_path("/tmp/go2/go2_mjx.xml")
mx_model = mjx.put_model(mj_model)

N_JOINTS = mj_model.nu
NQ = mj_model.nq
NV = mj_model.nv
OBS_DIM = N_JOINTS * 2 + 3 + 3 + 3 + 2  # joint_pos + joint_vel + linvel + angvel + grav + cmd

home_qpos = np.zeros(NQ)
home_qpos[2] = 0.35
home_qpos[3] = 1.0
joint_defaults = np.array([0, 0.8, -1.5] * 4)[:N_JOINTS]
home_qpos[7:7+N_JOINTS] = joint_defaults

HOME_QPOS = jnp.array(home_qpos)
JOINT_DEFAULTS = jnp.array(joint_defaults)

print(f"  nq={NQ}, nv={NV}, nu={N_JOINTS}, obs={OBS_DIM}D")


# ═══════════════════ ENV FUNCTIONS ═══════════════════

@jit
def get_obs(mx_data, vx_cmd):
    """Extract obs from batched MJX data. (N, obs_dim)"""
    qpos = mx_data.qpos
    qvel = mx_data.qvel
    joint_pos = qpos[:, 7:7+N_JOINTS]
    joint_vel = qvel[:, 6:6+N_JOINTS]
    base_linvel = qvel[:, :3]
    base_angvel = qvel[:, 3:6]
    w, x, y, z = qpos[:, 3], qpos[:, 4], qpos[:, 5], qpos[:, 6]
    gx = 2*(x*z - w*y)
    gy = 2*(y*z + w*x)
    gz = 1 - 2*(x*x + y*y)
    proj_grav = jnp.stack([gx, gy, gz], axis=-1)
    cmd = jnp.stack([vx_cmd, jnp.zeros_like(vx_cmd)], axis=-1)
    return jnp.concatenate([joint_pos, joint_vel, base_linvel, base_angvel, proj_grav, cmd], axis=-1)


@jit
def compute_reward(mx_data, vx_cmd):
    """Locomotion reward. Returns (N,) reward, (N,) done."""
    vx = mx_data.qvel[:, 0]
    height = mx_data.qpos[:, 2]

    # Velocity tracking (main reward)
    r_vel = 2.0 * jnp.exp(-4.0 * (vx - vx_cmd) ** 2)

    # Stay alive and upright
    r_height = jnp.exp(-50.0 * (height - 0.35) ** 2)
    r_upright = 0.5 * (1.0 - 2.0 * (mx_data.qpos[:, 4]**2 + mx_data.qpos[:, 5]**2))
    r_alive = 0.5

    # Penalties (kept small so walking reward dominates)
    r_energy = -0.00005 * jnp.sum(mx_data.ctrl ** 2, axis=-1)
    joint_vel = mx_data.qvel[:, 6:6+N_JOINTS]
    r_smooth = -0.0001 * jnp.sum(joint_vel ** 2, axis=-1)

    # Fall detection
    fallen = height < 0.18

    reward = (r_vel + r_height + r_upright + r_alive + r_energy + r_smooth) * (1.0 - fallen)
    return reward, fallen.astype(jnp.float32)


@jit
def apply_action(mx_data, action_offsets):
    """PD control from action offsets. Returns ctrl."""
    joint_pos = mx_data.qpos[:, 7:7+N_JOINTS]
    joint_vel = mx_data.qvel[:, 6:6+N_JOINTS]
    targets = JOINT_DEFAULTS[None, :] + action_offsets * 0.3
    torque = KP * (targets - joint_pos) - KD * joint_vel
    return jnp.clip(torque, -20.0, 20.0)


def step_env(mx_model, mx_data, ctrl, n_substeps):
    mx_data = mx_data.replace(ctrl=ctrl)
    def body(i, d):
        return jax.vmap(mjx.step, in_axes=(None, 0))(mx_model, d)
    return lax.fori_loop(0, n_substeps, body, mx_data)

step_env_jit = jit(step_env, static_argnums=(3,))


def reset_envs(mx_data, mask, home_qpos, key):
    """Reset flagged envs to standing with noise."""
    k1, k2 = random.split(key)
    N = mask.shape[0]
    new_qpos = jnp.broadcast_to(home_qpos, (N, NQ))
    noise_pos = random.normal(k1, (N, 3)) * 0.02
    noise_joint = random.normal(k2, (N, N_JOINTS)) * 0.05
    new_qpos = new_qpos.at[:, :3].add(noise_pos)
    new_qpos = new_qpos.at[:, 7:7+N_JOINTS].add(noise_joint)
    m = mask[:, None]
    qpos = jnp.where(m, new_qpos, mx_data.qpos)
    qvel = jnp.where(m, jnp.zeros_like(mx_data.qvel), mx_data.qvel)
    return mx_data.replace(qpos=qpos, qvel=qvel)

reset_envs_jit = jit(reset_envs)


# ═══════════════════ MLP POLICY + VALUE ═══════════════════

def init_mlp_params(key, obs_dim, act_dim, hidden=256):
    """Initialize a simple MLP policy and value function."""
    k1, k2, k3, k4, k5, k6 = random.split(key, 6)
    scale = 0.01

    params = {
        # Shared feature extractor
        'feat_w1': random.normal(k1, (obs_dim, hidden)) * np.sqrt(2.0 / obs_dim),
        'feat_b1': jnp.zeros(hidden),
        'feat_w2': random.normal(k2, (hidden, hidden)) * np.sqrt(2.0 / hidden),
        'feat_b2': jnp.zeros(hidden),

        # Policy head
        'pi_w': random.normal(k3, (hidden, act_dim)) * scale,
        'pi_b': jnp.zeros(act_dim),
        'log_std': jnp.full(act_dim, -0.5),  # Initial std ≈ 0.6

        # Value head
        'vf_w1': random.normal(k4, (hidden, hidden)) * np.sqrt(2.0 / hidden),
        'vf_b1': jnp.zeros(hidden),
        'vf_w': random.normal(k5, (hidden, 1)) * scale,
        'vf_b': jnp.zeros(1),
    }
    return params


@jit
def mlp_forward(params, obs):
    """Forward pass. Returns action_mean, log_std, value."""
    # Shared features
    x = jnp.tanh(obs @ params['feat_w1'] + params['feat_b1'])
    x = jnp.tanh(x @ params['feat_w2'] + params['feat_b2'])

    # Policy
    action_mean = jnp.tanh(x @ params['pi_w'] + params['pi_b'])
    log_std = params['log_std']

    # Value
    vf = jnp.tanh(x @ params['vf_w1'] + params['vf_b1'])
    value = (vf @ params['vf_w'] + params['vf_b']).squeeze(-1)

    return action_mean, log_std, value


@jit
def sample_action(params, obs, key):
    """Sample action from policy."""
    mean, log_std, value = mlp_forward(params, obs)
    std = jnp.exp(log_std)
    noise = random.normal(key, mean.shape)
    action = mean + std * noise
    action = jnp.clip(action, -1.0, 1.0)

    # Log probability
    log_prob = -0.5 * jnp.sum(((action - mean) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi), axis=-1)

    return action, log_prob, value


# ═══════════════════ PPO UPDATE ═══════════════════

@jit
def compute_gae(rewards, values, dones, last_value, gamma=GAMMA, lam=GAE_LAMBDA):
    """Compute GAE advantages and returns."""
    T = rewards.shape[0]
    values_ext = jnp.concatenate([values, last_value[None]], axis=0)

    def scan_fn(gae, t):
        t_rev = T - 1 - t
        delta = rewards[t_rev] + gamma * (1 - dones[t_rev]) * values_ext[t_rev + 1] - values_ext[t_rev]
        gae = delta + gamma * lam * (1 - dones[t_rev]) * gae
        return gae, gae

    _, advantages_rev = lax.scan(scan_fn, jnp.zeros(rewards.shape[1]), jnp.arange(T))
    advantages = advantages_rev[::-1]
    returns = advantages + values
    return advantages, returns


def ppo_update(params, obs_batch, act_batch, logp_batch, adv_batch, ret_batch, key):
    """Single PPO update step."""
    N = obs_batch.shape[0]
    batch_size = N // PPO_MINIBATCHES

    def loss_fn(params, obs, act, old_logp, adv, ret):
        mean, log_std, value = mlp_forward(params, obs)
        std = jnp.exp(log_std)

        # New log prob
        new_logp = -0.5 * jnp.sum(((act - mean) / (std + 1e-8)) ** 2 + 2 * log_std + np.log(2 * np.pi), axis=-1)

        # Policy loss (clipped)
        ratio = jnp.exp(new_logp - old_logp)
        adv_normalized = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg_loss1 = -adv_normalized * ratio
        pg_loss2 = -adv_normalized * jnp.clip(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS)
        pg_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

        # Value loss
        vf_loss = 0.5 * jnp.mean((value - ret) ** 2)

        # Entropy
        entropy = 0.5 * jnp.sum(log_std + 0.5 * np.log(2 * np.pi * np.e))

        total = pg_loss + VF_COEF * vf_loss - ENT_COEF * entropy
        return total, {'pg': pg_loss, 'vf': vf_loss, 'ent': entropy}

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)

    for epoch in range(PPO_EPOCHS):
        key, perm_key = random.split(key)
        perm = random.permutation(perm_key, N)
        for mb in range(PPO_MINIBATCHES):
            idx = perm[mb * batch_size:(mb + 1) * batch_size]
            (loss, metrics), grads = grad_fn(
                params, obs_batch[idx], act_batch[idx],
                logp_batch[idx], adv_batch[idx], ret_batch[idx]
            )
            # Simple SGD with gradient clipping
            grad_norm = jnp.sqrt(sum(jnp.sum(g**2) for g in jax.tree.leaves(grads)))
            clip_scale = jnp.minimum(1.0, MAX_GRAD_NORM / (grad_norm + 1e-8))
            grads = jax.tree.map(lambda g: g * clip_scale, grads)
            params = jax.tree.map(lambda p, g: p - LR * g, params, grads)

    return params, metrics


# ═══════════════════ INITIALIZE ═══════════════════

print("\n[INIT] Initializing...")

key = random.PRNGKey(42)
key, param_key, env_key = random.split(key, 3)

# MLP params
mlp_params = init_mlp_params(param_key, OBS_DIM, N_JOINTS)
n_params = sum(p.size for p in jax.tree.leaves(mlp_params))
print(f"  MLP policy params: {n_params:,} (throwaway — NOT SOMA)")

# Init environments
mj_data_cpu = mujoco.MjData(mj_model)
mj_data_cpu.qpos[:] = np.array(HOME_QPOS)
mujoco.mj_forward(mj_model, mj_data_cpu)
mx_data_single = mjx.put_data(mj_model, mj_data_cpu)

mx_data = jax.tree.map(lambda x: jnp.stack([x] * N_ENVS), mx_data_single)
keys = random.split(env_key, 3)
pos_noise = random.normal(keys[0], (N_ENVS, 3)) * 0.02
joint_noise = random.normal(keys[1], (N_ENVS, N_JOINTS)) * 0.05
vel_noise = random.normal(keys[2], (N_ENVS, NV)) * 0.1
new_qpos = mx_data.qpos.at[:, :3].add(pos_noise).at[:, 7:7+N_JOINTS].add(joint_noise)
mx_data = mx_data.replace(qpos=new_qpos, qvel=mx_data.qvel + vel_noise)
mx_data = jax.vmap(mjx.forward, in_axes=(None, 0))(mx_model, mx_data)

# Command: walk forward at 0.5 m/s
vx_cmd = jnp.ones(N_ENVS) * 0.5

print(f"  Envs: {N_ENVS} parallel")
print(f"  Command: vx = 0.5 m/s")

# Warmup JIT
print("\n[COMPILE] JIT compiling...")
t_comp = time.time()
obs = get_obs(mx_data, vx_cmd)
action, logp, value = sample_action(mlp_params, obs, random.PRNGKey(0))
ctrl = apply_action(mx_data, action)
mx_data_test = step_env_jit(mx_model, mx_data, ctrl, N_SUBSTEPS)
jax.block_until_ready(mx_data_test.qpos)
print(f"  ✓ Compiled in {time.time()-t_comp:.1f}s")


# ═══════════════════ PPO TRAINING ═══════════════════

print(f"\n[TRAIN] PPO Training — {TOTAL_UPDATES} updates")
print(f"  Steps per update: {N_ENVS} × {PPO_STEPS} = {N_ENVS * PPO_STEPS:,}")
print(f"  Total env steps: ~{TOTAL_UPDATES * N_ENVS * PPO_STEPS / 1e6:.0f}M")

print(f"\n  {'update':>6} | {'reward':>8} | {'vx':>8} | {'falls':>6} | {'pg_loss':>8} | {'time':>6}")
print(f"  {'-'*55}")

t_start = time.time()
best_reward = -999

for update in range(1, TOTAL_UPDATES + 1):
    key, rollout_key = random.split(key)

    # ── Collect rollout ──
    all_obs = []
    all_actions = []
    all_logprobs = []
    all_values = []
    all_rewards = []
    all_dones = []

    for step in range(PPO_STEPS):
        rollout_key, act_key, reset_key = random.split(rollout_key, 3)

        obs = get_obs(mx_data, vx_cmd)
        action, logp, value = sample_action(mlp_params, obs, act_key)

        # Step env
        ctrl = apply_action(mx_data, action)
        mx_data = step_env_jit(mx_model, mx_data, ctrl, N_SUBSTEPS)

        reward, fallen = compute_reward(mx_data, vx_cmd)

        all_obs.append(obs)
        all_actions.append(action)
        all_logprobs.append(logp)
        all_values.append(value)
        all_rewards.append(reward)
        all_dones.append(fallen)

        # Reset fallen envs
        if jnp.any(fallen > 0.5):
            mx_data = reset_envs_jit(mx_data, fallen, HOME_QPOS, reset_key)
            mx_data = jax.vmap(mjx.forward, in_axes=(None, 0))(mx_model, mx_data)

    # Stack rollout: (T, N_ENVS, ...)
    all_obs = jnp.stack(all_obs)
    all_actions = jnp.stack(all_actions)
    all_logprobs = jnp.stack(all_logprobs)
    all_values = jnp.stack(all_values)
    all_rewards = jnp.stack(all_rewards)
    all_dones = jnp.stack(all_dones)

    # Bootstrap value
    last_obs = get_obs(mx_data, vx_cmd)
    _, _, last_value = mlp_forward(mlp_params, last_obs)

    # GAE
    advantages, returns = compute_gae(all_rewards, all_values, all_dones, last_value)

    # Flatten (T, N) → (T*N,)
    T, N = all_obs.shape[:2]
    flat_obs = all_obs.reshape(T * N, -1)
    flat_act = all_actions.reshape(T * N, -1)
    flat_logp = all_logprobs.reshape(T * N)
    flat_adv = advantages.reshape(T * N)
    flat_ret = returns.reshape(T * N)

    # PPO update
    key, ppo_key = random.split(key)
    mlp_params, metrics = ppo_update(
        mlp_params, flat_obs, flat_act, flat_logp, flat_adv, flat_ret, ppo_key
    )

    # ── Logging ──
    mean_reward = float(jnp.mean(all_rewards))
    mean_vx = float(jnp.mean(mx_data.qvel[:, 0]))
    fall_rate = float(jnp.mean(all_dones)) * 100

    if mean_reward > best_reward:
        best_reward = mean_reward
        best_params = jax.tree.map(lambda x: x.copy(), mlp_params)

    if update <= 5 or update % 10 == 0 or update == TOTAL_UPDATES:
        elapsed = time.time() - t_start
        print(f"  {update:6d} | {mean_reward:8.3f} | {mean_vx:8.3f} | {fall_rate:5.1f}% | "
              f"{float(metrics['pg']):8.4f} | {elapsed:5.0f}s")

    # Early stopping: if walking well
    if mean_reward > 3.0 and mean_vx > 0.3 and update > 50:
        print(f"\n  ✓ Robot is walking! (reward={mean_reward:.2f}, vx={mean_vx:.2f})")
        print(f"  Stopping PPO early at update {update}")
        break

train_time = time.time() - t_start
print(f"\n  Training time: {train_time/60:.1f} min")
print(f"  Best reward: {best_reward:.3f}")

# Use best params
mlp_params = best_params


# ═══════════════════ VALIDATE WALKER ═══════════════════

print(f"\n[VALIDATE] Testing trained walker...")

# Reset all envs
key, val_key = random.split(key)
mx_data = jax.tree.map(lambda x: jnp.stack([x] * N_ENVS), mx_data_single)
mx_data = mx_data.replace(qpos=mx_data.qpos.at[:, :].set(HOME_QPOS[None, :]))
mx_data = jax.vmap(mjx.forward, in_axes=(None, 0))(mx_model, mx_data)

vx_samples = []
for step in range(200):
    val_key, ak = random.split(val_key)
    obs = get_obs(mx_data, vx_cmd)
    action, _, _ = sample_action(mlp_params, obs, ak)
    ctrl = apply_action(mx_data, action)
    mx_data = step_env_jit(mx_model, mx_data, ctrl, N_SUBSTEPS)
    vx_samples.append(float(jnp.mean(mx_data.qvel[:, 0])))

avg_vx = np.mean(vx_samples[50:])  # Skip first 50 steps (startup)
print(f"  Average forward velocity: {avg_vx:.3f} m/s (target: 0.5)")
print(f"  Final height: {float(jnp.mean(mx_data.qpos[:, 2])):.3f} m")

if avg_vx > 0.2:
    print(f"  ✓ Walker is moving forward!")
elif avg_vx > 0.05:
    print(f"  ⚠ Walker is shuffling. Might need more training.")
else:
    print(f"  ✗ Walker isn't walking. May need reward tuning or more updates.")
    print(f"    Increase TOTAL_UPDATES to 500-1000 and re-run.")


# ═══════════════════ COLLECT WALKING DATA ═══════════════════

print(f"\n[COLLECT] Gathering {COLLECT_TRANSITIONS/1e6:.0f}M transitions from trained walker...")

# Reset envs
key, collect_key = random.split(key)
mx_data = jax.tree.map(lambda x: jnp.stack([x] * N_ENVS), mx_data_single)
noise_keys = random.split(collect_key, 3)
new_qpos = mx_data.qpos.at[:, :].set(HOME_QPOS[None, :])
new_qpos = new_qpos.at[:, :3].add(random.normal(noise_keys[0], (N_ENVS, 3)) * 0.02)
new_qpos = new_qpos.at[:, 7:7+N_JOINTS].add(random.normal(noise_keys[1], (N_ENVS, N_JOINTS)) * 0.05)
mx_data = mx_data.replace(qpos=new_qpos, qvel=mx_data.qvel + random.normal(noise_keys[2], (N_ENVS, NV)) * 0.1)
mx_data = jax.vmap(mjx.forward, in_axes=(None, 0))(mx_model, mx_data)

# Vary commands for diversity
key, cmd_key = random.split(key)
vx_cmds = random.uniform(cmd_key, (N_ENVS,), minval=0.0, maxval=1.0)  # Forward only

max_batches = COLLECT_TRANSITIONS // N_ENVS + 100
obs_buf = np.zeros((max_batches, N_ENVS, OBS_DIM), dtype=np.float32)
act_buf = np.zeros((max_batches, N_ENVS, N_JOINTS), dtype=np.float32)
nobs_buf = np.zeros((max_batches, N_ENVS, OBS_DIM), dtype=np.float32)
rew_buf = np.zeros((max_batches, N_ENVS), dtype=np.float32)
done_buf = np.zeros((max_batches, N_ENVS), dtype=np.float32)
cmd_buf = np.zeros((max_batches, N_ENVS, 2), dtype=np.float32)

total = 0
batch_idx = 0
t_collect = time.time()

while total < COLLECT_TRANSITIONS:
    key, act_key, reset_key, newcmd_key = random.split(key, 4)

    obs = get_obs(mx_data, vx_cmds)
    action, _, _ = sample_action(mlp_params, obs, act_key)

    # Add exploration noise (10% of action std) for diversity
    action = action + random.normal(act_key, action.shape) * 0.1
    action = jnp.clip(action, -1.0, 1.0)

    ctrl = apply_action(mx_data, action)
    mx_data = step_env_jit(mx_model, mx_data, ctrl, N_SUBSTEPS)

    next_obs = get_obs(mx_data, vx_cmds)
    reward, fallen = compute_reward(mx_data, vx_cmds)

    if batch_idx < max_batches:
        obs_buf[batch_idx] = np.array(obs)
        act_buf[batch_idx] = np.array(action)
        nobs_buf[batch_idx] = np.array(next_obs)
        rew_buf[batch_idx] = np.array(reward)
        done_buf[batch_idx] = np.array(fallen)
        cmd_buf[batch_idx] = np.stack([np.array(vx_cmds), np.zeros(N_ENVS)], axis=-1)

    # Reset fallen
    if jnp.any(fallen > 0.5):
        mx_data = reset_envs_jit(mx_data, fallen, HOME_QPOS, reset_key)
        mx_data = jax.vmap(mjx.forward, in_axes=(None, 0))(mx_model, mx_data)
        new_cmds = random.uniform(newcmd_key, (N_ENVS,), minval=0.0, maxval=1.0)
        vx_cmds = jnp.where(fallen > 0.5, new_cmds, vx_cmds)

    # Vary commands periodically
    if batch_idx % 200 == 0 and batch_idx > 0:
        key, ck = random.split(key)
        vx_cmds = random.uniform(ck, (N_ENVS,), minval=0.0, maxval=1.0)

    total += N_ENVS
    batch_idx += 1

    if batch_idx % 100 == 0:
        pct = total / COLLECT_TRANSITIONS * 100
        print(f"  {total/1e6:.2f}M / {COLLECT_TRANSITIONS/1e6:.0f}M ({pct:.0f}%)")

collect_time = time.time() - t_collect

# Reshape
n_used = min(batch_idx, max_batches)
n_total = n_used * N_ENVS

walking_data = {
    'obs': obs_buf[:n_used].reshape(n_total, OBS_DIM)[:COLLECT_TRANSITIONS],
    'actions': act_buf[:n_used].reshape(n_total, N_JOINTS)[:COLLECT_TRANSITIONS],
    'next_obs': nobs_buf[:n_used].reshape(n_total, OBS_DIM)[:COLLECT_TRANSITIONS],
    'rewards': rew_buf[:n_used].reshape(n_total)[:COLLECT_TRANSITIONS],
    'dones': done_buf[:n_used].reshape(n_total)[:COLLECT_TRANSITIONS],
    'commands': cmd_buf[:n_used].reshape(n_total, 2)[:COLLECT_TRANSITIONS],
}

# Clean NaN
mask = np.all(np.isfinite(walking_data['obs']), 1) & np.all(np.isfinite(walking_data['next_obs']), 1)
if not mask.all():
    walking_data = {k: v[mask] for k, v in walking_data.items()}

# Save walking data
np.savez_compressed(SAVE_PATH, **walking_data)
walk_mb = os.path.getsize(SAVE_PATH) / 1024**2

# Quality check
walk_vx = walking_data['obs'][:, 2 * N_JOINTS]
print(f"\n  --- Walking Data Quality ---")
print(f"  Transitions: {walking_data['obs'].shape[0]:,}")
print(f"  Base vx: mean={walk_vx.mean():.3f}, std={walk_vx.std():.3f}")
print(f"  Walking (vx>0.3): {(walk_vx>0.3).mean()*100:.1f}%")
print(f"  Reward: mean={walking_data['rewards'].mean():.3f}")
print(f"  Falls: {walking_data['dones'].mean()*100:.1f}%")


# ═══════════════════ MERGE WITH EXISTING DATA ═══════════════════

print(f"\n[MERGE] Combining walking + standing data...")

if os.path.exists("go2_physics_data.npz"):
    standing_data = dict(np.load("go2_physics_data.npz"))
    print(f"  Standing data: {standing_data['obs'].shape[0]:,} transitions")
    print(f"  Walking data:  {walking_data['obs'].shape[0]:,} transitions")

    merged = {
        k: np.concatenate([standing_data[k], walking_data[k]], axis=0)
        for k in walking_data.keys()
    }

    # Shuffle
    rng = np.random.RandomState(42)
    perm = rng.permutation(len(merged['obs']))
    merged = {k: v[perm] for k, v in merged.items()}

    np.savez_compressed(MERGED_PATH, **merged)
    merged_mb = os.path.getsize(MERGED_PATH) / 1024**2

    total_vx = merged['obs'][:, 2 * N_JOINTS]
    print(f"\n  --- Merged Data Quality ---")
    print(f"  Total: {merged['obs'].shape[0]:,} transitions")
    print(f"  Base vx: mean={total_vx.mean():.3f}")
    print(f"  Walking (vx>0.3): {(total_vx>0.3).mean()*100:.1f}%")
    print(f"  Reward: mean={merged['rewards'].mean():.3f}")
    print(f"  File: {MERGED_PATH} ({merged_mb:.0f} MB)")
else:
    print(f"  ⚠ No standing data found. Using walking data only.")
    MERGED_PATH = SAVE_PATH


# ═══════════════════ SUMMARY ═══════════════════

print(f"\n{'='*60}")
print(f"  ✓ STEP 1.5 COMPLETE")
print(f"{'='*60}")
print(f"  PPO training: {train_time/60:.1f} min")
print(f"  Best reward: {best_reward:.3f}")
print(f"  Walker vx: {avg_vx:.3f} m/s")
print(f"  Walking data: {SAVE_PATH} ({walk_mb:.0f} MB)")
print(f"  Merged data: {MERGED_PATH}")
print(f"{'='*60}")
print(f"""
  PASS/FAIL:
    Walker moves forward:  {'✓' if avg_vx > 0.1 else '✗'}
    Walking % > 20:        {'✓' if (walk_vx>0.3).mean() > 0.2 else '✗'}
    Reward > 0:            {'✓' if walking_data['rewards'].mean() > 0 else '✗'}

  Next: Retrain world model on merged data
    1. Change data_path in Step 2: "{MERGED_PATH}"
    2. Re-run soma_v12_step2_world_model.py
""")
