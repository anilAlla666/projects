"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║     HYPERFLUX HIGHGUARD TIER 1 KERNELS - MASTER TRAINING NOTEBOOK                    ║
║                                                                                      ║
║     This notebook trains all Tier 1 Performance Kernels:                             ║
║       1. Audio Occlusion v1.0 - O(1) sound propagation                               ║
║       2. LOD Selection v1.0 - O(1) batch LOD prediction                              ║
║       3. Occlusion Culling v1.0 - O(1) visibility without GPU stalls                 ║
║       4. Lumen GI Approximation v1.0 - O(1) indirect lighting                        ║
║                                                                                      ║
║     FEATURES:                                                                        ║
║       - Google Drive mounting & folder organization                                  ║
║       - Auto-versioning on training failure                                          ║
║       - Checkpoint saving after each epoch                                           ║
║       - Resume from checkpoint support                                               ║
║       - SDK generation for each kernel                                               ║
║                                                                                      ║
║     Author: Anil Kumar Alla | HyperFlux Neural Dynamics                              ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

INSTRUCTIONS:
1. Run this notebook in Google Colab with GPU runtime
2. It will mount your Google Drive
3. All files will be saved to: /content/drive/MyDrive/Highguard/Tier1/
4. If training fails, run again - it auto-increments version numbers
"""

# ═══════════════════════════════════════════════════════════════════════════════════════
# CELL 1: SETUP & DRIVE MOUNTING
# ═══════════════════════════════════════════════════════════════════════════════════════

import os
import sys
import json
import shutil
from datetime import datetime

# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Create folder structure
DRIVE_BASE = "/content/drive/MyDrive/Highguard"
TIER1_DIR = f"{DRIVE_BASE}/Tier1"
CHECKPOINTS_DIR = f"{TIER1_DIR}/checkpoints"
SDKS_DIR = f"{TIER1_DIR}/SDKs"
LOGS_DIR = f"{TIER1_DIR}/logs"
MODELS_DIR = f"{TIER1_DIR}/models"

for d in [DRIVE_BASE, TIER1_DIR, CHECKPOINTS_DIR, SDKS_DIR, LOGS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)
    print(f"✓ {d}")

# Version tracking file
VERSION_FILE = f"{TIER1_DIR}/versions.json"

def load_versions():
    if os.path.exists(VERSION_FILE):
        with open(VERSION_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_versions(versions):
    with open(VERSION_FILE, 'w') as f:
        json.dump(versions, f, indent=2)

def get_next_version(kernel_name):
    """Get next version number for a kernel (auto-increment on failure)"""
    versions = load_versions()
    if kernel_name not in versions:
        versions[kernel_name] = {"current": 1, "history": []}
    else:
        versions[kernel_name]["current"] += 1
    save_versions(versions)
    return versions[kernel_name]["current"]

def mark_success(kernel_name, version, metrics):
    """Mark a kernel version as successful"""
    versions = load_versions()
    versions[kernel_name]["history"].append({
        "version": version,
        "status": "success",
        "metrics": metrics,
        "timestamp": datetime.now().isoformat()
    })
    save_versions(versions)

def mark_failure(kernel_name, version, error):
    """Mark a kernel version as failed"""
    versions = load_versions()
    versions[kernel_name]["history"].append({
        "version": version,
        "status": "failed",
        "error": str(error),
        "timestamp": datetime.now().isoformat()
    })
    save_versions(versions)

print(f"\n{'═'*70}")
print(f" HYPERFLUX HIGHGUARD TIER 1 TRAINING")
print(f" Drive mounted at: {DRIVE_BASE}")
print(f" Tier 1 folder: {TIER1_DIR}")
print(f"{'═'*70}\n")

# Show current versions
versions = load_versions()
print("Current kernel versions:")
for k, v in versions.items():
    print(f"  {k}: v{v['current']}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# CELL 2: IMPORTS & DEVICE SETUP
# ═══════════════════════════════════════════════════════════════════════════════════════

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ═══════════════════════════════════════════════════════════════════════════════════════
# CELL 3: COMMON UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
    """Residual block with LayerNorm and GELU"""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
    
    def forward(self, x):
        return F.gelu(x + self.net(x))


def save_checkpoint(model, optimizer, epoch, loss, kernel_name, version):
    """Save training checkpoint to Drive"""
    checkpoint_path = f"{CHECKPOINTS_DIR}/{kernel_name}_v{version}_epoch{epoch}.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, checkpoint_path)
    print(f"   💾 Checkpoint saved: {checkpoint_path}")


def load_checkpoint(kernel_name, version):
    """Load latest checkpoint for a kernel"""
    import glob
    pattern = f"{CHECKPOINTS_DIR}/{kernel_name}_v{version}_epoch*.pt"
    checkpoints = glob.glob(pattern)
    if not checkpoints:
        return None
    latest = max(checkpoints, key=lambda x: int(x.split('epoch')[-1].split('.')[0]))
    return torch.load(latest)


def generate_sdk_header(kernel_name, version, metrics, structs, functions):
    """Generate C header file for SDK"""
    header = f'''/*
 * HYPERFLUX {kernel_name.upper()} v{version} - HIGHGUARD
 * Auto-generated SDK
 * 
 * METRICS:
'''
    for k, v in metrics.items():
        if isinstance(v, float):
            header += f' *   {k}: {v*100:.2f}%\n'
        else:
            header += f' *   {k}: {v}\n'
    
    header += f''' *
 * Generated: {datetime.now().isoformat()}
 */
#ifndef HG_{kernel_name.upper()}_H
#define HG_{kernel_name.upper()}_H

{structs}

{functions}

#endif
'''
    return header


# ═══════════════════════════════════════════════════════════════════════════════════════
# CELL 4: KERNEL 1 - AUDIO OCCLUSION
# ═══════════════════════════════════════════════════════════════════════════════════════

def train_audio_occlusion(num_samples=2_000_000, epochs=100, batch_size=8192, patience=15):
    """
    Train Audio Occlusion Kernel
    Replaces O(N × raycasts) with O(1) neural inference
    """
    KERNEL_NAME = "AudioOcclusion"
    VERSION = get_next_version(KERNEL_NAME)
    
    print(f"\n{'█'*70}")
    print(f"█ TRAINING {KERNEL_NAME} v{VERSION} ".ljust(69) + "█")
    print(f"{'█'*70}")
    
    try:
        # ═══════════════════════════════════════════════════════════════════════
        # CONFIGURATION
        # ═══════════════════════════════════════════════════════════════════════
        
        NUM_SOUND_TYPES = 8
        SOUND_TYPES = {
            0: 'gunfire', 1: 'explosion', 2: 'footstep', 3: 'mount',
            4: 'ability', 5: 'destruction', 6: 'ambient', 7: 'voice',
        }
        
        NUM_ENV_TYPES = 6
        ENV_TYPES = {
            0: 'outdoor_open', 1: 'outdoor_terrain', 2: 'base_exterior',
            3: 'base_interior', 4: 'siege_tower', 5: 'cave_mine',
        }
        
        NUM_WALL_CONFIGS = 5
        WALL_CONFIGS = {
            0: 'none', 1: 'thin_wood', 2: 'thick_stone', 3: 'reinforced', 4: 'partial',
        }
        
        # Sound properties [base_freq_hz, freq_spread, directionality, decay_rate]
        SOUND_PROPERTIES = torch.tensor([
            [2000, 4000, 0.8, 0.3], [100, 2000, 0.2, 0.5], [200, 1000, 0.5, 0.8],
            [150, 800, 0.3, 0.7], [500, 3000, 0.6, 0.4], [80, 1500, 0.1, 0.6],
            [100, 500, 0.1, 0.9], [300, 3000, 0.9, 0.2],
        ], dtype=torch.float32, device=device)
        
        # Environment properties [reverb_base, absorption, diffusion, rt60]
        ENV_PROPERTIES = torch.tensor([
            [0.05, 0.1, 0.2, 0.3], [0.10, 0.2, 0.4, 0.5], [0.25, 0.4, 0.5, 0.8],
            [0.50, 0.6, 0.7, 1.2], [0.60, 0.7, 0.6, 1.0], [0.70, 0.5, 0.8, 1.5],
        ], dtype=torch.float32, device=device)
        
        # Wall absorption [low_abs, mid_abs, high_abs, transmission_loss_db]
        WALL_ABSORPTION = torch.tensor([
            [0.0, 0.0, 0.0, 0], [0.1, 0.2, 0.4, 10], [0.3, 0.5, 0.8, 25],
            [0.4, 0.6, 0.9, 35], [0.2, 0.3, 0.5, 15],
        ], dtype=torch.float32, device=device)
        
        # ═══════════════════════════════════════════════════════════════════════
        # GROUND TRUTH COMPUTATION
        # ═══════════════════════════════════════════════════════════════════════
        
        def compute_audio_occlusion(listener_pos, source_pos, sound_type, env_type,
                                     wall_config, wall_distance, wall_thickness):
            B = len(listener_pos)
            
            diff = source_pos - listener_pos
            distance = torch.norm(diff, dim=1).clamp(min=0.1)
            
            # Distance attenuation
            ref_distance, max_distance, rolloff = 1.0, 100.0, 1.0
            distance_atten = ref_distance / (ref_distance + rolloff * (distance - ref_distance).clamp(min=0))
            distance_atten = distance_atten.clamp(0, 1)
            
            sound_props = SOUND_PROPERTIES[sound_type]
            env_props = ENV_PROPERTIES[env_type]
            wall_abs = WALL_ABSORPTION[wall_config]
            
            # Attenuation
            wall_atten = torch.pow(10, -wall_abs[:, 3] / 20)
            thickness_factor = torch.exp(-wall_thickness * 0.5)
            wall_atten = wall_atten * thickness_factor + (1 - thickness_factor) * 0.1
            wall_proximity = torch.exp(-wall_distance * 0.1)
            effective_wall_atten = wall_atten * wall_proximity + (1 - wall_proximity) * 1.0
            attenuation = (distance_atten * effective_wall_atten).clamp(0.01, 1.0)
            
            # Lowpass cutoff
            max_cutoff, min_cutoff = 20000.0, 200.0
            wall_cutoff_factor = 1.0 - wall_abs[:, 2] * wall_proximity
            air_absorption = torch.exp(-distance * 0.01)
            cutoff_factor = wall_cutoff_factor * air_absorption
            lowpass_cutoff = (min_cutoff + (max_cutoff - min_cutoff) * cutoff_factor).clamp(min_cutoff, max_cutoff)
            
            # Reverb
            reverb = env_props[:, 0] + (distance / max_distance).clamp(0, 1) * 0.3
            reverb = (reverb * (0.7 + 0.3 * (1.0 - sound_props[:, 2]))).clamp(0.0, 0.95)
            
            # Delay
            speed_of_sound = 343.0
            delay_ms = ((distance / speed_of_sound) * 1000 + 
                       wall_proximity * wall_thickness * 3.0 + 
                       env_props[:, 3] * 5.0).clamp(0, 500)
            
            return attenuation, lowpass_cutoff, reverb, delay_ms
        
        # ═══════════════════════════════════════════════════════════════════════
        # DATA GENERATION
        # ═══════════════════════════════════════════════════════════════════════
        
        print(f"\n   Generating {num_samples:,} samples...")
        start = time.time()
        B = num_samples
        
        # Listener positions
        listener_pos = torch.zeros(B, 3, device=device)
        listener_pos[:, 0] = torch.rand(B, device=device) * 200 - 100
        listener_pos[:, 1] = torch.rand(B, device=device) * 200 - 100
        listener_pos[:, 2] = torch.rand(B, device=device) * 3 + 1
        
        # Source positions (varying distances)
        distance_type = torch.rand(B, device=device)
        source_distance = torch.zeros(B, device=device)
        source_distance[distance_type < 0.3] = torch.rand((distance_type < 0.3).sum(), device=device) * 10
        source_distance[(distance_type >= 0.3) & (distance_type < 0.7)] = torch.rand(((distance_type >= 0.3) & (distance_type < 0.7)).sum(), device=device) * 40 + 10
        source_distance[distance_type >= 0.7] = torch.rand((distance_type >= 0.7).sum(), device=device) * 50 + 50
        
        theta = torch.rand(B, device=device) * 2 * np.pi
        phi = torch.rand(B, device=device) * np.pi - np.pi / 2
        
        source_pos = torch.zeros(B, 3, device=device)
        source_pos[:, 0] = listener_pos[:, 0] + source_distance * torch.cos(phi) * torch.cos(theta)
        source_pos[:, 1] = listener_pos[:, 1] + source_distance * torch.cos(phi) * torch.sin(theta)
        source_pos[:, 2] = (torch.rand(B, device=device) * 5 + 0.5).clamp(0.5, 10)
        
        # Sound types (weighted)
        sound_weights = torch.tensor([0.25, 0.20, 0.10, 0.08, 0.15, 0.10, 0.05, 0.07], device=device)
        sound_type = torch.multinomial(sound_weights.expand(B, -1), 1).squeeze()
        
        # Environment types
        env_weights = torch.tensor([0.15, 0.15, 0.25, 0.25, 0.10, 0.10], device=device)
        env_type = torch.multinomial(env_weights.expand(B, -1), 1).squeeze()
        
        # Wall configuration
        wall_prob = torch.zeros(B, device=device)
        wall_prob[env_type == 0] = 0.1
        wall_prob[env_type == 1] = 0.2
        wall_prob[env_type == 2] = 0.5
        wall_prob[env_type == 3] = 0.7
        wall_prob[env_type == 4] = 0.6
        wall_prob[env_type == 5] = 0.4
        
        has_wall = torch.rand(B, device=device) < wall_prob
        wall_config = torch.zeros(B, dtype=torch.long, device=device)
        wall_config[has_wall] = torch.randint(1, NUM_WALL_CONFIGS, (has_wall.sum(),), device=device)
        
        wall_distance = torch.zeros(B, device=device)
        wall_distance[has_wall] = torch.rand(has_wall.sum(), device=device) * source_distance[has_wall] * 0.5
        wall_distance[~has_wall] = 1000
        
        wall_thickness = torch.zeros(B, device=device)
        wall_thickness[wall_config == 1] = torch.rand((wall_config == 1).sum(), device=device) * 0.2 + 0.1
        wall_thickness[wall_config == 2] = torch.rand((wall_config == 2).sum(), device=device) * 0.5 + 0.3
        wall_thickness[wall_config == 3] = torch.rand((wall_config == 3).sum(), device=device) * 0.3 + 0.5
        wall_thickness[wall_config == 4] = torch.rand((wall_config == 4).sum(), device=device) * 0.4 + 0.1
        
        # Compute ground truth
        attenuation, lowpass_cutoff, reverb, delay_ms = compute_audio_occlusion(
            listener_pos, source_pos, sound_type, env_type,
            wall_config, wall_distance, wall_thickness
        )
        
        # Build features
        rel_source = source_pos - listener_pos
        sound_oh = F.one_hot(sound_type, NUM_SOUND_TYPES).float()
        env_oh = F.one_hot(env_type, NUM_ENV_TYPES).float()
        wall_oh = F.one_hot(wall_config, NUM_WALL_CONFIGS).float()
        
        X = torch.cat([
            rel_source / 100.0, source_distance.unsqueeze(1) / 100.0,
            listener_pos[:, 2:3] / 5.0, sound_oh, env_oh, wall_oh,
            wall_distance.unsqueeze(1) / 100.0, wall_thickness.unsqueeze(1) / 1.0,
        ], dim=1)
        
        Y = torch.stack([
            attenuation, lowpass_cutoff / 20000.0, reverb, delay_ms / 500.0
        ], dim=1)
        
        idx = torch.randperm(B, device=device)
        X, Y = X[idx], Y[idx]
        
        print(f"   ✓ Generated in {time.time()-start:.1f}s | Features: {X.shape[1]}")
        
        # ═══════════════════════════════════════════════════════════════════════
        # MODEL
        # ═══════════════════════════════════════════════════════════════════════
        
        class AudioOcclusionNet(nn.Module):
            def __init__(self, input_dim=26, hidden=256, blocks=6):
                super().__init__()
                self.proj = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
                self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(blocks)])
                self.heads = nn.ModuleList([
                    nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())
                    for _ in range(4)
                ])
                self.n_params = sum(p.numel() for p in self.parameters())
            
            def forward(self, x):
                x = self.proj(x)
                for block in self.blocks:
                    x = block(x)
                return torch.cat([head(x) for head in self.heads], dim=1)
        
        # ═══════════════════════════════════════════════════════════════════════
        # TRAINING
        # ═══════════════════════════════════════════════════════════════════════
        
        n_val = num_samples // 10
        X_train, X_val = X[n_val:], X[:n_val]
        Y_train, Y_val = Y[n_val:], Y[:n_val]
        
        model = AudioOcclusionNet(input_dim=X.shape[1]).to(device)
        print(f"   Model: {model.n_params:,} parameters")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, epochs=epochs,
            steps_per_epoch=(len(X_train) // batch_size) + 1
        )
        
        output_weights = torch.tensor([2.0, 1.0, 1.0, 0.5], device=device)
        
        print(f"\n   {'Ep':>3} | {'Train':>10} | {'Val':>10} | {'Atten':>8} | {'Best':>10}")
        print("   " + "-" * 55)
        
        best_loss = float('inf')
        best_state = None
        no_improve = 0
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            n_batches = 0
            
            idx = torch.randperm(len(X_train), device=device)
            for i in range(0, len(X_train), batch_size):
                batch_idx = idx[i:i+batch_size]
                xb, yb = X_train[batch_idx], Y_train[batch_idx]
                
                optimizer.zero_grad()
                pred = model(xb)
                loss = ((pred - yb).pow(2) * output_weights).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            train_loss /= n_batches
            
            model.eval()
            with torch.no_grad():
                pred = model(X_val)
                val_loss = ((pred - Y_val).pow(2) * output_weights).mean().item()
                atten_acc = ((pred[:, 0] - Y_val[:, 0]).abs() < 0.05).float().mean().item()
            
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            if epoch % 5 == 0 or is_best:
                print(f"   {epoch+1:>3} | {train_loss:>10.6f} | {val_loss:>10.6f} | "
                      f"{atten_acc*100:>7.2f}% | {best_loss:>10.6f}{'*' if is_best else ''}")
            
            # Save checkpoint every 10 epochs
            if epoch % 10 == 0:
                save_checkpoint(model, optimizer, epoch, val_loss, KERNEL_NAME, VERSION)
            
            if no_improve >= patience:
                print(f"\n   Early stop at epoch {epoch+1}")
                break
        
        model.load_state_dict(best_state)
        
        # Final metrics
        model.eval()
        with torch.no_grad():
            pred = model(X_val)
            atten_acc = ((pred[:, 0] - Y_val[:, 0]).abs() < 0.05).float().mean().item()
            lowpass_acc = ((pred[:, 1] - Y_val[:, 1]).abs() * 20000 < 1000).float().mean().item()
            reverb_acc = ((pred[:, 2] - Y_val[:, 2]).abs() < 0.05).float().mean().item()
            delay_acc = ((pred[:, 3] - Y_val[:, 3]).abs() * 500 < 10).float().mean().item()
        
        metrics = {
            'atten_acc': atten_acc,
            'lowpass_acc': lowpass_acc,
            'reverb_acc': reverb_acc,
            'delay_acc': delay_acc,
        }
        
        print(f"\n   ═══════════════════════════════════════════════════")
        print(f"   FINAL RESULTS:")
        print(f"   Attenuation: {atten_acc*100:.2f}% | Lowpass: {lowpass_acc*100:.2f}%")
        print(f"   Reverb: {reverb_acc*100:.2f}% | Delay: {delay_acc*100:.2f}%")
        print(f"   ═══════════════════════════════════════════════════")
        
        # Save model to Drive
        model_path = f"{MODELS_DIR}/{KERNEL_NAME}_v{VERSION}.pt"
        torch.save({'model': model.state_dict(), 'metrics': metrics}, model_path)
        print(f"   ✓ Model saved: {model_path}")
        
        # Mark success
        mark_success(KERNEL_NAME, VERSION, metrics)
        
        return model, metrics, VERSION
        
    except Exception as e:
        mark_failure(KERNEL_NAME, VERSION, e)
        print(f"\n   ❌ TRAINING FAILED: {e}")
        print(f"   Run again to train v{VERSION + 1}")
        raise


# ═══════════════════════════════════════════════════════════════════════════════════════
# CELL 5: KERNEL 2 - LOD SELECTION
# ═══════════════════════════════════════════════════════════════════════════════════════

def train_lod_selection(num_samples=2_000_000, epochs=100, batch_size=8192, patience=15):
    """
    Train LOD Selection Kernel
    Replaces O(N) per-object calculations with O(1) batch inference
    """
    KERNEL_NAME = "LODSelection"
    VERSION = get_next_version(KERNEL_NAME)
    
    print(f"\n{'█'*70}")
    print(f"█ TRAINING {KERNEL_NAME} v{VERSION} ".ljust(69) + "█")
    print(f"{'█'*70}")
    
    try:
        NUM_LOD_LEVELS = 5
        NUM_IMPORTANCE_LEVELS = 5
        
        LOD_THRESHOLDS = torch.tensor([
            [0, 15, 30, 60, 100], [0, 10, 25, 50, 80], [0, 8, 20, 40, 70],
            [0, 5, 15, 30, 50], [0, 3, 10, 20, 35],
        ], dtype=torch.float32, device=device)
        
        SCREEN_SIZE_THRESHOLDS = torch.tensor([
            [0.20, 0.10, 0.05, 0.02, 0.005], [0.15, 0.08, 0.04, 0.015, 0.004],
            [0.10, 0.05, 0.025, 0.01, 0.003], [0.08, 0.04, 0.02, 0.008, 0.002],
            [0.05, 0.025, 0.01, 0.005, 0.001],
        ], dtype=torch.float32, device=device)
        
        CULL_DISTANCES = torch.tensor([500, 300, 200, 150, 80], dtype=torch.float32, device=device)
        
        def compute_lod_selection(camera_pos, camera_forward, camera_fov, camera_velocity,
                                   object_pos, object_radius, object_importance):
            B = len(object_pos)
            
            if camera_pos.dim() == 1:
                camera_pos = camera_pos.unsqueeze(0).expand(B, -1)
            if camera_forward.dim() == 1:
                camera_forward = camera_forward.unsqueeze(0).expand(B, -1)
            if camera_velocity.dim() == 1:
                camera_velocity = camera_velocity.unsqueeze(0).expand(B, -1)
            if isinstance(camera_fov, (int, float)):
                camera_fov = torch.full((B,), camera_fov, device=device)
            
            to_object = object_pos - camera_pos
            distance = torch.norm(to_object, dim=1).clamp(min=0.1)
            
            camera_forward = F.normalize(camera_forward, dim=1)
            to_object_norm = F.normalize(to_object, dim=1)
            dot = (camera_forward * to_object_norm).sum(dim=1)
            in_front = dot > 0
            
            fov_rad = camera_fov * np.pi / 180
            tan_half_fov = torch.tan(fov_rad / 2)
            object_diameter = object_radius * 2
            screen_size_ratio = ((object_diameter / distance) / (2 * tan_half_fov)).clamp(0, 1)
            
            approach_rate = -(camera_velocity * to_object_norm).sum(dim=1)
            future_distance = (distance - approach_rate * 0.5).clamp(min=0.1)
            future_screen_ratio = ((object_diameter / future_distance) / (2 * tan_half_fov)).clamp(0, 1)
            effective_screen_ratio = torch.max(screen_size_ratio, future_screen_ratio)
            
            dist_thresholds = LOD_THRESHOLDS[object_importance]
            size_thresholds = SCREEN_SIZE_THRESHOLDS[object_importance]
            cull_dist = CULL_DISTANCES[object_importance]
            
            size_scale = (object_radius / 1.0).clamp(0.5, 5.0)
            scaled_dist_thresholds = dist_thresholds * size_scale.unsqueeze(1)
            
            lod_from_distance = torch.zeros(B, dtype=torch.long, device=device)
            for lod in range(NUM_LOD_LEVELS - 1, -1, -1):
                mask = distance < scaled_dist_thresholds[:, lod]
                lod_from_distance[mask] = lod
            
            lod_from_screen = torch.full((B,), NUM_LOD_LEVELS - 1, dtype=torch.long, device=device)
            for lod in range(NUM_LOD_LEVELS - 1, -1, -1):
                mask = effective_screen_ratio > size_thresholds[:, lod]
                lod_from_screen[mask] = lod
            
            lod_level = torch.max(lod_from_distance, lod_from_screen)
            
            current_lod = lod_level.clone()
            next_lod = (lod_level + 1).clamp(max=NUM_LOD_LEVELS - 1)
            current_dist_thresh = scaled_dist_thresholds.gather(1, current_lod.unsqueeze(1)).squeeze(1)
            next_dist_thresh = scaled_dist_thresholds.gather(1, next_lod.unsqueeze(1)).squeeze(1)
            thresh_range = (next_dist_thresh - current_dist_thresh).clamp(min=1.0)
            blend_factor = ((distance - current_dist_thresh) / thresh_range).clamp(0, 1)
            
            should_cull = (distance > cull_dist) | (~in_front) | (effective_screen_ratio < 0.0005)
            lod_level[should_cull] = NUM_LOD_LEVELS - 1
            
            return lod_level, blend_factor, should_cull
        
        # Data generation
        print(f"\n   Generating {num_samples:,} samples...")
        start = time.time()
        B = num_samples
        
        camera_pos = torch.zeros(B, 3, device=device)
        camera_pos[:, 0] = torch.rand(B, device=device) * 400 - 200
        camera_pos[:, 1] = torch.rand(B, device=device) * 400 - 200
        camera_pos[:, 2] = torch.rand(B, device=device) * 5 + 1.5
        
        yaw = torch.rand(B, device=device) * 2 * np.pi
        pitch = (torch.rand(B, device=device) - 0.5) * np.pi * 0.5
        
        camera_forward = torch.zeros(B, 3, device=device)
        camera_forward[:, 0] = torch.cos(pitch) * torch.cos(yaw)
        camera_forward[:, 1] = torch.cos(pitch) * torch.sin(yaw)
        camera_forward[:, 2] = torch.sin(pitch)
        
        camera_fov = torch.rand(B, device=device) * 20 + 90
        
        speed_type = torch.rand(B, device=device)
        camera_speed = torch.zeros(B, device=device)
        camera_speed[speed_type < 0.7] = torch.rand((speed_type < 0.7).sum(), device=device) * 2
        camera_speed[(speed_type >= 0.7) & (speed_type < 0.9)] = torch.rand(((speed_type >= 0.7) & (speed_type < 0.9)).sum(), device=device) * 5 + 2
        camera_speed[speed_type >= 0.9] = torch.rand((speed_type >= 0.9).sum(), device=device) * 15 + 7
        
        vel_yaw = yaw + (torch.rand(B, device=device) - 0.5) * 0.5
        camera_velocity = torch.zeros(B, 3, device=device)
        camera_velocity[:, 0] = camera_speed * torch.cos(vel_yaw)
        camera_velocity[:, 1] = camera_speed * torch.sin(vel_yaw)
        
        distance_type = torch.rand(B, device=device)
        object_distance = torch.zeros(B, device=device)
        object_distance[distance_type < 0.2] = torch.rand((distance_type < 0.2).sum(), device=device) * 10 + 1
        object_distance[(distance_type >= 0.2) & (distance_type < 0.5)] = torch.rand(((distance_type >= 0.2) & (distance_type < 0.5)).sum(), device=device) * 40 + 10
        object_distance[(distance_type >= 0.5) & (distance_type < 0.8)] = torch.rand(((distance_type >= 0.5) & (distance_type < 0.8)).sum(), device=device) * 100 + 50
        object_distance[distance_type >= 0.8] = torch.rand((distance_type >= 0.8).sum(), device=device) * 200 + 150
        
        obj_angle = torch.rand(B, device=device) * 2 * np.pi
        forward_bias = torch.rand(B, device=device) < 0.7
        obj_angle[forward_bias] = yaw[forward_bias] + (torch.rand(forward_bias.sum(), device=device) - 0.5) * np.pi * 0.8
        obj_pitch = (torch.rand(B, device=device) - 0.5) * 0.3
        
        object_pos = torch.zeros(B, 3, device=device)
        object_pos[:, 0] = camera_pos[:, 0] + object_distance * torch.cos(obj_pitch) * torch.cos(obj_angle)
        object_pos[:, 1] = camera_pos[:, 1] + object_distance * torch.cos(obj_pitch) * torch.sin(obj_angle)
        object_pos[:, 2] = torch.rand(B, device=device) * 20
        
        radius_type = torch.rand(B, device=device)
        object_radius = torch.zeros(B, device=device)
        object_radius[radius_type < 0.3] = torch.rand((radius_type < 0.3).sum(), device=device) * 0.5 + 0.2
        object_radius[(radius_type >= 0.3) & (radius_type < 0.6)] = torch.rand(((radius_type >= 0.3) & (radius_type < 0.6)).sum(), device=device) * 1.5 + 0.5
        object_radius[(radius_type >= 0.6) & (radius_type < 0.85)] = torch.rand(((radius_type >= 0.6) & (radius_type < 0.85)).sum(), device=device) * 3 + 2
        object_radius[radius_type >= 0.85] = torch.rand((radius_type >= 0.85).sum(), device=device) * 10 + 5
        
        importance_weights = torch.tensor([0.05, 0.10, 0.20, 0.35, 0.30], device=device)
        object_importance = torch.multinomial(importance_weights.expand(B, -1), 1).squeeze()
        
        lod_level, blend_factor, should_cull = compute_lod_selection(
            camera_pos, camera_forward, camera_fov, camera_velocity,
            object_pos, object_radius, object_importance
        )
        
        rel_pos = object_pos - camera_pos
        importance_oh = F.one_hot(object_importance, NUM_IMPORTANCE_LEVELS).float()
        
        X = torch.cat([
            camera_forward, (camera_fov.unsqueeze(1) - 90) / 30,
            camera_velocity / 15.0, rel_pos / 200.0, object_radius.unsqueeze(1) / 10.0,
            importance_oh,
        ], dim=1)
        
        Y_lod = lod_level
        Y_blend = blend_factor
        Y_cull = should_cull.float()
        
        idx = torch.randperm(B, device=device)
        X, Y_lod, Y_blend, Y_cull = X[idx], Y_lod[idx], Y_blend[idx], Y_cull[idx]
        
        print(f"   ✓ Generated in {time.time()-start:.1f}s | Features: {X.shape[1]}")
        
        # Model
        class LODSelectionNet(nn.Module):
            def __init__(self, input_dim=16, hidden=192, blocks=5):
                super().__init__()
                self.proj = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
                self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(blocks)])
                self.lod_head = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, NUM_LOD_LEVELS))
                self.blend_head = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())
                self.cull_head = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1))
                self.n_params = sum(p.numel() for p in self.parameters())
            
            def forward(self, x):
                x = self.proj(x)
                for block in self.blocks:
                    x = block(x)
                return self.lod_head(x), self.blend_head(x), self.cull_head(x)
        
        n_val = num_samples // 10
        X_train, X_val = X[n_val:], X[:n_val]
        Y_lod_train, Y_lod_val = Y_lod[n_val:], Y_lod[:n_val]
        Y_blend_train, Y_blend_val = Y_blend[n_val:], Y_blend[:n_val]
        Y_cull_train, Y_cull_val = Y_cull[n_val:], Y_cull[:n_val]
        
        model = LODSelectionNet(input_dim=X.shape[1]).to(device)
        print(f"   Model: {model.n_params:,} parameters")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, epochs=epochs,
            steps_per_epoch=(len(X_train) // batch_size) + 1
        )
        
        ce_loss = nn.CrossEntropyLoss()
        bce_loss = nn.BCEWithLogitsLoss()
        mse_loss = nn.MSELoss()
        
        print(f"\n   {'Ep':>3} | {'Train':>10} | {'Val':>10} | {'LOD Acc':>8} | {'Best':>10}")
        print("   " + "-" * 55)
        
        best_loss = float('inf')
        best_state = None
        no_improve = 0
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            n_batches = 0
            
            idx = torch.randperm(len(X_train), device=device)
            for i in range(0, len(X_train), batch_size):
                batch_idx = idx[i:i+batch_size]
                xb = X_train[batch_idx]
                
                optimizer.zero_grad()
                lod_logits, blend_pred, cull_logit = model(xb)
                
                loss = (ce_loss(lod_logits, Y_lod_train[batch_idx]) + 
                       0.5 * mse_loss(blend_pred.squeeze(), Y_blend_train[batch_idx]) +
                       0.5 * bce_loss(cull_logit.squeeze(), Y_cull_train[batch_idx]))
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            train_loss /= n_batches
            
            model.eval()
            with torch.no_grad():
                lod_logits, blend_pred, cull_logit = model(X_val)
                val_loss = (ce_loss(lod_logits, Y_lod_val).item() + 
                           0.5 * mse_loss(blend_pred.squeeze(), Y_blend_val).item() +
                           0.5 * bce_loss(cull_logit.squeeze(), Y_cull_val).item())
                
                lod_pred = lod_logits.argmax(dim=1)
                lod_acc = (lod_pred == Y_lod_val).float().mean().item()
            
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            if epoch % 5 == 0 or is_best:
                print(f"   {epoch+1:>3} | {train_loss:>10.4f} | {val_loss:>10.4f} | "
                      f"{lod_acc*100:>7.2f}% | {best_loss:>10.4f}{'*' if is_best else ''}")
            
            if epoch % 10 == 0:
                save_checkpoint(model, optimizer, epoch, val_loss, KERNEL_NAME, VERSION)
            
            if no_improve >= patience:
                print(f"\n   Early stop at epoch {epoch+1}")
                break
        
        model.load_state_dict(best_state)
        
        model.eval()
        with torch.no_grad():
            lod_logits, _, cull_logit = model(X_val)
            lod_pred = lod_logits.argmax(dim=1)
            lod_acc = (lod_pred == Y_lod_val).float().mean().item()
            lod_within_1 = ((lod_pred - Y_lod_val).abs() <= 1).float().mean().item()
            cull_acc = ((cull_logit.squeeze() > 0) == (Y_cull_val > 0.5)).float().mean().item()
        
        metrics = {'lod_acc': lod_acc, 'lod_within_1': lod_within_1, 'cull_acc': cull_acc}
        
        print(f"\n   ═══════════════════════════════════════════════════")
        print(f"   FINAL: LOD Exact: {lod_acc*100:.2f}% | Within-1: {lod_within_1*100:.2f}% | Cull: {cull_acc*100:.2f}%")
        print(f"   ═══════════════════════════════════════════════════")
        
        model_path = f"{MODELS_DIR}/{KERNEL_NAME}_v{VERSION}.pt"
        torch.save({'model': model.state_dict(), 'metrics': metrics}, model_path)
        print(f"   ✓ Model saved: {model_path}")
        
        mark_success(KERNEL_NAME, VERSION, metrics)
        return model, metrics, VERSION
        
    except Exception as e:
        mark_failure(KERNEL_NAME, VERSION, e)
        print(f"\n   ❌ TRAINING FAILED: {e}")
        raise


# ═══════════════════════════════════════════════════════════════════════════════════════
# CELL 6: KERNEL 3 - OCCLUSION CULLING
# ═══════════════════════════════════════════════════════════════════════════════════════

def train_occlusion_culling(num_samples=2_000_000, epochs=100, batch_size=8192, patience=15):
    """
    Train Occlusion Culling Kernel
    Replaces GPU occlusion queries (with pipeline stalls) with O(1) CPU inference
    """
    KERNEL_NAME = "OcclusionCulling"
    VERSION = get_next_version(KERNEL_NAME)
    
    print(f"\n{'█'*70}")
    print(f"█ TRAINING {KERNEL_NAME} v{VERSION} ".ljust(69) + "█")
    print(f"{'█'*70}")
    
    try:
        NUM_OCCLUDER_TYPES = 5
        OCCLUDER_EFFECTIVENESS = torch.tensor([0.0, 0.6, 0.75, 0.95, 0.85], dtype=torch.float32, device=device)
        
        def compute_visibility(camera_pos, camera_forward, camera_fov, object_pos, object_radius,
                                occluder_pos, occluder_size, occluder_type, prev_visible):
            B = len(object_pos)
            
            to_object = object_pos - camera_pos
            distance = torch.norm(to_object, dim=1).clamp(min=0.1)
            
            camera_forward = F.normalize(camera_forward, dim=1)
            to_object_norm = F.normalize(to_object, dim=1)
            
            dot = (camera_forward * to_object_norm).sum(dim=1)
            fov_rad = camera_fov * np.pi / 180
            half_fov_cos = torch.cos(fov_rad / 2)
            in_frustum = dot > half_fov_cos * 0.8
            
            # Check occlusion
            to_occluder = occluder_pos - camera_pos
            occluder_dist = torch.norm(to_occluder, dim=1).clamp(min=0.1)
            
            occluder_between = occluder_dist < distance
            
            to_occluder_norm = F.normalize(to_occluder, dim=1)
            alignment = (to_object_norm * to_occluder_norm).sum(dim=1)
            aligned = alignment > 0.9
            
            occluder_eff = OCCLUDER_EFFECTIVENESS[occluder_type]
            
            occluded = occluder_between & aligned & (torch.rand(B, device=device) < occluder_eff)
            
            # Temporal coherence
            temporal_factor = prev_visible * 0.1
            
            visible = in_frustum & ~occluded
            confidence = torch.where(visible, 0.9 + temporal_factor * 0.1, 0.1 + (1 - temporal_factor) * 0.1)
            
            return visible, confidence
        
        print(f"\n   Generating {num_samples:,} samples...")
        start = time.time()
        B = num_samples
        
        camera_pos = torch.zeros(B, 3, device=device)
        camera_pos[:, 0] = torch.rand(B, device=device) * 300 - 150
        camera_pos[:, 1] = torch.rand(B, device=device) * 300 - 150
        camera_pos[:, 2] = torch.rand(B, device=device) * 4 + 1.5
        
        yaw = torch.rand(B, device=device) * 2 * np.pi
        pitch = (torch.rand(B, device=device) - 0.5) * np.pi * 0.4
        
        camera_forward = torch.zeros(B, 3, device=device)
        camera_forward[:, 0] = torch.cos(pitch) * torch.cos(yaw)
        camera_forward[:, 1] = torch.cos(pitch) * torch.sin(yaw)
        camera_forward[:, 2] = torch.sin(pitch)
        
        camera_fov = torch.rand(B, device=device) * 20 + 90
        
        dist_type = torch.rand(B, device=device)
        object_distance = torch.zeros(B, device=device)
        object_distance[dist_type < 0.3] = torch.rand((dist_type < 0.3).sum(), device=device) * 20 + 1
        object_distance[(dist_type >= 0.3) & (dist_type < 0.7)] = torch.rand(((dist_type >= 0.3) & (dist_type < 0.7)).sum(), device=device) * 60 + 20
        object_distance[dist_type >= 0.7] = torch.rand((dist_type >= 0.7).sum(), device=device) * 120 + 80
        
        obj_angle = yaw + (torch.rand(B, device=device) - 0.5) * np.pi * 1.2
        obj_pitch = (torch.rand(B, device=device) - 0.5) * 0.4
        
        object_pos = torch.zeros(B, 3, device=device)
        object_pos[:, 0] = camera_pos[:, 0] + object_distance * torch.cos(obj_pitch) * torch.cos(obj_angle)
        object_pos[:, 1] = camera_pos[:, 1] + object_distance * torch.cos(obj_pitch) * torch.sin(obj_angle)
        object_pos[:, 2] = torch.rand(B, device=device) * 15
        
        object_radius = torch.rand(B, device=device) * 4 + 0.5
        
        # Occluder (wall, terrain, etc.)
        occluder_distance = object_distance * torch.rand(B, device=device) * 0.8
        occluder_offset = (torch.rand(B, device=device) - 0.5) * 0.3
        
        occluder_pos = torch.zeros(B, 3, device=device)
        occluder_pos[:, 0] = camera_pos[:, 0] + occluder_distance * torch.cos(obj_pitch) * torch.cos(obj_angle + occluder_offset)
        occluder_pos[:, 1] = camera_pos[:, 1] + occluder_distance * torch.cos(obj_pitch) * torch.sin(obj_angle + occluder_offset)
        occluder_pos[:, 2] = torch.rand(B, device=device) * 10
        
        occluder_size = torch.rand(B, device=device) * 8 + 2
        
        occluder_weights = torch.tensor([0.3, 0.2, 0.2, 0.15, 0.15], device=device)
        occluder_type = torch.multinomial(occluder_weights.expand(B, -1), 1).squeeze()
        
        prev_visible = (torch.rand(B, device=device) > 0.3).float()
        
        visible, confidence = compute_visibility(
            camera_pos, camera_forward, camera_fov, object_pos, object_radius,
            occluder_pos, occluder_size, occluder_type, prev_visible
        )
        
        rel_object = object_pos - camera_pos
        rel_occluder = occluder_pos - camera_pos
        occluder_oh = F.one_hot(occluder_type, NUM_OCCLUDER_TYPES).float()
        
        X = torch.cat([
            camera_forward,
            (camera_fov.unsqueeze(1) - 90) / 30,
            rel_object / 100.0,
            object_radius.unsqueeze(1) / 5.0,
            rel_occluder / 100.0,
            occluder_size.unsqueeze(1) / 10.0,
            occluder_oh,
            prev_visible.unsqueeze(1),
        ], dim=1)
        
        Y_visible = visible.float()
        Y_confidence = confidence
        
        idx = torch.randperm(B, device=device)
        X, Y_visible, Y_confidence = X[idx], Y_visible[idx], Y_confidence[idx]
        
        print(f"   ✓ Generated in {time.time()-start:.1f}s | Features: {X.shape[1]}")
        print(f"   Visible: {visible.sum().item():,} ({100*visible.sum().item()/B:.1f}%)")
        
        class OcclusionCullingNet(nn.Module):
            def __init__(self, input_dim=20, hidden=224, blocks=6):
                super().__init__()
                self.proj = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
                self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(blocks)])
                self.visible_head = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 1))
                self.confidence_head = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())
                self.n_params = sum(p.numel() for p in self.parameters())
            
            def forward(self, x):
                x = self.proj(x)
                for block in self.blocks:
                    x = block(x)
                return self.visible_head(x), self.confidence_head(x)
        
        n_val = num_samples // 10
        X_train, X_val = X[n_val:], X[:n_val]
        Y_vis_train, Y_vis_val = Y_visible[n_val:], Y_visible[:n_val]
        Y_conf_train, Y_conf_val = Y_confidence[n_val:], Y_confidence[:n_val]
        
        model = OcclusionCullingNet(input_dim=X.shape[1]).to(device)
        print(f"   Model: {model.n_params:,} parameters")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, epochs=epochs,
            steps_per_epoch=(len(X_train) // batch_size) + 1
        )
        
        bce_loss = nn.BCEWithLogitsLoss()
        mse_loss = nn.MSELoss()
        
        print(f"\n   {'Ep':>3} | {'Train':>10} | {'Val':>10} | {'Vis Acc':>8} | {'Best':>10}")
        print("   " + "-" * 55)
        
        best_loss = float('inf')
        best_state = None
        no_improve = 0
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            n_batches = 0
            
            idx = torch.randperm(len(X_train), device=device)
            for i in range(0, len(X_train), batch_size):
                batch_idx = idx[i:i+batch_size]
                xb = X_train[batch_idx]
                
                optimizer.zero_grad()
                vis_logit, conf_pred = model(xb)
                
                loss = bce_loss(vis_logit.squeeze(), Y_vis_train[batch_idx]) + 0.3 * mse_loss(conf_pred.squeeze(), Y_conf_train[batch_idx])
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            train_loss /= n_batches
            
            model.eval()
            with torch.no_grad():
                vis_logit, conf_pred = model(X_val)
                val_loss = bce_loss(vis_logit.squeeze(), Y_vis_val).item() + 0.3 * mse_loss(conf_pred.squeeze(), Y_conf_val).item()
                
                vis_pred = vis_logit.squeeze() > 0
                vis_acc = (vis_pred == (Y_vis_val > 0.5)).float().mean().item()
            
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            if epoch % 5 == 0 or is_best:
                print(f"   {epoch+1:>3} | {train_loss:>10.4f} | {val_loss:>10.4f} | "
                      f"{vis_acc*100:>7.2f}% | {best_loss:>10.4f}{'*' if is_best else ''}")
            
            if epoch % 10 == 0:
                save_checkpoint(model, optimizer, epoch, val_loss, KERNEL_NAME, VERSION)
            
            if no_improve >= patience:
                print(f"\n   Early stop at epoch {epoch+1}")
                break
        
        model.load_state_dict(best_state)
        
        model.eval()
        with torch.no_grad():
            vis_logit, _ = model(X_val)
            vis_pred = vis_logit.squeeze() > 0
            vis_true = Y_vis_val > 0.5
            
            vis_acc = (vis_pred == vis_true).float().mean().item()
            
            # False negative rate (critical - missing visible objects)
            true_visible = vis_true.sum().item()
            false_negatives = ((~vis_pred) & vis_true).sum().item()
            fn_rate = false_negatives / max(true_visible, 1)
            
            # Conservative accuracy (visible objects correctly identified)
            recall = (vis_pred & vis_true).sum().item() / max(true_visible, 1)
        
        metrics = {'visibility_acc': vis_acc, 'recall': recall, 'false_negative_rate': fn_rate}
        
        print(f"\n   ═══════════════════════════════════════════════════")
        print(f"   FINAL: Accuracy: {vis_acc*100:.2f}% | Recall: {recall*100:.2f}% | FN Rate: {fn_rate*100:.2f}%")
        print(f"   ═══════════════════════════════════════════════════")
        
        model_path = f"{MODELS_DIR}/{KERNEL_NAME}_v{VERSION}.pt"
        torch.save({'model': model.state_dict(), 'metrics': metrics}, model_path)
        print(f"   ✓ Model saved: {model_path}")
        
        mark_success(KERNEL_NAME, VERSION, metrics)
        return model, metrics, VERSION
        
    except Exception as e:
        mark_failure(KERNEL_NAME, VERSION, e)
        print(f"\n   ❌ TRAINING FAILED: {e}")
        raise


# ═══════════════════════════════════════════════════════════════════════════════════════
# CELL 7: KERNEL 4 - LUMEN GI APPROXIMATION
# ═══════════════════════════════════════════════════════════════════════════════════════

def train_lumen_gi(num_samples=2_000_000, epochs=100, batch_size=8192, patience=15):
    """
    Train Lumen GI Approximation Kernel
    Replaces UE5's expensive Lumen ray tracing with O(1) neural inference
    """
    KERNEL_NAME = "LumenGI"
    VERSION = get_next_version(KERNEL_NAME)
    
    print(f"\n{'█'*70}")
    print(f"█ TRAINING {KERNEL_NAME} v{VERSION} ".ljust(69) + "█")
    print(f"{'█'*70}")
    
    try:
        NUM_SURFACE_TYPES = 6
        SURFACE_TYPES = {
            0: 'grass', 1: 'stone', 2: 'wood', 3: 'metal', 4: 'sand', 5: 'water',
        }
        
        # Surface albedo (how much light they reflect)
        SURFACE_ALBEDO = torch.tensor([
            [0.2, 0.4, 0.1],   # grass - greenish
            [0.4, 0.4, 0.4],   # stone - gray
            [0.5, 0.3, 0.2],   # wood - brownish
            [0.6, 0.6, 0.65],  # metal - bright, slightly blue
            [0.8, 0.7, 0.5],   # sand - yellowish
            [0.1, 0.2, 0.3],   # water - dark blue
        ], dtype=torch.float32, device=device)
        
        def compute_indirect_lighting(surface_pos, surface_normal, sun_dir, sun_intensity,
                                        time_of_day, indoor, surface_type, nearby_surfaces):
            B = len(surface_pos)
            
            sun_dir = F.normalize(sun_dir, dim=1)
            surface_normal = F.normalize(surface_normal, dim=1)
            
            # Direct sun contribution to GI (sun hitting nearby surfaces)
            sun_dot = (sun_dir * surface_normal).sum(dim=1).clamp(0, 1)
            
            # Sky contribution (ambient)
            sky_factor = surface_normal[:, 2].clamp(0, 1)  # Upward-facing = more sky
            sky_color = torch.tensor([0.5, 0.7, 1.0], device=device) * sky_factor.unsqueeze(1)
            
            # Ground bounce
            ground_factor = (-surface_normal[:, 2]).clamp(0, 1)
            ground_albedo = SURFACE_ALBEDO[surface_type]
            ground_bounce = ground_albedo * ground_factor.unsqueeze(1) * sun_intensity.unsqueeze(1) * 0.3
            
            # Indoor darkening
            indoor_factor = 1.0 - indoor * 0.6
            
            # Time of day effect
            # 0 = midnight, 0.5 = noon, 1 = midnight
            day_factor = torch.sin(time_of_day * np.pi).clamp(0, 1)
            
            # Nearby surface bounce (simplified)
            nearby_contribution = nearby_surfaces.mean(dim=1) * 0.2
            
            # Combine indirect light
            indirect_color = (sky_color * 0.3 + ground_bounce + nearby_contribution.unsqueeze(1) * 0.1) * indoor_factor.unsqueeze(1) * day_factor.unsqueeze(1)
            indirect_intensity = indirect_color.norm(dim=1)
            
            # Ambient occlusion (simplified - based on normal orientation and indoor)
            ao = (0.8 + 0.2 * surface_normal[:, 2]) * (1.0 - indoor * 0.3)
            
            # Sky visibility
            sky_vis = (1.0 - indoor) * sky_factor
            
            return indirect_color, indirect_intensity, ao, sky_vis
        
        print(f"\n   Generating {num_samples:,} samples...")
        start = time.time()
        B = num_samples
        
        # Surface positions (world space)
        surface_pos = torch.zeros(B, 3, device=device)
        surface_pos[:, 0] = torch.rand(B, device=device) * 200 - 100
        surface_pos[:, 1] = torch.rand(B, device=device) * 200 - 100
        surface_pos[:, 2] = torch.rand(B, device=device) * 30  # 0-30m height
        
        # Surface normals (mostly upward/sideways)
        normal_type = torch.rand(B, device=device)
        surface_normal = torch.zeros(B, 3, device=device)
        
        # 40% upward (floors, ground)
        up_mask = normal_type < 0.4
        surface_normal[up_mask, 2] = 1.0
        
        # 40% sideways (walls)
        side_mask = (normal_type >= 0.4) & (normal_type < 0.8)
        wall_angle = torch.rand(side_mask.sum(), device=device) * 2 * np.pi
        surface_normal[side_mask, 0] = torch.cos(wall_angle)
        surface_normal[side_mask, 1] = torch.sin(wall_angle)
        
        # 20% downward (ceilings)
        down_mask = normal_type >= 0.8
        surface_normal[down_mask, 2] = -1.0
        
        # Add noise to normals
        surface_normal += torch.randn_like(surface_normal) * 0.1
        surface_normal = F.normalize(surface_normal, dim=1)
        
        # Sun direction (varies with time of day)
        time_of_day = torch.rand(B, device=device)  # 0 = midnight, 0.5 = noon
        sun_elevation = torch.sin(time_of_day * np.pi) * 0.8  # Max 80 degrees at noon
        sun_azimuth = torch.rand(B, device=device) * 2 * np.pi
        
        sun_dir = torch.zeros(B, 3, device=device)
        sun_dir[:, 0] = torch.cos(sun_azimuth) * torch.cos(torch.asin(sun_elevation))
        sun_dir[:, 1] = torch.sin(sun_azimuth) * torch.cos(torch.asin(sun_elevation))
        sun_dir[:, 2] = sun_elevation
        
        sun_intensity = (sun_elevation.clamp(0, 1) + 0.1) * torch.rand(B, device=device) * 0.5 + 0.5
        
        # Indoor/outdoor
        indoor = (torch.rand(B, device=device) < 0.4).float()
        
        # Surface types
        surface_weights = torch.tensor([0.25, 0.25, 0.15, 0.15, 0.1, 0.1], device=device)
        surface_type = torch.multinomial(surface_weights.expand(B, -1), 1).squeeze()
        
        # Nearby surfaces (simplified - just average albedo of neighbors)
        nearby_surfaces = torch.rand(B, 4, device=device) * 0.5 + 0.2
        
        indirect_color, indirect_intensity, ao, sky_vis = compute_indirect_lighting(
            surface_pos, surface_normal, sun_dir, sun_intensity,
            time_of_day, indoor, surface_type, nearby_surfaces
        )
        
        surface_oh = F.one_hot(surface_type, NUM_SURFACE_TYPES).float()
        
        X = torch.cat([
            surface_pos / 100.0,
            surface_normal,
            sun_dir,
            sun_intensity.unsqueeze(1),
            time_of_day.unsqueeze(1),
            indoor.unsqueeze(1),
            surface_oh,
            nearby_surfaces,
        ], dim=1)
        
        Y = torch.cat([
            indirect_color,
            indirect_intensity.unsqueeze(1),
            ao.unsqueeze(1),
            sky_vis.unsqueeze(1),
        ], dim=1)
        
        idx = torch.randperm(B, device=device)
        X, Y = X[idx], Y[idx]
        
        print(f"   ✓ Generated in {time.time()-start:.1f}s | Features: {X.shape[1]} | Outputs: {Y.shape[1]}")
        
        class LumenGINet(nn.Module):
            def __init__(self, input_dim=21, hidden=256, blocks=6):
                super().__init__()
                self.proj = nn.Sequential(nn.Linear(input_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
                self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(blocks)])
                
                self.color_head = nn.Sequential(nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 3), nn.Sigmoid())
                self.intensity_head = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())
                self.ao_head = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())
                self.skyvis_head = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1), nn.Sigmoid())
                
                self.n_params = sum(p.numel() for p in self.parameters())
            
            def forward(self, x):
                x = self.proj(x)
                for block in self.blocks:
                    x = block(x)
                return torch.cat([
                    self.color_head(x),
                    self.intensity_head(x),
                    self.ao_head(x),
                    self.skyvis_head(x),
                ], dim=1)
        
        n_val = num_samples // 10
        X_train, X_val = X[n_val:], X[:n_val]
        Y_train, Y_val = Y[n_val:], Y[:n_val]
        
        model = LumenGINet(input_dim=X.shape[1]).to(device)
        print(f"   Model: {model.n_params:,} parameters")
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-3, epochs=epochs,
            steps_per_epoch=(len(X_train) // batch_size) + 1
        )
        
        # Perceptual color loss weights (intensity matters most)
        output_weights = torch.tensor([1.0, 1.0, 1.0, 2.0, 1.5, 1.0], device=device)
        
        print(f"\n   {'Ep':>3} | {'Train':>10} | {'Val':>10} | {'Color':>8} | {'Best':>10}")
        print("   " + "-" * 55)
        
        best_loss = float('inf')
        best_state = None
        no_improve = 0
        
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            n_batches = 0
            
            idx = torch.randperm(len(X_train), device=device)
            for i in range(0, len(X_train), batch_size):
                batch_idx = idx[i:i+batch_size]
                xb, yb = X_train[batch_idx], Y_train[batch_idx]
                
                optimizer.zero_grad()
                pred = model(xb)
                loss = ((pred - yb).pow(2) * output_weights).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                n_batches += 1
            
            train_loss /= n_batches
            
            model.eval()
            with torch.no_grad():
                pred = model(X_val)
                val_loss = ((pred - Y_val).pow(2) * output_weights).mean().item()
                
                color_mae = (pred[:, :3] - Y_val[:, :3]).abs().mean().item()
            
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
                best_state = {k: v.clone() for k, v in model.state_dict().items()}
                no_improve = 0
            else:
                no_improve += 1
            
            if epoch % 5 == 0 or is_best:
                print(f"   {epoch+1:>3} | {train_loss:>10.6f} | {val_loss:>10.6f} | "
                      f"{color_mae:>8.4f} | {best_loss:>10.6f}{'*' if is_best else ''}")
            
            if epoch % 10 == 0:
                save_checkpoint(model, optimizer, epoch, val_loss, KERNEL_NAME, VERSION)
            
            if no_improve >= patience:
                print(f"\n   Early stop at epoch {epoch+1}")
                break
        
        model.load_state_dict(best_state)
        
        model.eval()
        with torch.no_grad():
            pred = model(X_val)
            
            color_mae = (pred[:, :3] - Y_val[:, :3]).abs().mean().item()
            color_acc = ((pred[:, :3] - Y_val[:, :3]).abs() < 0.05).all(dim=1).float().mean().item()
            intensity_mae = (pred[:, 3] - Y_val[:, 3]).abs().mean().item()
            ao_mae = (pred[:, 4] - Y_val[:, 4]).abs().mean().item()
        
        metrics = {'color_acc': color_acc, 'color_mae': color_mae, 'intensity_mae': intensity_mae, 'ao_mae': ao_mae}
        
        print(f"\n   ═══════════════════════════════════════════════════")
        print(f"   FINAL: Color Acc: {color_acc*100:.2f}% | MAE: {color_mae:.4f}")
        print(f"   Intensity MAE: {intensity_mae:.4f} | AO MAE: {ao_mae:.4f}")
        print(f"   ═══════════════════════════════════════════════════")
        
        model_path = f"{MODELS_DIR}/{KERNEL_NAME}_v{VERSION}.pt"
        torch.save({'model': model.state_dict(), 'metrics': metrics}, model_path)
        print(f"   ✓ Model saved: {model_path}")
        
        mark_success(KERNEL_NAME, VERSION, metrics)
        return model, metrics, VERSION
        
    except Exception as e:
        mark_failure(KERNEL_NAME, VERSION, e)
        print(f"\n   ❌ TRAINING FAILED: {e}")
        raise


# ═══════════════════════════════════════════════════════════════════════════════════════
# CELL 8: TRAIN ALL TIER 1 KERNELS
# ═══════════════════════════════════════════════════════════════════════════════════════

def train_all_tier1():
    """Train all Tier 1 kernels in sequence"""
    print("\n" + "█" * 70)
    print("█" + " HYPERFLUX TIER 1 TRAINING - ALL KERNELS ".center(68) + "█")
    print("█" * 70)
    
    results = {}
    
    # 1. Audio Occlusion
    try:
        model, metrics, version = train_audio_occlusion()
        results['AudioOcclusion'] = {'status': 'success', 'version': version, 'metrics': metrics}
    except Exception as e:
        results['AudioOcclusion'] = {'status': 'failed', 'error': str(e)}
    
    # 2. LOD Selection
    try:
        model, metrics, version = train_lod_selection()
        results['LODSelection'] = {'status': 'success', 'version': version, 'metrics': metrics}
    except Exception as e:
        results['LODSelection'] = {'status': 'failed', 'error': str(e)}
    
    # 3. Occlusion Culling
    try:
        model, metrics, version = train_occlusion_culling()
        results['OcclusionCulling'] = {'status': 'success', 'version': version, 'metrics': metrics}
    except Exception as e:
        results['OcclusionCulling'] = {'status': 'failed', 'error': str(e)}
    
    # 4. Lumen GI
    try:
        model, metrics, version = train_lumen_gi()
        results['LumenGI'] = {'status': 'success', 'version': version, 'metrics': metrics}
    except Exception as e:
        results['LumenGI'] = {'status': 'failed', 'error': str(e)}
    
    # Summary
    print("\n" + "█" * 70)
    print("█" + " TIER 1 TRAINING COMPLETE ".center(68) + "█")
    print("█" * 70)
    
    for kernel, result in results.items():
        if result['status'] == 'success':
            print(f"   ✅ {kernel} v{result['version']}")
        else:
            print(f"   ❌ {kernel}: {result['error']}")
    
    # Save summary
    summary_path = f"{TIER1_DIR}/training_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n   Summary saved: {summary_path}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════════════
# CELL 9: RUN TRAINING
# ═══════════════════════════════════════════════════════════════════════════════════════

# Uncomment and run ONE of these:

# Train all Tier 1 kernels:
# results = train_all_tier1()

# Or train individual kernels:
# model, metrics, version = train_audio_occlusion()
# model, metrics, version = train_lod_selection()
# model, metrics, version = train_occlusion_culling()
# model, metrics, version = train_lumen_gi()

print("\n" + "═" * 70)
print(" Ready to train! Uncomment one of the training calls above.")
print(" All files will be saved to: /content/drive/MyDrive/Highguard/Tier1/")
print("═" * 70)
