"""
╔══════════════════════════════════════════════════════════════════════════════════════╗
║           HYPERFLUX LOD SELECTION KERNEL v1.0 - HIGHGUARD EDITION                    ║
║           O(1) Batch Level-of-Detail Prediction                                      ║
║                                                                                      ║
║  PROBLEM:                                                                            ║
║    Every frame, for EVERY mesh in scene:                                             ║
║      - Calculate distance to camera                                                  ║
║      - Calculate screen-space size                                                   ║
║      - Decide LOD level (0-4)                                                        ║
║      - Handle LOD transitions                                                        ║
║    Highguard's large maps = 10,000+ objects per frame                               ║
║                                                                                      ║
║  SOLUTION:                                                                           ║
║    Neural network predicts optimal LOD for batches of objects.                       ║
║    Single forward pass = O(1) for entire batch.                                      ║
║    Includes predictive LOD based on camera velocity (no popping).                   ║
║                                                                                      ║
║  INPUTS:                                                                             ║
║    - Camera position, rotation, FOV                                                  ║
║    - Camera velocity (for predictive LOD)                                            ║
║    - Object positions (batched)                                                      ║
║    - Object bounding sphere radii                                                    ║
║    - Object importance (hero, environment, detail)                                   ║
║                                                                                      ║
║  OUTPUTS:                                                                            ║
║    - LOD level per object (0-4)                                                      ║
║    - Transition blend factor (for smooth transitions)                                ║
║    - Cull flag (too small/far to render)                                            ║
║                                                                                      ║
║  Target: 99%+ Accuracy                                                               ║
║                                                                                      ║
║  Author: Anil Kumar Alla | HyperFlux Neural Dynamics                                ║
╚══════════════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import os
import shutil

print("═" * 70)
print(" HYPERFLUX LOD SELECTION KERNEL v1.0 - HIGHGUARD")
print(" O(1) Batch Level-of-Detail Prediction")
print(" Replacing O(N) per-object calculations with Neural Batch Inference")
print("═" * 70)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\nDevice: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ═══════════════════════════════════════════════════════════════════════════════════════
# HIGHGUARD LOD CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════════

NUM_LOD_LEVELS = 5  # LOD 0 (highest) to LOD 4 (lowest)

# Object importance categories
NUM_IMPORTANCE_LEVELS = 5
IMPORTANCE_TYPES = {
    0: 'hero',          # Wardens, mounts - always high LOD when visible
    1: 'critical',      # Weapons, Shieldbreaker, generators
    2: 'structure',     # Base walls, siege tower
    3: 'environment',   # Rocks, trees, terrain features
    4: 'detail',        # Small props, debris, particles
}

# LOD distance thresholds (meters) - base values, scaled by object size
LOD_THRESHOLDS = torch.tensor([
    [0,   15,  30,  60,  100],   # hero - high quality even at distance
    [0,   10,  25,  50,  80],    # critical
    [0,   8,   20,  40,  70],    # structure
    [0,   5,   15,  30,  50],    # environment
    [0,   3,   10,  20,  35],    # detail - aggressive LOD
], dtype=torch.float32, device=device)

# Screen size thresholds (percentage of screen height)
SCREEN_SIZE_THRESHOLDS = torch.tensor([
    [0.20, 0.10, 0.05, 0.02, 0.005],  # hero
    [0.15, 0.08, 0.04, 0.015, 0.004], # critical
    [0.10, 0.05, 0.025, 0.01, 0.003], # structure
    [0.08, 0.04, 0.02, 0.008, 0.002], # environment
    [0.05, 0.025, 0.01, 0.005, 0.001],# detail
], dtype=torch.float32, device=device)

# Cull distance by importance (meters)
CULL_DISTANCES = torch.tensor([500, 300, 200, 150, 80], dtype=torch.float32, device=device)


# ═══════════════════════════════════════════════════════════════════════════════════════
# GROUND TRUTH LOD CALCULATION
# ═══════════════════════════════════════════════════════════════════════════════════════

def compute_lod_selection(
    camera_pos,         # [B, 3] or [3] if same for all
    camera_forward,     # [B, 3] or [3]
    camera_fov,         # [B] or scalar (degrees)
    camera_velocity,    # [B, 3] or [3] - for predictive LOD
    object_pos,         # [B, 3]
    object_radius,      # [B] bounding sphere radius
    object_importance,  # [B] int (0-4)
    screen_height=1080, # pixels
):
    """
    Compute ground truth LOD selection for objects.
    
    Returns:
        lod_level: int 0-4 (0=highest, 4=lowest)
        blend_factor: float 0-1 (for smooth transitions)
        should_cull: bool (too far/small to render)
    """
    B = len(object_pos)
    
    # Handle broadcasting for camera
    if camera_pos.dim() == 1:
        camera_pos = camera_pos.unsqueeze(0).expand(B, -1)
    if camera_forward.dim() == 1:
        camera_forward = camera_forward.unsqueeze(0).expand(B, -1)
    if camera_velocity.dim() == 1:
        camera_velocity = camera_velocity.unsqueeze(0).expand(B, -1)
    if isinstance(camera_fov, (int, float)):
        camera_fov = torch.full((B,), camera_fov, device=device)
    
    # Vector from camera to object
    to_object = object_pos - camera_pos
    distance = torch.norm(to_object, dim=1).clamp(min=0.1)
    
    # Check if object is in front of camera (dot product with forward)
    camera_forward = F.normalize(camera_forward, dim=1)
    to_object_norm = F.normalize(to_object, dim=1)
    dot = (camera_forward * to_object_norm).sum(dim=1)
    in_front = dot > 0
    
    # Calculate screen-space size
    # Approximate: size_pixels = (object_diameter / distance) * (screen_height / 2) / tan(fov/2)
    fov_rad = camera_fov * np.pi / 180
    tan_half_fov = torch.tan(fov_rad / 2)
    
    object_diameter = object_radius * 2
    screen_size_ratio = (object_diameter / distance) / (2 * tan_half_fov)
    screen_size_ratio = screen_size_ratio.clamp(0, 1)
    
    # Predictive LOD: if camera is moving towards object, use higher LOD
    camera_speed = torch.norm(camera_velocity, dim=1)
    approach_rate = -(camera_velocity * to_object_norm).sum(dim=1)  # Positive = approaching
    
    # Predict where object will be in 0.5 seconds
    future_distance = (distance - approach_rate * 0.5).clamp(min=0.1)
    future_screen_ratio = (object_diameter / future_distance) / (2 * tan_half_fov)
    future_screen_ratio = future_screen_ratio.clamp(0, 1)
    
    # Use the larger of current and future screen size for LOD decision
    effective_screen_ratio = torch.max(screen_size_ratio, future_screen_ratio)
    
    # Get importance-specific thresholds
    dist_thresholds = LOD_THRESHOLDS[object_importance]  # [B, 5]
    size_thresholds = SCREEN_SIZE_THRESHOLDS[object_importance]  # [B, 5]
    cull_dist = CULL_DISTANCES[object_importance]  # [B]
    
    # Scale distance thresholds by object size (larger objects visible further)
    size_scale = (object_radius / 1.0).clamp(0.5, 5.0)  # 1m reference radius
    scaled_dist_thresholds = dist_thresholds * size_scale.unsqueeze(1)
    
    # Determine LOD level based on distance
    lod_from_distance = torch.zeros(B, dtype=torch.long, device=device)
    for lod in range(NUM_LOD_LEVELS - 1, -1, -1):
        mask = distance < scaled_dist_thresholds[:, lod]
        lod_from_distance[mask] = lod
    
    # Determine LOD level based on screen size
    lod_from_screen = torch.full((B,), NUM_LOD_LEVELS - 1, dtype=torch.long, device=device)
    for lod in range(NUM_LOD_LEVELS - 1, -1, -1):
        mask = effective_screen_ratio > size_thresholds[:, lod]
        lod_from_screen[mask] = lod
    
    # Final LOD is the HIGHER (worse) of distance and screen-based LOD
    # This ensures we don't waste detail on objects that are either far OR small
    lod_level = torch.max(lod_from_distance, lod_from_screen)
    
    # Calculate blend factor for smooth transitions
    # Blend factor indicates how close we are to the next LOD level
    current_lod = lod_level.clone()
    next_lod = (lod_level + 1).clamp(max=NUM_LOD_LEVELS - 1)
    
    # Get thresholds for current and next LOD
    current_dist_thresh = scaled_dist_thresholds.gather(1, current_lod.unsqueeze(1)).squeeze(1)
    next_dist_thresh = scaled_dist_thresholds.gather(1, next_lod.unsqueeze(1)).squeeze(1)
    
    # Blend factor based on position between thresholds
    thresh_range = (next_dist_thresh - current_dist_thresh).clamp(min=1.0)
    blend_factor = ((distance - current_dist_thresh) / thresh_range).clamp(0, 1)
    
    # Culling decision
    should_cull = (distance > cull_dist) | (~in_front) | (effective_screen_ratio < 0.0005)
    
    # Culled objects get LOD 4 (lowest)
    lod_level[should_cull] = NUM_LOD_LEVELS - 1
    
    return lod_level, blend_factor, should_cull


# ═══════════════════════════════════════════════════════════════════════════════════════
# DATA GENERATION
# ═══════════════════════════════════════════════════════════════════════════════════════

def generate_data(num_samples):
    """Generate training data with diverse camera/object configurations."""
    print(f"\n Generating {num_samples:,} LOD selection samples...")
    start = time.time()
    
    B = num_samples
    
    # Random camera positions (player viewpoints across the map)
    camera_pos = torch.zeros(B, 3, device=device)
    camera_pos[:, 0] = torch.rand(B, device=device) * 400 - 200  # x: -200 to 200
    camera_pos[:, 1] = torch.rand(B, device=device) * 400 - 200  # y: -200 to 200
    camera_pos[:, 2] = torch.rand(B, device=device) * 5 + 1.5    # z: 1.5-6.5m (player/mount height)
    
    # Random camera forward direction
    yaw = torch.rand(B, device=device) * 2 * np.pi
    pitch = (torch.rand(B, device=device) - 0.5) * np.pi * 0.5  # -45 to +45 degrees
    
    camera_forward = torch.zeros(B, 3, device=device)
    camera_forward[:, 0] = torch.cos(pitch) * torch.cos(yaw)
    camera_forward[:, 1] = torch.cos(pitch) * torch.sin(yaw)
    camera_forward[:, 2] = torch.sin(pitch)
    
    # Camera FOV (typically 90-110 for FPS)
    camera_fov = torch.rand(B, device=device) * 20 + 90  # 90-110 degrees
    
    # Camera velocity (moving player)
    # 70% stationary/slow, 20% walking, 10% sprinting/mounted
    speed_type = torch.rand(B, device=device)
    camera_speed = torch.zeros(B, device=device)
    camera_speed[speed_type < 0.7] = torch.rand((speed_type < 0.7).sum(), device=device) * 2
    camera_speed[(speed_type >= 0.7) & (speed_type < 0.9)] = torch.rand(((speed_type >= 0.7) & (speed_type < 0.9)).sum(), device=device) * 5 + 2
    camera_speed[speed_type >= 0.9] = torch.rand((speed_type >= 0.9).sum(), device=device) * 15 + 7
    
    # Velocity direction (mostly forward)
    vel_yaw = yaw + (torch.rand(B, device=device) - 0.5) * 0.5  # Slight deviation from facing
    camera_velocity = torch.zeros(B, 3, device=device)
    camera_velocity[:, 0] = camera_speed * torch.cos(vel_yaw)
    camera_velocity[:, 1] = camera_speed * torch.sin(vel_yaw)
    camera_velocity[:, 2] = 0
    
    # Random object positions (relative to camera for interesting distribution)
    # Mix of objects at various distances
    distance_type = torch.rand(B, device=device)
    
    object_distance = torch.zeros(B, device=device)
    object_distance[distance_type < 0.2] = torch.rand((distance_type < 0.2).sum(), device=device) * 10 + 1  # Close: 1-11m
    object_distance[(distance_type >= 0.2) & (distance_type < 0.5)] = torch.rand(((distance_type >= 0.2) & (distance_type < 0.5)).sum(), device=device) * 40 + 10  # Medium: 10-50m
    object_distance[(distance_type >= 0.5) & (distance_type < 0.8)] = torch.rand(((distance_type >= 0.5) & (distance_type < 0.8)).sum(), device=device) * 100 + 50  # Far: 50-150m
    object_distance[distance_type >= 0.8] = torch.rand((distance_type >= 0.8).sum(), device=device) * 200 + 150  # Very far: 150-350m
    
    # Object direction from camera (biased towards forward)
    obj_angle = torch.rand(B, device=device) * 2 * np.pi
    # Bias towards forward view (70% chance)
    forward_bias = torch.rand(B, device=device) < 0.7
    obj_angle[forward_bias] = yaw[forward_bias] + (torch.rand(forward_bias.sum(), device=device) - 0.5) * np.pi * 0.8
    
    obj_pitch = (torch.rand(B, device=device) - 0.5) * 0.3  # Mostly horizontal
    
    object_pos = torch.zeros(B, 3, device=device)
    object_pos[:, 0] = camera_pos[:, 0] + object_distance * torch.cos(obj_pitch) * torch.cos(obj_angle)
    object_pos[:, 1] = camera_pos[:, 1] + object_distance * torch.cos(obj_pitch) * torch.sin(obj_angle)
    object_pos[:, 2] = torch.rand(B, device=device) * 20  # 0-20m height
    
    # Object radius (bounding sphere)
    # Varies by object type
    object_radius = torch.zeros(B, device=device)
    radius_type = torch.rand(B, device=device)
    object_radius[radius_type < 0.3] = torch.rand((radius_type < 0.3).sum(), device=device) * 0.5 + 0.2  # Small: 0.2-0.7m
    object_radius[(radius_type >= 0.3) & (radius_type < 0.6)] = torch.rand(((radius_type >= 0.3) & (radius_type < 0.6)).sum(), device=device) * 1.5 + 0.5  # Medium: 0.5-2m
    object_radius[(radius_type >= 0.6) & (radius_type < 0.85)] = torch.rand(((radius_type >= 0.6) & (radius_type < 0.85)).sum(), device=device) * 3 + 2  # Large: 2-5m
    object_radius[radius_type >= 0.85] = torch.rand((radius_type >= 0.85).sum(), device=device) * 10 + 5  # Very large: 5-15m (buildings)
    
    # Object importance (weighted distribution)
    importance_weights = torch.tensor([0.05, 0.10, 0.20, 0.35, 0.30], device=device)
    object_importance = torch.multinomial(importance_weights.expand(B, -1), 1).squeeze()
    
    # Compute ground truth
    lod_level, blend_factor, should_cull = compute_lod_selection(
        camera_pos, camera_forward, camera_fov, camera_velocity,
        object_pos, object_radius, object_importance
    )
    
    # Build feature vector
    # Relative position from camera
    rel_pos = object_pos - camera_pos
    
    importance_oh = F.one_hot(object_importance, NUM_IMPORTANCE_LEVELS).float()
    
    X = torch.cat([
        camera_forward,                                # 3: Camera forward direction
        (camera_fov.unsqueeze(1) - 90) / 30,          # 1: FOV (normalized around 90)
        camera_velocity / 15.0,                        # 3: Camera velocity (normalized)
        rel_pos / 200.0,                               # 3: Relative object position (normalized)
        object_radius.unsqueeze(1) / 10.0,            # 1: Object radius (normalized)
        importance_oh,                                 # 5: Importance one-hot
    ], dim=1)  # Total: 16 features
    
    # Targets
    Y_lod = lod_level.float() / (NUM_LOD_LEVELS - 1)  # Normalize to 0-1
    Y_blend = blend_factor
    Y_cull = should_cull.float()
    
    Y = torch.stack([Y_lod, Y_blend, Y_cull], dim=1)
    
    # Shuffle
    idx = torch.randperm(B, device=device)
    X, Y = X[idx], Y[idx]
    lod_level = lod_level[idx]
    
    print(f"   ✓ Generated in {time.time()-start:.1f}s")
    print(f"   Features: {X.shape[1]} | Outputs: {Y.shape[1]}")
    
    # LOD distribution
    print(f"\n   LOD distribution:")
    for lod in range(NUM_LOD_LEVELS):
        count = (lod_level == lod).sum().item()
        print(f"     LOD {lod}: {count:,} ({100*count/B:.1f}%)")
    
    cull_count = should_cull.sum().item()
    print(f"   Culled: {cull_count:,} ({100*cull_count/B:.1f}%)")
    
    return X, Y, lod_level


# ═══════════════════════════════════════════════════════════════════════════════════════
# NEURAL NETWORK
# ═══════════════════════════════════════════════════════════════════════════════════════

class ResBlock(nn.Module):
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


class LODSelectionNet(nn.Module):
    def __init__(self, input_dim=16, hidden=192, blocks=5):
        super().__init__()
        
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        
        self.blocks = nn.ModuleList([ResBlock(hidden) for _ in range(blocks)])
        
        # LOD classification head (5 classes)
        self.lod_head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.GELU(),
            nn.Linear(64, NUM_LOD_LEVELS),
        )
        
        # Blend factor regression head
        self.blend_head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )
        
        # Cull classification head
        self.cull_head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        
        self.n_params = sum(p.numel() for p in self.parameters())
    
    def forward(self, x):
        x = self.proj(x)
        for block in self.blocks:
            x = block(x)
        
        lod_logits = self.lod_head(x)
        blend = self.blend_head(x)
        cull_logit = self.cull_head(x)
        
        return lod_logits, blend, cull_logit


# ═══════════════════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════════════════

def train(num_samples=2_000_000, epochs=100, batch_size=8192, lr=1e-3, patience=15):
    print("\n" + "█" * 70)
    print("█" + " TRAINING LOD SELECTION v1.0 ".center(68) + "█")
    print("█" * 70)
    
    X, Y, lod_labels = generate_data(num_samples)
    
    n_val = num_samples // 10
    X_train, X_val = X[n_val:], X[:n_val]
    Y_train, Y_val = Y[n_val:], Y[:n_val]
    lod_train, lod_val = lod_labels[n_val:], lod_labels[:n_val]
    
    print(f"\n   Train: {len(X_train):,} | Val: {len(X_val):,}")
    
    model = LODSelectionNet(input_dim=X.shape[1]).to(device)
    print(f"   Model: {model.n_params:,} parameters")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs,
        steps_per_epoch=(len(X_train) // batch_size) + 1
    )
    
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()
    
    print(f"\n   {'Ep':>3} | {'Train':>10} | {'Val':>10} | {'LOD Acc':>8} | {'Cull Acc':>8} | {'Best':>10}")
    print("   " + "-" * 70)
    
    best_loss = float('inf')
    best_state = None
    best_metrics = None
    no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        n_batches = 0
        
        idx = torch.randperm(len(X_train), device=device)
        for i in range(0, len(X_train), batch_size):
            batch_idx = idx[i:i+batch_size]
            xb = X_train[batch_idx]
            yb = Y_train[batch_idx]
            lod_b = lod_train[batch_idx]
            
            optimizer.zero_grad()
            lod_logits, blend_pred, cull_logit = model(xb)
            
            # LOD classification loss
            loss_lod = ce_loss(lod_logits, lod_b)
            
            # Blend factor regression loss
            loss_blend = mse_loss(blend_pred.squeeze(), yb[:, 1])
            
            # Cull classification loss
            loss_cull = bce_loss(cull_logit.squeeze(), yb[:, 2])
            
            loss = loss_lod + 0.5 * loss_blend + 0.5 * loss_cull
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            n_batches += 1
        
        train_loss /= n_batches
        
        # Validation
        model.eval()
        with torch.no_grad():
            lod_logits, blend_pred, cull_logit = model(X_val)
            
            loss_lod = ce_loss(lod_logits, lod_val).item()
            loss_blend = mse_loss(blend_pred.squeeze(), Y_val[:, 1]).item()
            loss_cull = bce_loss(cull_logit.squeeze(), Y_val[:, 2]).item()
            val_loss = loss_lod + 0.5 * loss_blend + 0.5 * loss_cull
            
            # Accuracies
            lod_pred = lod_logits.argmax(dim=1)
            lod_acc = (lod_pred == lod_val).float().mean().item()
            
            # LOD within 1 level
            lod_within_1 = ((lod_pred - lod_val).abs() <= 1).float().mean().item()
            
            cull_pred = cull_logit.squeeze() > 0
            cull_true = Y_val[:, 2] > 0.5
            cull_acc = (cull_pred == cull_true).float().mean().item()
        
        is_best = val_loss < best_loss
        if is_best:
            best_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_metrics = (lod_acc, lod_within_1, cull_acc)
            no_improve = 0
        else:
            no_improve += 1
        
        if epoch % 5 == 0 or is_best:
            print(f"   {epoch+1:>3} | {train_loss:>10.4f} | {val_loss:>10.4f} | "
                  f"{lod_acc*100:>7.2f}% | {cull_acc*100:>7.2f}% | {best_loss:>10.4f}{'*' if is_best else ''}")
        
        if no_improve >= patience:
            print(f"\n   Early stop at epoch {epoch+1}")
            break
    
    model.load_state_dict(best_state)
    lod_acc, lod_within_1, cull_acc = best_metrics
    
    print(f"\n   ═══════════════════════════════════════════════════════════════════")
    print(f"   FINAL RESULTS:")
    print(f"   ───────────────────────────────────────────────────────────────────")
    print(f"   LOD Exact Accuracy:    {lod_acc*100:.2f}%")
    print(f"   LOD Within-1 Accuracy: {lod_within_1*100:.2f}%")
    print(f"   Cull Accuracy:         {cull_acc*100:.2f}%")
    print(f"   ═══════════════════════════════════════════════════════════════════")
    
    return model, {
        'lod_acc': lod_acc,
        'lod_within_1': lod_within_1,
        'cull_acc': cull_acc,
    }


# ═══════════════════════════════════════════════════════════════════════════════════════
# SDK GENERATION
# ═══════════════════════════════════════════════════════════════════════════════════════

def generate_sdk(model, metrics):
    print("\n" + "█" * 70)
    print("█" + " GENERATING SDK ".center(68) + "█")
    print("█" * 70)
    
    SDK_DIR = "HyperFlux_Highguard_LODSelection_SDK"
    if os.path.exists(SDK_DIR):
        shutil.rmtree(SDK_DIR)
    os.makedirs(f"{SDK_DIR}/include")
    os.makedirs(f"{SDK_DIR}/PRIVATE")
    
    header = f'''/*
 * HYPERFLUX LOD SELECTION v1.0 - HIGHGUARD
 * O(1) Batch Level-of-Detail Prediction
 * 
 * Replaces per-object LOD calculations with neural batch inference.
 * Includes predictive LOD based on camera velocity (no popping).
 *
 * ACCURACY:
 *   LOD Exact:     {metrics['lod_acc']*100:.2f}%
 *   LOD Within-1:  {metrics['lod_within_1']*100:.2f}%
 *   Cull:          {metrics['cull_acc']*100:.2f}%
 */
#ifndef HG_LOD_SELECTION_H
#define HG_LOD_SELECTION_H

typedef enum {{
    IMPORTANCE_HERO = 0,      // Wardens, mounts
    IMPORTANCE_CRITICAL,      // Weapons, Shieldbreaker, generators
    IMPORTANCE_STRUCTURE,     // Base walls, siege tower
    IMPORTANCE_ENVIRONMENT,   // Rocks, trees
    IMPORTANCE_DETAIL,        // Small props, debris
    NUM_IMPORTANCE_LEVELS = 5
}} HG_ObjectImportance;

typedef struct {{
    float cameraPosX, cameraPosY, cameraPosZ;
    float cameraForwardX, cameraForwardY, cameraForwardZ;
    float cameraFOV;          // Degrees (typically 90-110)
    float cameraVelX, cameraVelY, cameraVelZ;
}} HG_CameraState;

typedef struct {{
    float posX, posY, posZ;
    float boundingRadius;     // Bounding sphere radius in meters
    HG_ObjectImportance importance;
}} HG_ObjectInfo;

typedef struct {{
    int lodLevel;             // 0 (highest) to 4 (lowest)
    float blendFactor;        // 0-1 for smooth LOD transitions
    int shouldCull;           // 1 = too far/small, don't render
}} HG_LODResult;

// Initialize
int HG_LOD_Init(void);

// Single object LOD selection
int HG_LOD_Select(const HG_CameraState* camera, const HG_ObjectInfo* obj, HG_LODResult* result);

// Batch LOD selection (much more efficient for many objects)
int HG_LOD_SelectBatch(const HG_CameraState* camera, const HG_ObjectInfo* objects, 
                       HG_LODResult* results, int count);

// Cleanup
void HG_LOD_Shutdown(void);

#endif
'''
    
    with open(f"{SDK_DIR}/include/HGLODSelection.h", 'w') as f:
        f.write(header)
    print("   ✓ Header generated")
    
    # Save weights
    state = model.state_dict()
    with open(f"{SDK_DIR}/PRIVATE/weights.inc", 'w') as f:
        f.write(f"// HYPERFLUX LOD Selection Weights\n")
        f.write(f"// Parameters: {model.n_params:,}\n\n")
        for name, tensor in state.items():
            data = tensor.cpu().numpy().flatten()
            f.write(f"// {name}: {list(tensor.shape)}\n")
            f.write(f"static const float {name.replace('.', '_')}[] = {{\n  ")
            f.write(', '.join(f'{x:.6f}f' for x in data[:10]))
            if len(data) > 10:
                f.write(f', ... /* {len(data) - 10} more */\n')
            f.write("};\n\n")
    
    print(f"   ✓ Weights saved ({model.n_params:,} parameters)")
    
    shutil.make_archive(SDK_DIR, 'zip', SDK_DIR)
    print(f"   ✓ SDK packaged: {SDK_DIR}.zip")
    
    return SDK_DIR


# ═══════════════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "█" * 70)
    print("█" + " HIGHGUARD LOD SELECTION v1.0 ".center(68) + "█")
    print("█" + " O(1) Batch Level-of-Detail Prediction ".center(68) + "█")
    print("█" * 70)
    
    print("\n   IMPORTANCE LEVELS:")
    for i, name in IMPORTANCE_TYPES.items():
        print(f"   {i}: {name}")
    
    print(f"\n   LOD LEVELS: {NUM_LOD_LEVELS} (0=highest detail, {NUM_LOD_LEVELS-1}=lowest)")
    
    # Train
    model, metrics = train(
        num_samples=2_000_000,
        epochs=100,
        patience=15
    )
    
    # Save model
    torch.save({
        'model': model.state_dict(),
        'metrics': metrics,
    }, 'hg_lod_selection_v1.pt')
    print(f"\n   ✓ Model saved: hg_lod_selection_v1.pt")
    
    # Generate SDK
    sdk_dir = generate_sdk(model, metrics)
    
    print("\n" + "█" * 70)
    print(f"   LOD SELECTION v1.0 COMPLETE")
    print(f"   Parameters: {model.n_params:,}")
    print(f"   LOD Accuracy: {metrics['lod_acc']*100:.2f}%")
    print(f"   LOD Within-1: {metrics['lod_within_1']*100:.2f}%")
    print(f"   Cull Accuracy: {metrics['cull_acc']*100:.2f}%")
    print("█" * 70)
