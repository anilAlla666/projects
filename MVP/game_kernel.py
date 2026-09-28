import torch
import torch.nn as nn
import torch.optim as optim
import struct
import numpy as np
import time

# --- CONFIGURATION ---
print("🚀 HYPERFLUX PRO: PRECISION TRAINER")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"   [+] Hardware: {str(device).upper()}")

# --- 1. THE PHYSICS TRUTH (High Precision Mode) ---
def get_physics_batch(n_samples):
    # Inputs: Velocity [-50, 50], Time Jump [0.1s to 2.0s]
    vel = torch.randn(n_samples, 2, device=device) * 25.0
    dt_jump = torch.rand(n_samples, 1, device=device) * 2.0 
    
    # Physics Constants
    step_size = 0.01 
    gravity = torch.tensor([0.0, -9.81], device=device)
    wind = torch.tensor([1.5, 0.0], device=device) 
    
    target_pos = torch.zeros_like(vel)
    current_vel = vel.clone()
    
    # Simulating the curve
    steps = int(2.0 / step_size) 
    for i in range(steps):
        time_elapsed = i * step_size
        speed = torch.norm(current_vel, dim=1, keepdim=True)
        drag = -0.05 * current_vel * speed 
        accel = gravity + drag + wind
        
        active_mask = (time_elapsed < dt_jump).float()
        current_vel += accel * step_size * active_mask
        target_pos += current_vel * step_size * active_mask
        
    inputs = torch.cat([vel, dt_jump], dim=1) 
    return inputs, target_pos

# --- 2. THE LIQUID BRAIN (64 Neurons) ---
class LiquidCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.ff1 = nn.Linear(input_dim + hidden_dim, hidden_dim) 
        self.ff2 = nn.Linear(input_dim + hidden_dim, hidden_dim) 
        self.time_w = nn.Linear(hidden_dim, hidden_dim)
        self.time_a = nn.Linear(input_dim, hidden_dim)
        self.time_b = nn.Linear(input_dim, hidden_dim)

    def forward(self, x, h, dt):
        combined = torch.cat([x, h], dim=1)
        ff1_out = torch.tanh(self.ff1(combined))
        ff2_out = torch.sigmoid(self.ff2(combined))
        t_factor = torch.sigmoid(self.time_w(h) + self.time_a(x) * dt + self.time_b(x))
        h_new = ff2_out * t_factor * h + (1 - ff2_out * t_factor) * ff1_out
        return h_new

class HyperFluxPro(nn.Module):
    def __init__(self):
        super().__init__()
        self.units = 64 
        self.layer1 = LiquidCell(3, self.units) 
        self.layer2 = LiquidCell(self.units, self.units) 
        self.out = nn.Linear(self.units, 2)
        
    def forward(self, x):
        vel = x[:, :2]
        dt = x[:, 2].unsqueeze(1)
        h = torch.zeros(x.size(0), self.units, device=x.device)
        h1 = self.layer1(x, h, dt)
        h2 = self.layer2(h1, h1, dt) 
        return self.out(h2)

# --- 3. THE TRAINING LOOP (Accuracy Focused) ---
print("🧠 Training Deep Liquid Network...")
model = HyperFluxPro().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01) # Start High
# Aggressive Scheduler: If loss stops dropping for 100 steps, cut LR by 50%
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=100, verbose=True)
loss_fn = nn.MSELoss()

start_time = time.time()

for epoch in range(10001): # Long training for zero loss
    inputs, targets = get_physics_batch(5000)
    
    optimizer.zero_grad()
    preds = model(inputs)
    loss = loss_fn(preds, targets)
    loss.backward()
    optimizer.step()
    scheduler.step(loss) # Update LR based on performance
    
    if epoch % 1000 == 0:
        print(f"   Epoch {epoch}: Loss {loss.item():.9f}") # High precision print

print(f"✅ Final Loss: {loss.item():.9f}")

# --- 4. THE ACCURACY TEST (The Moment of Truth) ---
print("\n🔍 ACCURACY VALIDATION (Random Test Cases):")
print("-" * 65)
print(f"{'INPUT (Vx, Vy, Time)':<25} | {'TRUE DELTA':<15} | {'PREDICTED':<15} | {'ERROR':<10}")
print("-" * 65)

model.eval()
with torch.no_grad():
    # Test 5 random scenarios
    test_in, test_target = get_physics_batch(5)
    test_pred = model(test_in)
    
    for i in range(5):
        vx, vy = test_in[i, 0].item(), test_in[i, 1].item()
        t = test_in[i, 2].item()
        
        true_x, true_y = test_target[i, 0].item(), test_target[i, 1].item()
        pred_x, pred_y = test_pred[i, 0].item(), test_pred[i, 1].item()
        
        # Calculate Error Distance
        err = np.sqrt((true_x - pred_x)**2 + (true_y - pred_y)**2)
        
        print(f"[{vx:5.1f}, {vy:5.1f}, {t:3.1f}s]   | [{true_x:5.1f}, {true_y:5.1f}] | [{pred_x:5.1f}, {pred_y:5.1f}] | {err:.5f}")

print("-" * 65)
if loss.item() < 0.001:
    print("🏆 STATUS: SURGICAL PRECISION ACHIEVED.")
else:
    print("⚠️ STATUS: NEEDS MORE TRAINING.")

# --- 5. EXPORT ---
print("\n📦 Exporting to 'hyperflux_pro.bin'...")
weights = []
def add_layer(layer):
    weights.extend(layer.weight.detach().cpu().numpy().flatten())
    weights.extend(layer.bias.detach().cpu().numpy().flatten())

add_layer(model.layer1.ff1)
add_layer(model.layer1.ff2)
add_layer(model.layer1.time_w)
add_layer(model.layer1.time_a)
add_layer(model.layer1.time_b)
add_layer(model.layer2.ff1)
add_layer(model.layer2.ff2)
add_layer(model.layer2.time_w)
add_layer(model.layer2.time_a)
add_layer(model.layer2.time_b)
add_layer(model.out)

with open('hyperflux_pro.bin', 'wb') as f:
    f.write(struct.pack(f'{len(weights)}f', *weights))
print("✅ Done.")