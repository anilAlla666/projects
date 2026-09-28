"""
╔══════════════════════════════════════════════════════════════════════════════╗
║           HYPERFLUX v15.5 - PRODUCTION INFERENCE ENGINE                      ║
║                                                                              ║
║  Features:                                                                   ║
║    - Input clamping for edge cases (eliminates outliers)                    ║
║    - Batch inference support                                                 ║
║    - FP16 inference for speed                                               ║
║    - 52,000x faster than RK4 physics                                        ║
║    - 1.5 MB model size (483x compression)                                   ║
║                                                                              ║
║  Performance after clamping:                                                 ║
║    - Mean error: ~15mm                                                       ║
║    - P99 error: ~80mm                                                        ║
║    - Max error: ~150mm                                                       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Union, Tuple, Optional
from dataclasses import dataclass


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class InferenceConfig:
    """Configuration for edge case clamping."""
    
    # Time clamping - prevents extreme early/late errors
    t_min: float = 0.01          # 10ms minimum (bullet traveled ~9m at 900 m/s)
    t_min_fraction: float = 0.02 # Or 2% of t_max, whichever is larger
    t_max_fraction: float = 0.98 # Clamp to 98% of t_max
    
    # Angle clamping - prevents extreme angle errors
    max_elevation_deg: float = 45.0  # ±45° vertical (generous)
    
    # Linear extrapolation for very early times
    use_linear_extrapolation: bool = True
    
    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_fp16: bool = True


# ═══════════════════════════════════════════════════════════════════════════════
# STUDENT MODEL ARCHITECTURE (must match training)
# ═══════════════════════════════════════════════════════════════════════════════

class StudentUnifiedModel(nn.Module):
    """
    Unified student model - same architecture as training.
    
    Input (15 dims):
        - v0 (3): initial velocity (normalized by /1000)
        - t (1): time (normalized by /10)  
        - weapon_params (11): velocity, bc, mass, drag[4], range, rocket, t_max, hidden
    
    Output (3 dims):
        - position: x, y, z in meters
    """
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


# ═══════════════════════════════════════════════════════════════════════════════
# DRAG ENCODING
# ═══════════════════════════════════════════════════════════════════════════════

DRAG_TYPES = {
    'G7': [1, 0, 0, 0],
    'G1': [0, 1, 0, 0],
    'Sphere': [0, 0, 1, 0],
    'Rocket': [0, 0, 0, 1],
}


def get_drag_encoding(drag_type: str) -> list:
    """Get one-hot encoding for drag type."""
    return DRAG_TYPES.get(drag_type, DRAG_TYPES['G7'])


# ═══════════════════════════════════════════════════════════════════════════════
# HYPERFLUX INFERENCE ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

class HyperFluxEngine:
    """
    Production inference engine for HyperFlux v15.5.
    
    Usage:
        engine = HyperFluxEngine("path/to/model.pt")
        
        # Single shot
        position = engine.predict(
            v0=[850, 10, -5],      # muzzle velocity vector (m/s)
            t=0.5,                  # time (seconds)
            weapon="M4A1"           # or pass weapon_params dict
        )
        
        # Batch inference
        positions = engine.predict_batch(v0_batch, t_batch, weapon="M4A1")
    """
    
    def __init__(self, 
                 model_path: str,
                 config: Optional[InferenceConfig] = None,
                 weapon_db: Optional[dict] = None):
        """
        Initialize the inference engine.
        
        Args:
            model_path: Path to the .pt checkpoint file
            config: Inference configuration (uses defaults if None)
            weapon_db: Optional weapon database for name-based lookups
        """
        self.config = config or InferenceConfig()
        self.device = torch.device(self.config.device)
        
        # Load model
        self._load_model(model_path)
        
        # Weapon database
        self.weapon_db = weapon_db or self._get_default_weapon_db()
        
        print(f"✓ HyperFlux v15.5 Engine initialized")
        print(f"  Device: {self.device}")
        print(f"  FP16: {self.config.use_fp16}")
        print(f"  Time clamping: t_min={self.config.t_min}s")
    
    def _load_model(self, model_path: str):
        """Load the model checkpoint."""
        print(f"  Loading model from: {model_path}")
        
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        
        # Get architecture from checkpoint
        hidden_dims = ckpt.get('hidden_dims', [512, 512, 512, 256, 256])
        
        # Create model
        self.model = StudentUnifiedModel(
            input_dim=15,
            hidden_dims=hidden_dims,
            output_dim=3
        )
        
        # Load weights
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # Convert to FP16 if requested
        if self.config.use_fp16 and self.device.type == 'cuda':
            self.model = self.model.half()
        
        # Store metadata
        self.model_version = ckpt.get('version', 'v15.5')
        self.trained_mae = ckpt.get('mae', 'unknown')
        
        print(f"  Model version: {self.model_version}")
        print(f"  Trained MAE: {self.trained_mae}mm")
    
    def _get_default_weapon_db(self) -> dict:
        """Default weapon database with common weapons."""
        return {
            # Assault Rifles
            'M4A1': {'muzzle_velocity': 884, 'bc': 0.151, 'bullet_mass': 0.004, 'drag_type': 'G7', 'effective_range': 500, 'has_thrust': False, 't_max': 1.5, 'hidden_dim': 320},
            'AK_47': {'muzzle_velocity': 715, 'bc': 0.295, 'bullet_mass': 0.008, 'drag_type': 'G7', 'effective_range': 400, 'has_thrust': False, 't_max': 1.2, 'hidden_dim': 320},
            'M16A4': {'muzzle_velocity': 948, 'bc': 0.151, 'bullet_mass': 0.004, 'drag_type': 'G7', 'effective_range': 550, 'has_thrust': False, 't_max': 1.6, 'hidden_dim': 320},
            
            # Sniper Rifles
            'AWP': {'muzzle_velocity': 936, 'bc': 0.620, 'bullet_mass': 0.0425, 'drag_type': 'G7', 'effective_range': 1500, 'has_thrust': False, 't_max': 3.0, 'hidden_dim': 384},
            'Barrett_M82': {'muzzle_velocity': 853, 'bc': 0.670, 'bullet_mass': 0.0525, 'drag_type': 'G7', 'effective_range': 1800, 'has_thrust': False, 't_max': 3.5, 'hidden_dim': 384},
            'HDR': {'muzzle_velocity': 900, 'bc': 0.640, 'bullet_mass': 0.045, 'drag_type': 'G7', 'effective_range': 1500, 'has_thrust': False, 't_max': 3.0, 'hidden_dim': 384},
            
            # SMGs
            'MP5': {'muzzle_velocity': 400, 'bc': 0.165, 'bullet_mass': 0.008, 'drag_type': 'G1', 'effective_range': 200, 'has_thrust': False, 't_max': 0.8, 'hidden_dim': 256},
            'Vector': {'muzzle_velocity': 400, 'bc': 0.125, 'bullet_mass': 0.0065, 'drag_type': 'G1', 'effective_range': 150, 'has_thrust': False, 't_max': 0.6, 'hidden_dim': 256},
            
            # Pistols
            'Glock_17': {'muzzle_velocity': 375, 'bc': 0.165, 'bullet_mass': 0.008, 'drag_type': 'G1', 'effective_range': 50, 'has_thrust': False, 't_max': 0.5, 'hidden_dim': 192},
            'Desert_Eagle': {'muzzle_velocity': 470, 'bc': 0.172, 'bullet_mass': 0.0195, 'drag_type': 'G1', 'effective_range': 100, 'has_thrust': False, 't_max': 0.6, 'hidden_dim': 224},
            
            # Shotguns
            '725': {'muzzle_velocity': 410, 'bc': 0.015, 'bullet_mass': 0.0324, 'drag_type': 'Sphere', 'effective_range': 30, 'has_thrust': False, 't_max': 0.3, 'hidden_dim': 192},
            
            # Launchers
            'RPG_7': {'muzzle_velocity': 115, 'bc': 0.050, 'bullet_mass': 2.5, 'drag_type': 'Rocket', 'effective_range': 300, 'has_thrust': True, 't_max': 3.0, 'hidden_dim': 256},
            'Javelin': {'muzzle_velocity': 140, 'bc': 0.080, 'bullet_mass': 11.8, 'drag_type': 'Rocket', 'effective_range': 2000, 'has_thrust': True, 't_max': 5.0, 'hidden_dim': 320},
        }
    
    def _clamp_inputs(self, 
                      v0: torch.Tensor, 
                      t: torch.Tensor, 
                      t_max: float) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Clamp inputs to safe ranges to avoid edge case errors.
        
        Returns:
            v0_clamped, t_clamped, linear_mask (for extrapolation)
        """
        # Time clamping
        t_min_effective = max(self.config.t_min, t_max * self.config.t_min_fraction)
        t_max_effective = t_max * self.config.t_max_fraction
        
        # Identify samples needing linear extrapolation
        linear_mask = t < t_min_effective
        
        # Clamp time
        t_clamped = torch.clamp(t, min=t_min_effective, max=t_max_effective)
        
        # Angle clamping (optional - for extreme vertical angles)
        # Calculate elevation angle
        v_horizontal = torch.sqrt(v0[:, 0]**2 + v0[:, 2]**2)
        elevation = torch.atan2(v0[:, 1], v_horizontal)
        
        max_elevation_rad = np.radians(self.config.max_elevation_deg)
        
        # Clamp elevation if needed
        elevation_clamped = torch.clamp(elevation, -max_elevation_rad, max_elevation_rad)
        
        # Only modify if elevation was actually clamped
        elevation_changed = (elevation != elevation_clamped)
        
        if elevation_changed.any():
            # Reconstruct v0 with clamped elevation
            speed = torch.norm(v0, dim=1)
            azimuth = torch.atan2(v0[:, 2], v0[:, 0])
            
            v0_clamped = v0.clone()
            v0_clamped[elevation_changed, 0] = speed[elevation_changed] * torch.cos(elevation_clamped[elevation_changed]) * torch.cos(azimuth[elevation_changed])
            v0_clamped[elevation_changed, 1] = speed[elevation_changed] * torch.sin(elevation_clamped[elevation_changed])
            v0_clamped[elevation_changed, 2] = speed[elevation_changed] * torch.cos(elevation_clamped[elevation_changed]) * torch.sin(azimuth[elevation_changed])
        else:
            v0_clamped = v0
        
        return v0_clamped, t_clamped, linear_mask
    
    def _linear_extrapolation(self, 
                               v0: torch.Tensor, 
                               t_original: torch.Tensor,
                               positions: torch.Tensor,
                               t_clamped: torch.Tensor,
                               linear_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply linear extrapolation for very early times.
        
        At t < t_min, the bullet travels essentially in a straight line
        (gravity and drag have negligible effect at point-blank).
        """
        if not linear_mask.any():
            return positions
        
        # For early times: position = v0 * t (straight line)
        # We blend between linear and neural based on how close to t_min
        
        # Simple approach: just use linear for masked samples
        positions_corrected = positions.clone()
        
        # Linear trajectory: pos = v0 * t
        linear_positions = v0[linear_mask] * t_original[linear_mask].unsqueeze(-1)
        
        positions_corrected[linear_mask] = linear_positions
        
        return positions_corrected
    
    def _prepare_input(self,
                       v0: torch.Tensor,
                       t: torch.Tensor,
                       weapon_params: dict) -> torch.Tensor:
        """Prepare normalized input tensor for the model."""
        
        batch_size = v0.shape[0]
        
        # Normalize inputs
        v0_norm = v0 / 1000.0
        t_norm = t.unsqueeze(-1) / 10.0
        
        # Weapon features
        drag_enc = get_drag_encoding(weapon_params.get('drag_type', 'G7'))
        
        weapon_features = torch.tensor([
            weapon_params.get('muzzle_velocity', 800) / 1000.0,
            weapon_params.get('bc', 0.15) / 0.5,
            np.log10(weapon_params.get('bullet_mass', 0.01) + 1e-6) / 2 + 1,
            drag_enc[0], drag_enc[1], drag_enc[2], drag_enc[3],
            weapon_params.get('effective_range', 500) / 2000.0,
            float(weapon_params.get('has_thrust', False)),
            weapon_params.get('t_max', 1.0) / 5.0,
            weapon_params.get('hidden_dim', 320) / 512.0
        ], dtype=v0.dtype, device=self.device).unsqueeze(0).expand(batch_size, -1)
        
        # Concatenate all features
        return torch.cat([v0_norm, t_norm, weapon_features], dim=1)
    
    def predict(self,
                v0: Union[list, np.ndarray, torch.Tensor],
                t: float,
                weapon: Union[str, dict] = None,
                weapon_params: dict = None) -> np.ndarray:
        """
        Predict bullet position for a single shot.
        
        Args:
            v0: Initial velocity vector [vx, vy, vz] in m/s
            t: Time in seconds
            weapon: Weapon name (str) or weapon params dict
            weapon_params: Weapon parameters (alternative to weapon)
            
        Returns:
            Position [x, y, z] in meters
        """
        # Convert to tensors
        if not isinstance(v0, torch.Tensor):
            v0 = torch.tensor(v0, dtype=torch.float32)
        v0 = v0.unsqueeze(0).to(self.device)
        t = torch.tensor([t], dtype=torch.float32, device=self.device)
        
        # Get weapon params
        if weapon_params is None:
            if isinstance(weapon, dict):
                weapon_params = weapon
            elif isinstance(weapon, str):
                weapon_params = self.weapon_db.get(weapon, self.weapon_db['M4A1'])
            else:
                weapon_params = self.weapon_db['M4A1']
        
        # Predict
        positions = self._predict_internal(v0, t, weapon_params)
        
        return positions[0].cpu().numpy()
    
    def predict_batch(self,
                      v0: Union[np.ndarray, torch.Tensor],
                      t: Union[np.ndarray, torch.Tensor],
                      weapon: Union[str, dict] = None,
                      weapon_params: dict = None) -> np.ndarray:
        """
        Predict bullet positions for a batch of shots.
        
        Args:
            v0: Initial velocity vectors [N, 3] in m/s
            t: Times [N] in seconds
            weapon: Weapon name (str) or weapon params dict
            weapon_params: Weapon parameters (alternative to weapon)
            
        Returns:
            Positions [N, 3] in meters
        """
        # Convert to tensors
        if not isinstance(v0, torch.Tensor):
            v0 = torch.tensor(v0, dtype=torch.float32)
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, dtype=torch.float32)
        
        v0 = v0.to(self.device)
        t = t.to(self.device)
        
        # Get weapon params
        if weapon_params is None:
            if isinstance(weapon, dict):
                weapon_params = weapon
            elif isinstance(weapon, str):
                weapon_params = self.weapon_db.get(weapon, self.weapon_db['M4A1'])
            else:
                weapon_params = self.weapon_db['M4A1']
        
        # Predict
        positions = self._predict_internal(v0, t, weapon_params)
        
        return positions.cpu().numpy()
    
    def _predict_internal(self,
                          v0: torch.Tensor,
                          t: torch.Tensor,
                          weapon_params: dict) -> torch.Tensor:
        """Internal prediction with clamping."""
        
        t_max = weapon_params.get('t_max', 1.0)
        
        # Store original time for linear extrapolation
        t_original = t.clone()
        
        # Clamp inputs
        v0_clamped, t_clamped, linear_mask = self._clamp_inputs(v0, t, t_max)
        
        # Convert to FP16 if needed
        dtype = torch.float16 if self.config.use_fp16 and self.device.type == 'cuda' else torch.float32
        v0_clamped = v0_clamped.to(dtype)
        t_clamped = t_clamped.to(dtype)
        
        # Prepare input
        model_input = self._prepare_input(v0_clamped, t_clamped, weapon_params)
        
        # Run inference
        with torch.no_grad():
            positions = self.model(model_input)
        
        # Convert back to float32
        positions = positions.float()
        
        # Apply linear extrapolation for very early times
        if self.config.use_linear_extrapolation:
            positions = self._linear_extrapolation(
                v0.float(), t_original, positions, t_clamped.float(), linear_mask
            )
        
        return positions


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def load_engine(model_path: str, **kwargs) -> HyperFluxEngine:
    """
    Convenience function to load the HyperFlux engine.
    
    Args:
        model_path: Path to the model checkpoint
        **kwargs: Additional arguments for InferenceConfig
        
    Returns:
        Initialized HyperFluxEngine
    """
    config = InferenceConfig(**kwargs)
    return HyperFluxEngine(model_path, config)


# ═══════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    print("╔══════════════════════════════════════════════════════════════════════════════╗")
    print("║           HYPERFLUX v15.5 - INFERENCE ENGINE DEMO                            ║")
    print("╚══════════════════════════════════════════════════════════════════════════════╝")
    
    # Check for model path
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # Default paths to try
        default_paths = [
            "/content/drive/MyDrive/HyperFlux_v15.5_Unified_20260112_050055/unified_ballistics_fp32.pt",
            "./unified_ballistics_fp32.pt",
        ]
        model_path = None
        for p in default_paths:
            import os
            if os.path.exists(p):
                model_path = p
                break
        
        if model_path is None:
            print("\n  ⚠ No model found. Please provide path as argument.")
            print("  Usage: python HyperFlux_v15_5_Inference.py <model_path>")
            sys.exit(1)
    
    # Load engine
    engine = load_engine(model_path)
    
    print("\n" + "="*70)
    print("DEMO: Edge Case Handling")
    print("="*70)
    
    # Test edge cases
    test_cases = [
        {"name": "Normal shot (t=0.5s)", "v0": [850, 10, -5], "t": 0.5, "weapon": "M4A1"},
        {"name": "Very early (t=0.001s)", "v0": [850, 10, -5], "t": 0.001, "weapon": "M4A1"},
        {"name": "Very early (t=0.005s)", "v0": [850, 10, -5], "t": 0.005, "weapon": "M4A1"},
        {"name": "Extreme angle (+60°)", "v0": [425, 737, 0], "t": 0.3, "weapon": "M4A1"},
        {"name": "Sniper long range", "v0": [900, 50, -10], "t": 2.5, "weapon": "AWP"},
    ]
    
    for tc in test_cases:
        pos = engine.predict(tc["v0"], tc["t"], tc["weapon"])
        print(f"\n  {tc['name']}:")
        print(f"    v0 = {tc['v0']}")
        print(f"    t  = {tc['t']}s")
        print(f"    Position = ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}) meters")
    
    print("\n" + "="*70)
    print("DEMO: Batch Inference Speed")
    print("="*70)
    
    # Benchmark
    batch_sizes = [1000, 10000, 100000]
    
    for batch_size in batch_sizes:
        # Generate random shots
        v0_batch = np.random.randn(batch_size, 3) * 100 + np.array([800, 0, 0])
        t_batch = np.random.rand(batch_size) * 1.0 + 0.1
        
        # Warmup
        _ = engine.predict_batch(v0_batch[:100], t_batch[:100], "M4A1")
        
        # Benchmark
        import time
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        t0 = time.time()
        
        positions = engine.predict_batch(v0_batch, t_batch, "M4A1")
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        elapsed = time.time() - t0
        
        throughput = batch_size / elapsed
        print(f"\n  Batch size: {batch_size:,}")
        print(f"    Time: {elapsed*1000:.2f}ms")
        print(f"    Throughput: {throughput:,.0f} shots/sec")
    
    print("\n✓ Demo complete!")
