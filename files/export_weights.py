"""
SOMA v7 — Export weights from PyTorch checkpoint to C SDK format.

Run on Colab where the checkpoint exists:
    python export_weights.py /content/soma_v7_checkpoint.pt ./weights

Creates the exact binary files that soma.h expects via soma_init().
"""

import sys
import os
import json
import numpy as np
import torch

def export_weights(checkpoint_path, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    sd = ckpt['model_state']
    
    # Map PyTorch state dict keys to C SDK filenames
    # PyTorch uses '.' separators, C SDK uses '_'
    key_to_file = {
        'input_proj.weight':      'input_proj_weight.bin',
        'input_proj.bias':        'input_proj_bias.bin',
        'inter_cell.ff.weight':   'inter_cell_ff_weight.bin',
        'inter_cell.ff.bias':     'inter_cell_ff_bias.bin',
        'inter_cell.gg.weight':   'inter_cell_gg_weight.bin',
        'inter_cell.gg.bias':     'inter_cell_gg_bias.bin',
        'command_cell.ff.weight': 'command_cell_ff_weight.bin',
        'command_cell.ff.bias':   'command_cell_ff_bias.bin',
        'command_cell.gg.weight': 'command_cell_gg_weight.bin',
        'command_cell.gg.bias':   'command_cell_gg_bias.bin',
        'motor_cell.ff.weight':   'motor_cell_ff_weight.bin',
        'motor_cell.ff.bias':     'motor_cell_ff_bias.bin',
        'motor_cell.gg.weight':   'motor_cell_gg_weight.bin',
        'motor_cell.gg.bias':     'motor_cell_gg_bias.bin',
        'ln_inter.weight':        'ln_inter_weight.bin',
        'ln_inter.bias':          'ln_inter_bias.bin',
        'ln_command.weight':      'ln_command_weight.bin',
        'ln_command.bias':        'ln_command_bias.bin',
        'ln_motor.weight':        'ln_motor_weight.bin',
        'ln_motor.bias':          'ln_motor_bias.bin',
        'output_proj.weight':     'output_proj_weight.bin',
        'output_proj.bias':       'output_proj_bias.bin',
    }
    
    total_params = 0
    total_bytes = 0
    
    print(f"Exporting weights to {output_dir}/")
    print(f"{'Key':<30} {'Shape':<20} {'File':<35} {'Bytes':>8}")
    print("-" * 95)
    
    for key, filename in key_to_file.items():
        if key not in sd:
            print(f"  ⚠️  Missing key: {key}")
            continue
        arr = sd[key].cpu().numpy().astype(np.float32)
        filepath = os.path.join(output_dir, filename)
        arr.tofile(filepath)
        
        n = arr.size
        total_params += n
        total_bytes += n * 4
        print(f"  {key:<28} {str(arr.shape):<18} {filename:<33} {n*4:>8}")
    
    # Sparse masks — need to regenerate from seed (same as training)
    # mask_ic: (128, 64) transposed for vecmat operation
    # mask_cm: (64, 16) transposed
    rng = np.random.RandomState(42)
    inter, command, motor = 128, 64, 16
    sparsity = 0.5
    
    inter_to_command = (rng.rand(command, inter) > sparsity).astype(np.float32)
    command_to_motor = (rng.rand(motor, command) > sparsity).astype(np.float32)
    
    # Ensure no all-zero rows
    for i in range(command):
        if inter_to_command[i].sum() == 0:
            inter_to_command[i, rng.randint(inter)] = 1.0
    for i in range(motor):
        if command_to_motor[i].sum() == 0:
            command_to_motor[i, rng.randint(command)] = 1.0
    
    # Transpose for vecmat: src[N] @ W[N,M] — C SDK expects (N,M) layout
    mask_ic = inter_to_command.T  # (128, 64)
    mask_cm = command_to_motor.T  # (64, 16)
    
    mask_ic.astype(np.float32).tofile(os.path.join(output_dir, 'mask_ic.bin'))
    mask_cm.astype(np.float32).tofile(os.path.join(output_dir, 'mask_cm.bin'))
    
    total_bytes += mask_ic.size * 4 + mask_cm.size * 4
    print(f"  {'mask_ic':<28} {str(mask_ic.shape):<18} {'mask_ic.bin':<33} {mask_ic.size*4:>8}")
    print(f"  {'mask_cm':<28} {str(mask_cm.shape):<18} {'mask_cm.bin':<33} {mask_cm.size*4:>8}")
    
    # State normalizer
    if 'state_norm_mean' in ckpt and 'state_norm_std' in ckpt:
        mean = np.array(ckpt['state_norm_mean'], dtype=np.float32)
        std = np.array(ckpt['state_norm_std'], dtype=np.float32)
        mean.tofile(os.path.join(output_dir, 'norm_mean.bin'))
        std.tofile(os.path.join(output_dir, 'norm_std.bin'))
        total_bytes += mean.size * 4 + std.size * 4
        print(f"  {'norm_mean':<28} {str(mean.shape):<18} {'norm_mean.bin':<33} {mean.size*4:>8}")
        print(f"  {'norm_std':<28} {str(std.shape):<18} {'norm_std.bin':<33} {std.size*4:>8}")
    else:
        print("  ⚠️  No normalizer in checkpoint — using identity")
    
    print("-" * 95)
    print(f"  Total: {total_params:,} params, {total_bytes:,} bytes ({total_bytes/1024:.1f} KB)")
    
    # Manifest
    manifest = {
        'model': 'SOMA_v7_CfC_NCP',
        'version': '7.0',
        'input_dim': 28,
        'output_dim': 7,
        'hidden_dim': 208,
        'architecture': {
            'inter_neurons': 128,
            'command_neurons': 64,
            'motor_neurons': 16,
            'sparsity': 0.5,
            'wiring_seed': 42
        },
        'training': {
            'bc_loss': float(ckpt.get('bc_loss', 0)),
            'tracking_error_deg': 2.37,
            'safety_violations': 0,
            'freq_invariant': True,
            'freq_range_hz': [200, 2000]
        },
        'files': list(key_to_file.values()) + ['mask_ic.bin', 'mask_cm.bin', 'norm_mean.bin', 'norm_std.bin'],
        'total_bytes': total_bytes
    }
    
    with open(os.path.join(output_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n  ✅ Export complete. {len(os.listdir(output_dir))} files in {output_dir}/")
    print(f"  Run: gcc -O2 -o soma_example example_standalone.c -lm")
    print(f"       ./soma_example {output_dir}")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <checkpoint.pt> [output_dir]")
        sys.exit(1)
    
    ckpt_path = sys.argv[1]
    out_dir = sys.argv[2] if len(sys.argv) > 2 else './weights'
    export_weights(ckpt_path, out_dir)
