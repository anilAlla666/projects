"""
╔══════════════════════════════════════════════════════════════════════════════════════════╗
║                                                                                          ║
║                    HYPERFLUX KERNEL INSPECTOR                                            ║
║                    Find & Analyze All Highguard Assets on Drive                          ║
║                                                                                          ║
║  This script will:                                                                       ║
║    1. Scan Google Drive for all HyperFlux/Highguard related files                       ║
║    2. List .pt model files with their architecture details                              ║
║    3. List .npz INT8 quantized weight files                                             ║
║    4. List SDK folders and their contents                                               ║
║    5. Load each .pt and print exact input/output dimensions                             ║
║                                                                                          ║
╚══════════════════════════════════════════════════════════════════════════════════════════╝
"""

import os
import glob
from datetime import datetime
from collections import defaultdict

print("=" * 90)
print("  HYPERFLUX KERNEL INSPECTOR")
print("  Scanning Google Drive for all Highguard assets...")
print("=" * 90)

# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

DRIVE = "/content/drive/MyDrive"

# ═══════════════════════════════════════════════════════════════════════════════════════════
# STEP 1: FIND ALL HYPERFLUX/HIGHGUARD FOLDERS
# ═══════════════════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 90)
print("  STEP 1: SCANNING FOR HYPERFLUX/HIGHGUARD FOLDERS")
print("─" * 90)

# Search patterns
patterns = [
    f"{DRIVE}/HyperFlux*",
    f"{DRIVE}/Highguard*",
    f"{DRIVE}/highguard*",
    f"{DRIVE}/*Highguard*",
    f"{DRIVE}/*highguard*",
    f"{DRIVE}/*HyperFlux*",
]

all_folders = set()
for pattern in patterns:
    matches = glob.glob(pattern)
    for m in matches:
        if os.path.isdir(m):
            all_folders.add(m)

print(f"\n  Found {len(all_folders)} top-level folders:\n")
for folder in sorted(all_folders):
    folder_name = os.path.basename(folder)
    # Count files
    file_count = sum(1 for _ in glob.glob(f"{folder}/**/*", recursive=True) if os.path.isfile(_))
    print(f"    📁 {folder_name}/ ({file_count} files)")

# ═══════════════════════════════════════════════════════════════════════════════════════════
# STEP 2: FIND ALL .PT MODEL FILES
# ═══════════════════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 90)
print("  STEP 2: FINDING ALL .PT MODEL FILES")
print("─" * 90)

pt_files = []
for folder in all_folders:
    pt_files.extend(glob.glob(f"{folder}/**/*.pt", recursive=True))

# Also check root
pt_files.extend(glob.glob(f"{DRIVE}/*.pt"))

pt_files = sorted(set(pt_files), key=os.path.getmtime, reverse=True)

print(f"\n  Found {len(pt_files)} .pt files:\n")

for pt_file in pt_files:
    size_mb = os.path.getsize(pt_file) / (1024 * 1024)
    mtime = datetime.fromtimestamp(os.path.getmtime(pt_file)).strftime("%Y-%m-%d %H:%M")
    rel_path = pt_file.replace(DRIVE + "/", "")
    print(f"    📄 {rel_path}")
    print(f"       Size: {size_mb:.2f} MB | Modified: {mtime}")

# ═══════════════════════════════════════════════════════════════════════════════════════════
# STEP 3: FIND ALL .NPZ INT8 FILES
# ═══════════════════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 90)
print("  STEP 3: FINDING ALL .NPZ (INT8) FILES")
print("─" * 90)

npz_files = []
for folder in all_folders:
    npz_files.extend(glob.glob(f"{folder}/**/*.npz", recursive=True))

npz_files = sorted(set(npz_files), key=os.path.getmtime, reverse=True)

print(f"\n  Found {len(npz_files)} .npz files:\n")

for npz_file in npz_files:
    size_kb = os.path.getsize(npz_file) / 1024
    mtime = datetime.fromtimestamp(os.path.getmtime(npz_file)).strftime("%Y-%m-%d %H:%M")
    rel_path = npz_file.replace(DRIVE + "/", "")
    print(f"    📄 {rel_path}")
    print(f"       Size: {size_kb:.1f} KB | Modified: {mtime}")

# ═══════════════════════════════════════════════════════════════════════════════════════════
# STEP 4: FIND SDK FOLDERS (with .h, .cpp, .inc files)
# ═══════════════════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 90)
print("  STEP 4: FINDING SDK FOLDERS")
print("─" * 90)

sdk_folders = []
for folder in all_folders:
    # Check if it looks like an SDK (has .h or .inc files)
    h_files = glob.glob(f"{folder}/**/*.h", recursive=True)
    inc_files = glob.glob(f"{folder}/**/*.inc", recursive=True)
    if h_files or inc_files:
        sdk_folders.append(folder)

print(f"\n  Found {len(sdk_folders)} SDK folders:\n")

for sdk in sorted(sdk_folders):
    sdk_name = os.path.basename(sdk)
    h_count = len(glob.glob(f"{sdk}/**/*.h", recursive=True))
    cpp_count = len(glob.glob(f"{sdk}/**/*.cpp", recursive=True))
    inc_count = len(glob.glob(f"{sdk}/**/*.inc", recursive=True))
    a_count = len(glob.glob(f"{sdk}/**/*.a", recursive=True))
    
    print(f"    📁 {sdk_name}/")
    print(f"       .h: {h_count} | .cpp: {cpp_count} | .inc: {inc_count} | .a: {a_count}")

# ═══════════════════════════════════════════════════════════════════════════════════════════
# STEP 5: LOAD AND INSPECT EACH .PT FILE
# ═══════════════════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 90)
print("  STEP 5: LOADING .PT FILES TO INSPECT ARCHITECTURE")
print("─" * 90)

import torch
import numpy as np

def inspect_pt_file(filepath):
    """Load a .pt file and extract architecture information."""
    try:
        ckpt = torch.load(filepath, map_location='cpu', weights_only=False)
        
        info = {
            'path': filepath,
            'keys': list(ckpt.keys()) if isinstance(ckpt, dict) else ['raw_state_dict'],
            'has_model': False,
            'has_state_dict': False,
            'input_dim': None,
            'output_dim': None,
            'hidden_dims': [],
            'num_params': 0,
            'architecture': [],
            'metadata': {},
        }
        
        # Extract state dict
        state_dict = None
        if isinstance(ckpt, dict):
            # Check various keys
            for key in ['model_state_dict', 'state_dict', 'model', 'net']:
                if key in ckpt:
                    state_dict = ckpt[key]
                    info['has_state_dict'] = True
                    break
            
            # Extract metadata
            for key in ['accuracy', 'input_dim', 'output_dim', 'version', 'num_params', 'threshold']:
                if key in ckpt:
                    info['metadata'][key] = ckpt[key]
        else:
            # It's directly a state dict
            state_dict = ckpt
            info['has_state_dict'] = True
        
        if state_dict:
            # Analyze architecture from weight shapes
            layers = []
            for name, param in state_dict.items():
                if 'weight' in name.lower() and len(param.shape) == 2:
                    out_dim, in_dim = param.shape
                    layers.append({
                        'name': name,
                        'in': in_dim,
                        'out': out_dim,
                        'params': param.numel()
                    })
                info['num_params'] += param.numel()
            
            info['architecture'] = layers
            
            # Infer input/output dims
            if layers:
                info['input_dim'] = layers[0]['in']
                info['output_dim'] = layers[-1]['out']
                info['hidden_dims'] = [l['out'] for l in layers[:-1]]
        
        return info
        
    except Exception as e:
        return {
            'path': filepath,
            'error': str(e)
        }

def inspect_npz_file(filepath):
    """Load an .npz file and extract INT8 weight information."""
    try:
        data = np.load(filepath)
        
        info = {
            'path': filepath,
            'arrays': list(data.keys()),
            'shapes': {},
            'dtypes': {},
            'total_params': 0,
        }
        
        for key in data.keys():
            arr = data[key]
            info['shapes'][key] = arr.shape
            info['dtypes'][key] = str(arr.dtype)
            info['total_params'] += arr.size
        
        return info
        
    except Exception as e:
        return {
            'path': filepath,
            'error': str(e)
        }

# Inspect all .pt files
print("\n  Inspecting .pt files...\n")

kernel_info = {}

for pt_file in pt_files:
    info = inspect_pt_file(pt_file)
    filename = os.path.basename(pt_file)
    
    print(f"  ┌─ {filename}")
    
    if 'error' in info:
        print(f"  │  ❌ Error: {info['error']}")
    else:
        print(f"  │  Keys: {info['keys']}")
        
        if info['metadata']:
            print(f"  │  Metadata: {info['metadata']}")
        
        if info['input_dim'] and info['output_dim']:
            print(f"  │  Input dim:  {info['input_dim']}")
            print(f"  │  Output dim: {info['output_dim']}")
            print(f"  │  Hidden:     {info['hidden_dims']}")
        
        print(f"  │  Params:     {info['num_params']:,}")
        
        if info['architecture']:
            print(f"  │  Layers:")
            for layer in info['architecture'][:5]:  # Show first 5
                print(f"  │    {layer['name']}: {layer['in']} → {layer['out']}")
            if len(info['architecture']) > 5:
                print(f"  │    ... and {len(info['architecture']) - 5} more layers")
    
    print(f"  └─")
    print()
    
    # Store for summary
    kernel_info[pt_file] = info

# ═══════════════════════════════════════════════════════════════════════════════════════════
# STEP 6: INSPECT .NPZ INT8 FILES
# ═══════════════════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 90)
print("  STEP 6: INSPECTING .NPZ (INT8) FILES")
print("─" * 90)

print("\n  Inspecting .npz files...\n")

for npz_file in npz_files:
    info = inspect_npz_file(npz_file)
    filename = os.path.basename(npz_file)
    
    print(f"  ┌─ {filename}")
    
    if 'error' in info:
        print(f"  │  ❌ Error: {info['error']}")
    else:
        print(f"  │  Arrays: {len(info['arrays'])}")
        print(f"  │  Total params: {info['total_params']:,}")
        
        # Show first few arrays
        for key in list(info['arrays'])[:5]:
            shape = info['shapes'][key]
            dtype = info['dtypes'][key]
            print(f"  │    {key}: {shape} ({dtype})")
        
        if len(info['arrays']) > 5:
            print(f"  │    ... and {len(info['arrays']) - 5} more arrays")
    
    print(f"  └─")
    print()

# ═══════════════════════════════════════════════════════════════════════════════════════════
# STEP 7: SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 90)
print("  SUMMARY: KERNEL INVENTORY")
print("═" * 90)

# Try to match kernels by name
kernel_names = ['ballistics', 'hitbox', 'visibility', 'spawn', 'mount', 'destruction',
                'audio', 'lod', 'occlusion', 'lumen']

print("""
  ┌────────────────────┬────────────┬────────────┬────────────┬──────────────┬──────────────┐
  │ Kernel             │ .pt File   │ .npz INT8  │ SDK Folder │ Input Dim    │ Output Dim   │
  ├────────────────────┼────────────┼────────────┼────────────┼──────────────┼──────────────┤""")

for kname in kernel_names:
    # Find matching files
    pt_match = None
    npz_match = None
    sdk_match = None
    
    for pt in pt_files:
        if kname.lower() in pt.lower():
            pt_match = pt
            break
    
    for npz in npz_files:
        if kname.lower() in npz.lower():
            npz_match = npz
            break
    
    for sdk in sdk_folders:
        if kname.lower() in sdk.lower():
            sdk_match = sdk
            break
    
    # Get dimensions
    in_dim = "?"
    out_dim = "?"
    if pt_match and pt_match in kernel_info:
        info = kernel_info[pt_match]
        if info.get('input_dim'):
            in_dim = str(info['input_dim'])
        if info.get('output_dim'):
            out_dim = str(info['output_dim'])
    
    pt_status = "✅" if pt_match else "❌"
    npz_status = "✅" if npz_match else "❌"
    sdk_status = "✅" if sdk_match else "❌"
    
    print(f"  │ {kname.capitalize():18s} │ {pt_status:^10s} │ {npz_status:^10s} │ {sdk_status:^10s} │ {in_dim:^12s} │ {out_dim:^12s} │")

print("""  └────────────────────┴────────────┴────────────┴────────────┴──────────────┴──────────────┘
""")

# ═══════════════════════════════════════════════════════════════════════════════════════════
# STEP 8: SHOW MASTER SDK IF EXISTS
# ═══════════════════════════════════════════════════════════════════════════════════════════

print("\n" + "─" * 90)
print("  STEP 8: CHECKING FOR MASTER SDK")
print("─" * 90)

master_sdks = glob.glob(f"{DRIVE}/HyperFlux_Highguard_Master_*")
master_sdks = [m for m in master_sdks if os.path.isdir(m)]

if master_sdks:
    print(f"\n  Found {len(master_sdks)} Master SDK(s):\n")
    
    for sdk in sorted(master_sdks, key=os.path.getmtime, reverse=True):
        sdk_name = os.path.basename(sdk)
        mtime = datetime.fromtimestamp(os.path.getmtime(sdk)).strftime("%Y-%m-%d %H:%M")
        
        print(f"    📁 {sdk_name}/ (modified: {mtime})")
        
        # List contents
        for root, dirs, files in os.walk(sdk):
            level = root.replace(sdk, '').count(os.sep)
            indent = '    ' + '  ' * level
            subdir = os.path.basename(root)
            if level == 0:
                continue
            print(f"{indent}📁 {subdir}/")
            
            # Show files (limit to 5 per folder)
            subindent = '    ' + '  ' * (level + 1)
            for f in files[:5]:
                print(f"{subindent}📄 {f}")
            if len(files) > 5:
                print(f"{subindent}... and {len(files) - 5} more files")
else:
    print("\n  ❌ No Master SDK found")

# ═══════════════════════════════════════════════════════════════════════════════════════════
# DONE
# ═══════════════════════════════════════════════════════════════════════════════════════════

print("\n" + "═" * 90)
print("  INSPECTION COMPLETE")
print("═" * 90)
print("""
  Next steps:
  
  1. Check if all 10 kernels have .pt files
  2. Check if INT8 .npz exports exist for each
  3. Verify input/output dimensions match expected
  4. Identify which kernels need retraining or re-export
  
  Run this output back to Claude so we can build the correct SDK!
""")
