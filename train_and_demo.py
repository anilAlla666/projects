#!/usr/bin/env python3
"""
╔═══════════════════════════════════════════════════════════════════════════════╗
║              HYPERFLUX BALLISTICS KERNEL - TRAINING & DEMO                    ║
║                                                                                ║
║  Run this script to:                                                           ║
║  1. Train all 5 segments on arcade ballistics                                  ║
║  2. Compile for O(1) inference                                                 ║
║  3. Benchmark against RK4 (target: 100x speedup)                              ║
║  4. Export for game engine integration                                         ║
║                                                                                ║
║  Usage: python train_and_demo.py                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝

Author: Neural Dynamics Team
"""

import torch
import numpy as np
import time
import json
from pathlib import Path

# Import the kernel
from arcade_ballistics_kernel import (
    HyperFluxBallisticsKernel,
    WeaponPresets,
    BatchShotInput,
    WorldGeometry,
    DEVICE
)


def train_full_kernel():
    """Train the complete ballistics kernel."""
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*15 + "HYPERFLUX BALLISTICS KERNEL TRAINING" + " "*17 + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    # Initialize
    kernel = HyperFluxBallisticsKernel()
    kernel.to(DEVICE)
    
    # Register weapons
    print("\n📋 Registering weapons...")
    m4_id = kernel.register_weapon(WeaponPresets.m4a1())
    ak_id = kernel.register_weapon(WeaponPresets.ak47())
    awp_id = kernel.register_weapon(WeaponPresets.awp())
    mp5_id = kernel.register_weapon(WeaponPresets.mp5())
    deagle_id = kernel.register_weapon(WeaponPresets.deagle())
    
    # Train on M4A1 (representative assault rifle)
    print("\n🎯 Training on M4A1 (arcade assault rifle profile)...")
    
    results = kernel.train_all_segments(
        weapon_id=m4_id,
        samples_per_segment=30000,  # Reduced for faster demo
        epochs=50  # Reduced for faster demo
    )
    
    # Compile for O(1) inference
    print("\n⚡ Compiling for O(1) inference...")
    kernel.compile_for_inference()
    
    # Benchmark
    print("\n🏁 Running benchmark...")
    benchmark_results = kernel.benchmark(weapon_id=m4_id, n_iterations=5000)
    
    # Save trained kernel
    save_path = Path("./trained_kernel.pt")
    kernel.save(str(save_path))
    
    return kernel, results, benchmark_results


def demo_gameplay_scenarios(kernel):
    """Demonstrate typical gameplay scenarios."""
    
    print("\n" + "="*70)
    print("  GAMEPLAY SCENARIO DEMOS")
    print("="*70)
    
    m4_id = 0  # M4A1
    
    # Scenario 1: Close range combat (0-50m)
    print("\n📍 Scenario 1: Close Range Combat (25m)")
    print("-"*50)
    
    origin = torch.tensor([0., 1.7, 0.], device=DEVICE)
    target_pos = torch.tensor([25., 1.5, 2.], device=DEVICE)
    direction = target_pos - origin
    direction = direction / torch.norm(direction)
    
    result = kernel.predict_single(origin, direction, m4_id)
    print(f"  Shot origin: ({origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f})")
    print(f"  Target: ({target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f})")
    print(f"  Predicted hit: ({result.hit_position[0]:.2f}, {result.hit_position[1]:.2f}, {result.hit_position[2]:.2f})")
    print(f"  Distance: {result.hit_distance:.2f}m")
    print(f"  Time of flight: {result.time_of_flight*1000:.2f}ms")
    print(f"  Damage: {result.damage:.1f}")
    print(f"  Segment used: S{result.segment_used}")
    
    # Scenario 2: Medium range engagement (100-200m)
    print("\n📍 Scenario 2: Medium Range Engagement (150m)")
    print("-"*50)
    
    target_pos = torch.tensor([150., 1.5, 10.], device=DEVICE)
    direction = target_pos - origin
    direction = direction / torch.norm(direction)
    
    result = kernel.predict_single(origin, direction, m4_id)
    print(f"  Target: ({target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f})")
    print(f"  Predicted hit: ({result.hit_position[0]:.2f}, {result.hit_position[1]:.2f}, {result.hit_position[2]:.2f})")
    print(f"  Distance: {result.hit_distance:.2f}m")
    print(f"  Bullet drop: {origin[1] - result.hit_position[1]:.3f}m")
    print(f"  Damage: {result.damage:.1f} (falloff applied)")
    print(f"  Segment used: S{result.segment_used}")
    
    # Scenario 3: Long range sniper shot (500m)
    print("\n📍 Scenario 3: Long Range Sniper Shot (500m)")
    print("-"*50)
    
    awp_id = 2  # AWP
    target_pos = torch.tensor([500., 1.5, 20.], device=DEVICE)
    direction = target_pos - origin
    direction = direction / torch.norm(direction)
    
    result = kernel.predict_single(origin, direction, awp_id)
    print(f"  Weapon: AWP")
    print(f"  Target: ({target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f})")
    print(f"  Predicted hit: ({result.hit_position[0]:.2f}, {result.hit_position[1]:.2f}, {result.hit_position[2]:.2f})")
    print(f"  Bullet drop: {origin[1] - result.hit_position[1]:.3f}m")
    print(f"  Time of flight: {result.time_of_flight*1000:.2f}ms")
    print(f"  Damage: {result.damage:.1f}")
    print(f"  Segment used: S{result.segment_used}")
    
    # Scenario 4: Moving target interception
    print("\n📍 Scenario 4: Moving Target Interception")
    print("-"*50)
    
    target_pos = torch.tensor([100., 1.5, 0.], device=DEVICE)
    target_vel = torch.tensor([0., 0., 5.], device=DEVICE)  # Moving sideways at 5 m/s
    
    aim_dir, intercept_time, possible = kernel.predict_interception(
        origin, direction, m4_id, target_pos, target_vel
    )
    
    if possible:
        intercept_point = target_pos + target_vel * intercept_time
        print(f"  Target position: ({target_pos[0]:.1f}, {target_pos[1]:.1f}, {target_pos[2]:.1f})")
        print(f"  Target velocity: ({target_vel[0]:.1f}, {target_vel[1]:.1f}, {target_vel[2]:.1f}) m/s")
        print(f"  Aim direction: ({aim_dir[0]:.3f}, {aim_dir[1]:.3f}, {aim_dir[2]:.3f})")
        print(f"  Intercept time: {intercept_time*1000:.2f}ms")
        print(f"  Intercept point: ({intercept_point[0]:.2f}, {intercept_point[1]:.2f}, {intercept_point[2]:.2f})")
    else:
        print(f"  Target too fast - interception not possible!")
    
    # Scenario 5: Batch prediction (spray pattern)
    print("\n📍 Scenario 5: Batch Prediction (10-shot spray)")
    print("-"*50)
    
    n_shots = 10
    origins = origin.unsqueeze(0).repeat(n_shots, 1)
    
    # Simulate spray pattern (slight random deviation)
    base_dir = torch.tensor([1., 0.05, 0.], device=DEVICE)
    base_dir = base_dir / torch.norm(base_dir)
    
    directions = []
    for i in range(n_shots):
        # Add recoil pattern
        recoil_up = 0.01 * i  # Increasing vertical recoil
        recoil_side = 0.005 * np.sin(i * 0.5)  # Side-to-side
        
        dir_i = base_dir + torch.tensor([0., recoil_up, recoil_side], device=DEVICE)
        dir_i = dir_i / torch.norm(dir_i)
        directions.append(dir_i)
    
    directions = torch.stack(directions)
    weapon_ids = torch.zeros(n_shots, dtype=torch.long, device=DEVICE)
    
    batch = BatchShotInput(
        origins=origins,
        directions=directions,
        weapon_ids=weapon_ids
    )
    
    t0 = time.perf_counter()
    batch_result = kernel.predict_batch(batch)
    t1 = time.perf_counter()
    
    print(f"  Shots fired: {n_shots}")
    print(f"  Batch prediction time: {(t1-t0)*1000:.3f}ms")
    print(f"  Time per shot: {(t1-t0)*1000/n_shots:.3f}ms")
    print(f"  Hit positions:")
    for i in range(n_shots):
        pos = batch_result.hit_position[i]
        print(f"    Shot {i+1}: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})")


def demo_world_geometry(kernel):
    """Demonstrate world geometry collision."""
    
    print("\n" + "="*70)
    print("  WORLD GEOMETRY COLLISION DEMO")
    print("="*70)
    
    # Create simple world geometry (a wall)
    wall_aabb = torch.tensor([
        [50., 0., -10., 51., 5., 10.],  # Wall at x=50
        [0., 0., 45., 100., 3., 46.],   # Wall at z=45
    ], device=DEVICE)
    
    geometry = WorldGeometry(aabb_bounds=wall_aabb)
    kernel.set_world_geometry(geometry)
    
    # Shot that hits wall
    print("\n📍 Shot toward wall (x=50):")
    origin = torch.tensor([0., 1.7, 0.], device=DEVICE)
    direction = torch.tensor([1., 0., 0.], device=DEVICE)
    
    result = kernel.predict_single(origin, direction, weapon_id=0, check_geometry=True)
    
    print(f"  Origin: ({origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f})")
    print(f"  Direction: ({direction[0]:.1f}, {direction[1]:.1f}, {direction[2]:.1f})")
    print(f"  Hit wall: {result.did_hit_geometry.item()}")
    if result.did_hit_geometry:
        print(f"  Wall hit point: ({result.hit_position[0]:.2f}, {result.hit_position[1]:.2f}, {result.hit_position[2]:.2f})")
    
    # Shot that misses wall
    print("\n📍 Shot that misses wall (angled up):")
    direction = torch.tensor([1., 0.5, 0.], device=DEVICE)
    direction = direction / torch.norm(direction)
    
    result = kernel.predict_single(origin, direction, weapon_id=0, check_geometry=True)
    
    print(f"  Direction: ({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f})")
    print(f"  Hit wall: {result.did_hit_geometry.item()}")
    print(f"  Final position: ({result.hit_position[0]:.2f}, {result.hit_position[1]:.2f}, {result.hit_position[2]:.2f})")


def export_for_game_engine(kernel):
    """Export kernel for game engine integration."""
    
    print("\n" + "="*70)
    print("  EXPORTING FOR GAME ENGINE")
    print("="*70)
    
    # Export to ONNX
    print("\n📦 Exporting to ONNX format...")
    
    # We export each segment separately for flexibility
    for seg_id in kernel.SEGMENT_CONFIG.keys():
        segment = kernel.segments[str(seg_id)]
        
        # Create dummy input
        dummy_input = torch.randn(1, kernel.input_dim, device=DEVICE)
        
        onnx_path = f"./ballistics_segment_{seg_id}.onnx"
        
        try:
            # Export just the forward pass components
            # (Full ONNX export requires more setup, this is demonstrative)
            print(f"  Segment {seg_id}: Would export to {onnx_path}")
            
        except Exception as e:
            print(f"  Segment {seg_id}: Export skipped ({e})")
    
    # Export inference matrices directly (for direct integration)
    print("\n📦 Exporting inference matrices...")
    
    matrices = {}
    for seg_id in kernel.SEGMENT_CONFIG.keys():
        segment = kernel.segments[str(seg_id)]
        if segment.inference_matrix is not None:
            matrices[f"segment_{seg_id}"] = {
                "inference_matrix": segment.inference_matrix.cpu().numpy().tolist(),
                "input_mean": segment.input_mean.cpu().numpy().tolist(),
                "input_std": segment.input_std.cpu().numpy().tolist(),
                "output_mean": segment.output_mean.cpu().numpy().tolist(),
                "output_std": segment.output_std.cpu().numpy().tolist(),
                "distance_range": [segment.distance_min, segment.distance_max],
                "max_error_mm": segment.max_error_mm
            }
    
    # Save as JSON for easy parsing in any engine
    with open("./ballistics_matrices.json", "w") as f:
        json.dump(matrices, f, indent=2)
    
    print("  Saved: ballistics_matrices.json")
    
    # Generate C++ header for direct integration
    print("\n📦 Generating C++ integration header...")
    
    cpp_header = '''// HyperFlux Ballistics Kernel - Auto-generated
// Neural Dynamics Team

#pragma once

#include <array>
#include <cmath>

namespace HyperFlux {

// Segment configuration
struct SegmentConfig {
    float distance_min;
    float distance_max;
    float max_error_mm;
};

constexpr std::array<SegmentConfig, 5> SEGMENTS = {{
    {0.0f, 100.0f, 5.0f},
    {101.0f, 300.0f, 10.0f},
    {301.0f, 600.0f, 20.0f},
    {601.0f, 1000.0f, 30.0f},
    {1001.0f, 1500.0f, 50.0f}
}};

// Select segment based on estimated range
inline int SelectSegment(float estimated_range) {
    for (int i = 0; i < 5; i++) {
        if (estimated_range >= SEGMENTS[i].distance_min && 
            estimated_range <= SEGMENTS[i].distance_max) {
            return i;
        }
    }
    return estimated_range > 1000.0f ? 4 : 0;
}

// Predict hit point (O(1) - single matrix multiplication)
// Input: encoded shot parameters [24]
// Output: hit position [3]
// Matrix: loaded from ballistics_matrices.json
void PredictHit(
    const float* input,
    const float* inference_matrix,  // [encoded_dim x 3]
    const float* input_mean,
    const float* input_std,
    const float* output_mean,
    const float* output_std,
    int input_dim,
    int encoded_dim,
    float* output
);

} // namespace HyperFlux
'''
    
    with open("./hyperflux_ballistics.h", "w") as f:
        f.write(cpp_header)
    
    print("  Saved: hyperflux_ballistics.h")
    print("\n✅ Export complete!")


def print_final_summary(results, benchmark_results):
    """Print final summary."""
    
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + " "*20 + "HYPERFLUX KERNEL - FINAL SUMMARY" + " "*16 + "█")
    print("█" + " "*68 + "█")
    print("█"*70)
    
    print("\n📊 SEGMENT ACCURACY:")
    print("-"*50)
    
    all_passed = True
    for seg_id, result in results.items():
        config = HyperFluxBallisticsKernel.SEGMENT_CONFIG[seg_id]
        status = "✅" if result["passed"] else "❌"
        all_passed = all_passed and result["passed"]
        print(f"  S{seg_id} ({config['range'][0]:4d}-{config['range'][1]:4d}m): "
              f"{result['final_mean_error_mm']:6.2f}mm / ±{config['error_mm']:2d}mm {status}")
    
    print("\n⚡ PERFORMANCE:")
    print("-"*50)
    print(f"  O(1) LNN inference: {benchmark_results['lnn_us_per_shot']:.2f} μs/shot")
    print(f"  O(N) RK4 baseline:  {benchmark_results['rk4_us_per_shot']:.2f} μs/shot")
    print(f"  SPEEDUP:            {benchmark_results['speedup']:.1f}x")
    
    target_achieved = benchmark_results['speedup'] >= 100
    print(f"\n  Target (100x): {'✅ ACHIEVED!' if target_achieved else '❌ Not yet'}")
    
    print("\n🎯 USE CASES:")
    print("-"*50)
    print(f"  ✓ Arcade FPS ballistics (COD-style)")
    print(f"  ✓ Batch prediction for spray patterns")
    print(f"  ✓ Moving target interception")
    print(f"  ✓ World geometry collision")
    
    print("\n📁 OUTPUT FILES:")
    print("-"*50)
    print(f"  trained_kernel.pt       - Full trained kernel")
    print(f"  ballistics_matrices.json - Inference matrices (JSON)")
    print(f"  hyperflux_ballistics.h  - C++ integration header")
    
    print("\n" + "█"*70)
    

def main():
    """Main entry point."""
    
    # Train the kernel
    kernel, results, benchmark_results = train_full_kernel()
    
    # Demo gameplay scenarios
    demo_gameplay_scenarios(kernel)
    
    # Demo world geometry
    demo_world_geometry(kernel)
    
    # Export for game engine
    export_for_game_engine(kernel)
    
    # Final summary
    print_final_summary(results, benchmark_results)
    
    print("\n🚀 HyperFlux Ballistics Kernel ready for integration!")
    print("   Contact: anilkumaralla@neuraldynamicsteam.io")
    print("   Website: neuraldynamicsteam.io")
    
    return kernel


if __name__ == "__main__":
    main()
