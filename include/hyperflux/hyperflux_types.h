// ==========================================================================
// HyperFlux SDK - Public Types
// Copyright (c) 2025-2026 Neural Dynamics. All rights reserved.
// ==========================================================================
#pragma once

#include <cstdint>
#include <cstddef>
#include "hyperflux_config.h"

namespace hyperflux {

// Error codes
enum class Status : int32_t {
    OK = 0,
    ERROR_NOT_INITIALIZED = -1,
    ERROR_INVALID_INPUT = -2,
    ERROR_INVALID_BATCH_SIZE = -3,
    ERROR_OUT_OF_MEMORY = -4,
    ERROR_SIMD_NOT_AVAILABLE = -5,
    ERROR_WEIGHT_CORRUPT = -6,
    ERROR_INTERNAL = -99,
};

// SIMD backend selection
enum class SimdBackend : uint32_t {
    AUTO = 0,       // Detect best available
    SCALAR = 1,     // Portable fallback
    AVX2 = 2,       // x86-64 AVX2
    AVX512 = 3,     // x86-64 AVX-512
    VNNI = 4,       // x86-64 AVX-512 VNNI
    NEON = 5,       // ARM64 NEON
};

// Library initialization config
struct InitConfig {
    SimdBackend preferred_backend = SimdBackend::AUTO;
    bool enable_profiling = false;
    bool enable_validation = false;     // Range-check inputs
    void* custom_allocator = nullptr;   // Optional custom alloc (nullptr = default)
};

// Kernel runtime statistics (available in profile mode)
struct KernelStats {
    double last_inference_us;       // Last single inference time (microseconds)
    double avg_inference_us;        // Running average
    double min_inference_us;
    double max_inference_us;
    uint64_t total_inferences;
    size_t model_size_bytes;        // INT8 model footprint
};

// ====================================================================
// Ballistics - Projectile trajectory computation
// ====================================================================
struct BallisticsInput {
    float origin[3];                         // Fire position (world space, meters)
    float direction[3];                      // Fire direction (normalized)
    float muzzle_velocity;                   // Initial speed (m/s)
    float projectile_mass;                   // Mass (kg)
    float drag_coefficient;                  // Aerodynamic drag Cd
    float gravity;                           // Gravitational acceleration (m/s^2)
    float wind[3];                           // Wind vector (m/s)
    float air_density;                       // Air density (kg/m^3)
    float spin_rate;                         // Projectile spin (RPM)
    float caliber;                           // Projectile diameter (m)
    float barrel_length;                     // Barrel length (m)
    float temperature;                       // Ambient temperature (C)
};

struct BallisticsOutput {
    float impact_point[3];                   // Predicted impact position
    float impact_velocity[3];                // Velocity at impact (m/s)
    float bullet_drop;                       // Vertical drop (m)
    float wind_drift;                        // Horizontal drift (m)
    float time_of_flight;                    // Flight duration (s)
    float kinetic_energy;                    // Energy at impact (J)
};

// ====================================================================
// Hitbox - Collision detection against skeletal meshes
// ====================================================================
struct HitboxInput {
    float ray_origin[3];                     // Ray start position (world space)
    float ray_direction[3];                  // Ray direction (normalized)
    float ray_length;                        // Maximum ray distance (m)
    float bone_transforms[16][3];            // Bone positions (world space, up to 16 bones)
    float bone_radii[16];                    // Bone capsule radii
    uint8_t active_bone_count;               // Number of active bones
};

struct HitboxOutput {
    float hit_probabilities[16];             // Per-bone hit probability (0-1)
};

// ====================================================================
// Visibility - Line-of-sight visibility determination
// ====================================================================
struct VisibilityInput {
    float observer_pos[3];                   // Observer position (world space)
    float observer_dir[3];                   // Observer facing direction
    float target_pos[3];                     // Target position (world space)
    float target_bounds[3];                  // Target bounding box half-extents
    float occluder_data[32];                 // Encoded occluder geometry (spatial hash)
    float fog_density;                       // Atmospheric fog density
    float smoke_alpha;                       // Smoke particle alpha
    float foliage_density;                   // Vegetation density between points
};

struct VisibilityOutput {
    float visibility;                        // Visibility probability (0=occluded, 1=visible)
};

// ====================================================================
// SpawnSelection - Optimal spawn point scoring and ranking
// ====================================================================
struct SpawnSelectionInput {
    float player_positions[64][2];           // Player XY positions (up to 64 players)
    float player_teams[64];                  // Team assignments (encoded)
    float player_alive[64];                  // Alive status per player
    float objective_positions[8][2];         // Objective XY positions
    float objective_states[8];               // Objective ownership/status
    float spawn_heat[20];                    // Recent spawn usage frequency
    float gunfight_zones[16][2];             // Active combat zone centers
    float gunfight_intensity[16];            // Combat intensity per zone
    float map_control[10];                   // Zone control percentages
    uint8_t player_count;                    // Active player count
    uint8_t requesting_team;                 // Spawning player team ID
};

struct SpawnSelectionOutput {
    float scores[20];                        // Score per spawn point (higher=better)
    uint8_t ranked[20];                      // Spawn indices sorted best-first
    uint8_t recommended;                     // Top spawn recommendation
};

// ====================================================================
// MountDynamics - Vehicle/mount physics state prediction
// ====================================================================
struct MountDynamicsInput {
    float position[3];                       // Current mount position (world space)
    float velocity[3];                       // Current velocity (m/s)
    float orientation[4];                    // Quaternion orientation
    float throttle;                          // Throttle input (0-1)
    float steering;                          // Steering input (-1 to 1)
    float brake;                             // Brake input (0-1)
    float terrain_normal[3];                 // Ground surface normal
    float terrain_friction;                  // Surface friction coefficient
    float terrain_slope;                     // Surface slope angle (rad)
    float mass;                              // Mount mass (kg)
    float engine_torque;                     // Engine output torque (Nm)
    float suspension_state[4];               // Per-wheel suspension compression
    float dt;                                // Time delta (s)
};

struct MountDynamicsOutput {
    float next_position[3];                  // Predicted next position
    float next_velocity[3];                  // Predicted next velocity
    float next_orientation[4];               // Predicted next orientation
    float next_angular_velocity[3];          // Angular velocity
    float wheel_slip[4];                     // Per-wheel slip ratio
    float suspension_force;                  // Total suspension force (N)
};

// ====================================================================
// Destruction - Destructible environment damage propagation
// ====================================================================
struct DestructionInput {
    float impact_point[3];                   // Damage impact position
    float impact_force[3];                   // Force direction and magnitude
    float impact_radius;                     // Damage radius (m)
    float material_strength;                 // Material structural strength
    float material_density;                  // Material density (kg/m^3)
    float material_brittleness;              // Brittleness factor (0-1)
    float current_health;                    // Current structural health (0-1)
    float connected_health[4];               // Health of adjacent elements
    float element_mass;                      // Element mass (kg)
    float element_size[3];                   // Element bounding dimensions
    float stress_history;                    // Accumulated stress
    float load_bearing;                      // Load-bearing factor (0-1)
    float explosive;                         // Is explosive damage (0 or 1)
};

struct DestructionOutput {
    float damage_result[4];                  // Damage propagation to adjacent elements
    float remaining_health;                  // Post-damage health (0-1)
    float fracture_probability;              // Probability of fracture event
    float debris_velocity;                   // Debris ejection speed (m/s)
    float collapse_risk;                     // Structural collapse probability
};

// ====================================================================
// LumenGI - Global illumination probe computation
// ====================================================================
struct LumenGIInput {
    float probe_position[3];                 // Probe world position
    float probe_normal[3];                   // Surface normal at probe
    float direct_light[4][4];                // Direct light sources (pos+color, up to 4)
    float sky_visibility[6];                 // Sky visibility per hemisphere direction
    float bounce_colors[8][3];               // Nearby surface bounce colors
    float bounce_distances[8];               // Distance to bounce surfaces
    float ao_factor;                         // Ambient occlusion
    float emissive_nearby[4][4];             // Nearby emissive sources (pos+intensity)
    float time_of_day;                       // Time-of-day factor (0-1)
    float indoor_factor;                     // Indoor vs outdoor (0-1)
};

struct LumenGIOutput {
    float irradiance_sh[27];                 // Spherical harmonics L2 (RGB × 9 coefficients)
    float dominant_direction[3];             // Dominant light direction
    float luminance;                         // Total estimated luminance
    float color_temperature;                 // Estimated color temperature
};

// ====================================================================
// AudioOcclusion - Sound propagation and occlusion
// ====================================================================
struct AudioOcclusionInput {
    float listener_pos[3];                   // Listener position
    float listener_dir[3];                   // Listener facing direction
    float source_pos[3];                     // Sound source position
    float source_radius;                     // Source emission radius
    float wall_data[8][3];                   // Nearby wall plane normals
    float wall_distances[8];                 // Distance to nearby walls
    float opening_data[4][3];                // Doorway/window positions
    float material_absorption[4];            // Wall absorption coefficients
    float room_volume;                       // Estimated room volume
    float outdoor_factor;                    // Indoor/outdoor blend (0-1)
};

struct AudioOcclusionOutput {
    float attenuation;                       // Overall volume attenuation (0-1)
    float occlusion;                         // Occlusion factor (0=blocked, 1=clear)
    float reverb_wet;                        // Reverb wet mix (0-1)
    float reverb_time;                       // Estimated RT60 (seconds)
    float eq_bands[4];                       // EQ adjustment per frequency band
    float direction[3];                      // Perceived sound direction (listener space)
    float spread;                            // Perceived source spread angle (rad)
};

// ====================================================================
// OcclusionCulling - GPU-free occlusion culling determination
// ====================================================================
struct OcclusionCullingInput {
    float camera_pos[3];                     // Camera world position
    float camera_dir[3];                     // Camera forward direction
    float camera_fov;                        // Field of view (radians)
    float camera_near;                       // Near plane distance
    float camera_far;                        // Far plane distance
    float object_positions[16][3];           // Object positions to test
    float object_radii[16];                  // Object bounding sphere radii
};

struct OcclusionCullingOutput {
    float visibility[16];                    // Per-object visibility (0=culled, 1=visible)
};

// ====================================================================
// LODSelection - Level-of-detail selection for mesh rendering
// ====================================================================
struct LODSelectionInput {
    float camera_pos[3];                     // Camera position
    float camera_dir[3];                     // Camera direction
    float object_pos[3];                     // Object position
    float object_radius;                     // Object bounding radius
    float screen_coverage;                   // Estimated screen area ratio
    float velocity_toward_camera;            // Approach speed (m/s)
    float importance;                        // Object importance weight
    float triangle_counts[8];                // Tri count per LOD level
    float gpu_budget_remaining;              // Remaining GPU triangle budget (normalized)
    float current_lod;                       // Current LOD level
};

struct LODSelectionOutput {
    float lod_scores[8];                     // Score per LOD level
};

} // namespace hyperflux
