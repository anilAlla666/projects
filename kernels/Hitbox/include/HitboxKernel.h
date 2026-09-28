/*
 * HyperFlux Hitbox Kernel v2.2
 * O(1) Ray-Skeleton Collision Detection
 * 
 * Copyright (c) 2026 HyperFlux Neural Dynamics
 * All Rights Reserved - Proprietary and Confidential
 * 
 * This software is licensed, not sold. Unauthorized copying,
 * modification, or distribution is strictly prohibited.
 */

#ifndef HYPERFLUX_HITBOX_KERNEL_H
#define HYPERFLUX_HITBOX_KERNEL_H

#include <cstdint>
#include <cstddef>
#include <array>

namespace HyperFlux {
namespace Hitbox {

// Version info
constexpr const char* VERSION = "2.2.0";
constexpr const char* BUILD_DATE = "2026-01-12";

// Body part identifiers
enum class BodyPart : uint8_t {
    Head = 0,
    Chest = 1,
    Arms = 2,
    Legs = 3,
    Feet = 4,
    Miss = 255
};

// Pose identifiers
enum class Pose : uint8_t {
    Standing = 0,
    StandingMoving = 1,
    Crouching = 2,
    CrouchingMoving = 3,
    Prone = 4,
    ProneMoving = 5
};

// 3D Vector
struct Vec3 {
    float x, y, z;
    
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    
    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(float s) const { return {x * s, y * s, z * s}; }
    float dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    float norm() const;
    Vec3 normalized() const;
};

// Hit result structure
struct HitResult {
    bool hit;                    // Did the ray hit?
    BodyPart part;               // Which body part (if hit)
    float confidence;            // Confidence score [0, 1]
    float hit_distance;          // Distance to hit point (if applicable)
    Vec3 hit_point;              // World-space hit point (if applicable)
};

// Player state for collision check
struct PlayerState {
    Vec3 position;               // World position (feet)
    float yaw;                   // Facing direction (radians)
    Pose pose;                   // Current pose
};

// Ray definition
struct Ray {
    Vec3 origin;                 // Ray start point
    Vec3 direction;              // Normalized direction
};

// Main engine class
class Engine {
public:
    Engine();
    ~Engine();
    
    // Initialize the engine (call once at startup)
    bool Initialize();
    
    // Check if engine is ready
    bool IsReady() const;
    
    // Single ray-player collision check - O(1)
    HitResult CheckHit(const Ray& ray, const PlayerState& player) const;
    
    // Batch collision check - optimal for multiple rays
    void CheckHitBatch(
        const Ray* rays,
        const PlayerState& player,
        HitResult* results,
        size_t count
    ) const;
    
    // Batch against multiple players
    void CheckHitBatchMultiPlayer(
        const Ray* rays,
        size_t ray_count,
        const PlayerState* players,
        size_t player_count,
        HitResult* results  // [ray_count * player_count]
    ) const;
    
    // Get engine statistics
    struct Stats {
        uint64_t total_checks;
        uint64_t total_hits;
        double avg_inference_us;  // Microseconds per check
    };
    Stats GetStats() const;
    void ResetStats();
    
private:
    class Impl;
    Impl* impl_;
    
    // Disable copy
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
};

// Utility functions
const char* BodyPartToString(BodyPart part);
float GetDamageMultiplier(BodyPart part);  // Head=2.0, Chest=1.0, Arms=0.8, Legs=0.9, Feet=0.75

} // namespace Hitbox
} // namespace HyperFlux

#endif // HYPERFLUX_HITBOX_KERNEL_H
