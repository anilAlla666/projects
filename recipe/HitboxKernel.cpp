/*
 * HyperFlux Hitbox Kernel v2.2 - IMPLEMENTATION
 * CONFIDENTIAL - DO NOT DISTRIBUTE
 */

#include "HitboxKernel.h"
#include <cmath>
#include <algorithm>
#include <cstring>

// Include weights
#include "hitbox_weights.inc"

namespace HyperFlux {
namespace Hitbox {

// ═══════════════════════════════════════════════════════════════════════════════
// Vec3 Implementation
// ═══════════════════════════════════════════════════════════════════════════════

float Vec3::norm() const {
    return std::sqrt(x * x + y * y + z * z);
}

Vec3 Vec3::normalized() const {
    float n = norm();
    if (n < 1e-8f) return {0, 0, 0};
    return {x / n, y / n, z / n};
}

// ═══════════════════════════════════════════════════════════════════════════════
// Neural Network Layers
// ═══════════════════════════════════════════════════════════════════════════════

inline float silu(float x) {
    return x / (1.0f + std::exp(-x));
}

inline float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

void layer_norm(float* x, const float* gamma, const float* beta, int dim) {
    float mean = 0, var = 0;
    for (int i = 0; i < dim; i++) mean += x[i];
    mean /= dim;
    for (int i = 0; i < dim; i++) var += (x[i] - mean) * (x[i] - mean);
    var /= dim;
    float std_inv = 1.0f / std::sqrt(var + 1e-5f);
    for (int i = 0; i < dim; i++) {
        x[i] = (x[i] - mean) * std_inv * gamma[i] + beta[i];
    }
}

void linear(const float* in, float* out, const float* W, const float* b, int in_dim, int out_dim) {
    for (int o = 0; o < out_dim; o++) {
        out[o] = b[o];
        for (int i = 0; i < in_dim; i++) {
            out[o] += in[i] * W[o * in_dim + i];
        }
    }
}

void linear_silu(const float* in, float* out, const float* W, const float* b, int in_dim, int out_dim) {
    linear(in, out, W, b, in_dim, out_dim);
    for (int i = 0; i < out_dim; i++) out[i] = silu(out[i]);
}

// ResBlock: out = silu(x + net(x))
void resblock(float* x, float* tmp, 
              const float* W1, const float* b1, const float* g1, const float* be1,
              const float* W2, const float* b2, const float* g2, const float* be2,
              int dim) {
    // tmp = Linear(x)
    linear(x, tmp, W1, b1, dim, dim);
    layer_norm(tmp, g1, be1, dim);
    for (int i = 0; i < dim; i++) tmp[i] = silu(tmp[i]);
    
    // tmp = Linear(tmp)
    float tmp2[512];  // Max dim
    linear(tmp, tmp2, W2, b2, dim, dim);
    layer_norm(tmp2, g2, be2, dim);
    
    // x = silu(x + tmp2)
    for (int i = 0; i < dim; i++) {
        x[i] = silu(x[i] + tmp2[i]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Geometric Feature Computation
// ═══════════════════════════════════════════════════════════════════════════════

void compute_geometric_features(
    const Ray& ray,
    const PlayerState& player,
    float* features  // Output: 16 features
) {
    const float PLAYER_RADIUS = 0.35f;
    
    // Height offset based on pose
    float height_off = 0.0f;
    if (player.pose == Pose::Crouching || player.pose == Pose::CrouchingMoving) {
        height_off = -0.35f;
    } else if (player.pose == Pose::Prone || player.pose == Pose::ProneMoving) {
        height_off = -1.40f;
    }
    
    // Player center
    Vec3 center = player.position;
    center.y = 0.9f + height_off;
    
    // Closest point on ray to player center
    Vec3 w = ray.origin - center;
    float t_closest = -w.dot(ray.direction);
    if (t_closest < 0) t_closest = 0;
    Vec3 closest = ray.origin + ray.direction * t_closest;
    
    // Distances
    float dist_to_center = (closest - center).norm();
    float horiz_dist = std::sqrt(
        (closest.x - center.x) * (closest.x - center.x) +
        (closest.z - center.z) * (closest.z - center.z)
    );
    float closest_height = closest.y - player.position.y;
    
    // Player height based on pose
    float player_height = 1.85f;
    if (player.pose == Pose::Crouching || player.pose == Pose::CrouchingMoving) {
        player_height = 1.50f;
    } else if (player.pose == Pose::Prone || player.pose == Pose::ProneMoving) {
        player_height = 0.45f;
    }
    
    // Cylinder check
    float cylinder_hit = (horiz_dist < PLAYER_RADIUS && closest_height >= 0 && closest_height <= player_height) ? 1.0f : 0.0f;
    
    // Direction features
    float cos_y = std::cos(player.yaw);
    float sin_y = std::sin(player.yaw);
    Vec3 player_forward = {sin_y, 0, cos_y};
    float front_back = ray.direction.dot(player_forward);
    
    Vec3 to_shooter = ray.origin - player.position;
    to_shooter.y = 0;
    float ts_norm = to_shooter.norm();
    if (ts_norm > 1e-8f) {
        to_shooter.x /= ts_norm;
        to_shooter.z /= ts_norm;
    }
    float alignment = -(ray.direction.x * to_shooter.x + ray.direction.z * to_shooter.z);
    
    float vertical_angle = std::acos(std::max(-1.0f, std::min(1.0f, ray.direction.y)));
    float shooter_dist = (ray.origin - player.position).norm();
    float t_normalized = t_closest / std::max(1.0f, shooter_dist);
    
    // Local coordinates
    float local_x = (closest.x - player.position.x) * cos_y - (closest.z - player.position.z) * sin_y;
    float local_z = (closest.x - player.position.x) * sin_y + (closest.z - player.position.z) * cos_y;
    
    // Height zone
    float height_zone = 0;
    if (closest_height > 0.15f) height_zone = 1.0f / 3.0f;
    if (closest_height > 0.95f) height_zone = 2.0f / 3.0f;
    if (closest_height > 1.55f) height_zone = 1.0f;
    
    // Cylinder margin
    float horiz_margin = horiz_dist - PLAYER_RADIUS;
    float vert_margin_top = closest_height - player_height;
    float vert_margin_bot = -closest_height;
    float cylinder_margin = std::max(horiz_margin, std::max(vert_margin_top, vert_margin_bot));
    
    // Pose encoding
    float pose_standing = (player.pose == Pose::Standing || player.pose == Pose::StandingMoving) ? 1.0f : 0.0f;
    float pose_crouch = (player.pose == Pose::Crouching || player.pose == Pose::CrouchingMoving) ? 1.0f : 0.0f;
    float pose_prone = (player.pose == Pose::Prone || player.pose == Pose::ProneMoving) ? 1.0f : 0.0f;
    
    // Output features
    features[0] = dist_to_center;
    features[1] = horiz_dist;
    features[2] = closest_height;
    features[3] = cylinder_hit;
    features[4] = front_back;
    features[5] = alignment;
    features[6] = vertical_angle / 3.14159265f;
    features[7] = shooter_dist / 50.0f;
    features[8] = t_normalized;
    features[9] = local_x;
    features[10] = local_z;
    features[11] = height_zone;
    features[12] = cylinder_margin;
    features[13] = pose_standing;
    features[14] = pose_crouch;
    features[15] = pose_prone;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Engine Implementation
// ═══════════════════════════════════════════════════════════════════════════════

class Engine::Impl {
public:
    bool ready = false;
    mutable uint64_t total_checks = 0;
    mutable uint64_t total_hits = 0;
    mutable double total_time_us = 0;
    
    // Inference buffers
    float geo_features[16];
    float pose_enc[24];
    float input_s1[40];
    float input_s2[43];
    float tmp1[128];
    float tmp2[128];
    float h1[128];
    float h2[64];
    
    bool RunHitClassifier(const float* input) const {
        // input: 40 dims -> output: hit probability
        float x[128];
        
        // Layer 0: Linear 40->128 + LayerNorm + SiLU
        linear(input, x, HIT_W0, HIT_B0, 40, 128);
        layer_norm(x, HIT_LN0_G, HIT_LN0_B, 128);
        for (int i = 0; i < 128; i++) x[i] = silu(x[i]);
        
        // ResBlock 1
        float tmp[128];
        resblock(x, tmp, HIT_RES1_W1, HIT_RES1_B1, HIT_RES1_LN1_G, HIT_RES1_LN1_B,
                 HIT_RES1_W2, HIT_RES1_B2, HIT_RES1_LN2_G, HIT_RES1_LN2_B, 128);
        
        // ResBlock 2
        resblock(x, tmp, HIT_RES2_W1, HIT_RES2_B1, HIT_RES2_LN1_G, HIT_RES2_LN1_B,
                 HIT_RES2_W2, HIT_RES2_B2, HIT_RES2_LN2_G, HIT_RES2_LN2_B, 128);
        
        // Layer: Linear 128->64 + LayerNorm + SiLU
        float x64[64];
        linear(x, x64, HIT_W3, HIT_B3, 128, 64);
        layer_norm(x64, HIT_LN3_G, HIT_LN3_B, 64);
        for (int i = 0; i < 64; i++) x64[i] = silu(x64[i]);
        
        // Final: Linear 64->1
        float logit = HIT_B4[0];
        for (int i = 0; i < 64; i++) logit += x64[i] * HIT_W4[i];
        
        return sigmoid(logit) > 0.5f;
    }
    
    int RunPartClassifier(const float* input) const {
        // input: 43 dims -> output: class 0-4
        float x[128];
        
        // Layer 0: Linear 43->128 + LayerNorm + SiLU
        linear(input, x, PART_W0, PART_B0, 43, 128);
        layer_norm(x, PART_LN0_G, PART_LN0_B, 128);
        for (int i = 0; i < 128; i++) x[i] = silu(x[i]);
        
        // ResBlock 1
        float tmp[128];
        resblock(x, tmp, PART_RES1_W1, PART_RES1_B1, PART_RES1_LN1_G, PART_RES1_LN1_B,
                 PART_RES1_W2, PART_RES1_B2, PART_RES1_LN2_G, PART_RES1_LN2_B, 128);
        
        // ResBlock 2
        resblock(x, tmp, PART_RES2_W1, PART_RES2_B1, PART_RES2_LN1_G, PART_RES2_LN1_B,
                 PART_RES2_W2, PART_RES2_B2, PART_RES2_LN2_G, PART_RES2_LN2_B, 128);
        
        // Layer: Linear 128->64 + LayerNorm + SiLU
        float x64[64];
        linear(x, x64, PART_W3, PART_B3, 128, 64);
        layer_norm(x64, PART_LN3_G, PART_LN3_B, 64);
        for (int i = 0; i < 64; i++) x64[i] = silu(x64[i]);
        
        // Final: Linear 64->5
        float logits[5];
        for (int c = 0; c < 5; c++) {
            logits[c] = PART_B4[c];
            for (int i = 0; i < 64; i++) logits[c] += x64[i] * PART_W4[c * 64 + i];
        }
        
        // Argmax
        int best = 0;
        for (int c = 1; c < 5; c++) {
            if (logits[c] > logits[best]) best = c;
        }
        return best;
    }
};

Engine::Engine() : impl_(new Impl()) {}
Engine::~Engine() { delete impl_; }

bool Engine::Initialize() {
    impl_->ready = true;
    return true;
}

bool Engine::IsReady() const {
    return impl_->ready;
}

HitResult Engine::CheckHit(const Ray& ray, const PlayerState& player) const {
    HitResult result;
    result.hit = false;
    result.part = BodyPart::Miss;
    result.confidence = 0;
    result.hit_distance = 0;
    result.hit_point = {0, 0, 0};
    
    if (!impl_->ready) return result;
    
    impl_->total_checks++;
    
    // Compute geometric features
    float geo[16];
    compute_geometric_features(ray, player, geo);
    
    // Build Stage 1 input: geo(16) + pose_enc(24) = 40
    float input_s1[40];
    std::memcpy(input_s1, geo, 16 * sizeof(float));
    std::memset(input_s1 + 16, 0, 24 * sizeof(float));
    int pose_idx = static_cast<int>(player.pose);
    if (pose_idx < 12) input_s1[16 + pose_idx] = 1.0f;
    
    // Run hit classifier
    bool hit = impl_->RunHitClassifier(input_s1);
    
    if (!hit) {
        return result;
    }
    
    result.hit = true;
    impl_->total_hits++;
    
    // Estimate hit point (simplified - use closest approach)
    Vec3 w = ray.origin - player.position;
    float t = -w.dot(ray.direction);
    if (t < 0) t = 0;
    result.hit_point = ray.origin + ray.direction * t;
    result.hit_distance = t;
    
    // Build Stage 2 input: geo(16) + pose_enc(24) + hit_point(3) = 43
    float input_s2[43];
    std::memcpy(input_s2, input_s1, 40 * sizeof(float));
    Vec3 local_hit = result.hit_point - player.position;
    input_s2[40] = local_hit.x;
    input_s2[41] = local_hit.y;
    input_s2[42] = local_hit.z;
    
    // Run part classifier
    int part_idx = impl_->RunPartClassifier(input_s2);
    result.part = static_cast<BodyPart>(part_idx);
    result.confidence = 0.95f;  // Simplified
    
    return result;
}

void Engine::CheckHitBatch(
    const Ray* rays,
    const PlayerState& player,
    HitResult* results,
    size_t count
) const {
    for (size_t i = 0; i < count; i++) {
        results[i] = CheckHit(rays[i], player);
    }
}

void Engine::CheckHitBatchMultiPlayer(
    const Ray* rays,
    size_t ray_count,
    const PlayerState* players,
    size_t player_count,
    HitResult* results
) const {
    for (size_t r = 0; r < ray_count; r++) {
        for (size_t p = 0; p < player_count; p++) {
            results[r * player_count + p] = CheckHit(rays[r], players[p]);
        }
    }
}

Engine::Stats Engine::GetStats() const {
    Stats s;
    s.total_checks = impl_->total_checks;
    s.total_hits = impl_->total_hits;
    s.avg_inference_us = impl_->total_checks > 0 ? impl_->total_time_us / impl_->total_checks : 0;
    return s;
}

void Engine::ResetStats() {
    impl_->total_checks = 0;
    impl_->total_hits = 0;
    impl_->total_time_us = 0;
}

const char* BodyPartToString(BodyPart part) {
    switch (part) {
        case BodyPart::Head: return "Head";
        case BodyPart::Chest: return "Chest";
        case BodyPart::Arms: return "Arms";
        case BodyPart::Legs: return "Legs";
        case BodyPart::Feet: return "Feet";
        default: return "Miss";
    }
}

float GetDamageMultiplier(BodyPart part) {
    switch (part) {
        case BodyPart::Head: return 2.0f;
        case BodyPart::Chest: return 1.0f;
        case BodyPart::Arms: return 0.8f;
        case BodyPart::Legs: return 0.9f;
        case BodyPart::Feet: return 0.75f;
        default: return 0.0f;
    }
}

} // namespace Hitbox
} // namespace HyperFlux
