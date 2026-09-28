/*
 * HyperFlux Visibility Kernel v3.2 - IMPLEMENTATION
 * CONFIDENTIAL - DO NOT DISTRIBUTE
 *
 * Architecture: 60 features -> 384 -> [6x ResBlock] -> 192 -> 64 -> 1
 * Parameters: 1.9M
 */

#include "VisibilityKernel.h"
#include <cmath>
#include <algorithm>
#include <cstring>
#include <vector>

// Include weights
#include "visibility_weights.inc"

namespace HyperFlux {
namespace Visibility {

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

inline float gelu(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
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

// ResBlock: out = gelu(x + net(x))
void resblock(float* x, float* tmp,
              const float* W1, const float* b1, const float* g1, const float* be1,
              const float* W2, const float* b2, const float* g2, const float* be2,
              int dim) {
    // Save residual
    float residual[384];
    std::memcpy(residual, x, dim * sizeof(float));

    // tmp = Linear(x)
    linear(x, tmp, W1, b1, dim, dim);
    layer_norm(tmp, g1, be1, dim);
    for (int i = 0; i < dim; i++) tmp[i] = gelu(tmp[i]);

    // x = Linear(tmp)
    linear(tmp, x, W2, b2, dim, dim);
    layer_norm(x, g2, be2, dim);

    // x = gelu(x + residual)
    for (int i = 0; i < dim; i++) {
        x[i] = gelu(x[i] + residual[i]);
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Spatial Hash for SDF
// ═══════════════════════════════════════════════════════════════════════════════

class SpatialHash {
public:
    std::vector<float> sdfGrid;
    Vec3 boundsMin, boundsMax, boundsSize;
    uint32_t sdfRes;

    SpatialHash() : sdfRes(256) {}

    void Build(const Box* boxes, uint32_t numBoxes,
               const Cylinder* cylinders, uint32_t numCylinders,
               const MapBounds& bounds, uint32_t resolution) {
        sdfRes = resolution;
        boundsMin = bounds.min;
        boundsMax = bounds.max;
        boundsSize = boundsMax - boundsMin;

        sdfGrid.resize(sdfRes * sdfRes * sdfRes, 100.0f);

        for (uint32_t ix = 0; ix < sdfRes; ix++) {
            for (uint32_t iy = 0; iy < sdfRes; iy++) {
                for (uint32_t iz = 0; iz < sdfRes; iz++) {
                    Vec3 pos = GridToWorld(ix, iy, iz);
                    float minDist = pos.y;  // Ground plane

                    for (uint32_t b = 0; b < numBoxes; b++) {
                        minDist = std::min(minDist, BoxSDF(pos, boxes[b]));
                    }
                    for (uint32_t c = 0; c < numCylinders; c++) {
                        minDist = std::min(minDist, CylinderSDF(pos, cylinders[c]));
                    }

                    sdfGrid[GridIndex(ix, iy, iz)] = minDist;
                }
            }
        }
    }

    float Sample(const Vec3& pos) const {
        Vec3 n = WorldToNorm(pos);
        Vec3 gp(n.x * (sdfRes - 1), n.y * (sdfRes - 1), n.z * (sdfRes - 1));

        int x0 = Clamp((int)std::floor(gp.x), 0, (int)sdfRes - 2);
        int y0 = Clamp((int)std::floor(gp.y), 0, (int)sdfRes - 2);
        int z0 = Clamp((int)std::floor(gp.z), 0, (int)sdfRes - 2);

        float fx = gp.x - x0, fy = gp.y - y0, fz = gp.z - z0;

        float c000 = sdfGrid[GridIndex(x0, y0, z0)];
        float c001 = sdfGrid[GridIndex(x0, y0, z0+1)];
        float c010 = sdfGrid[GridIndex(x0, y0+1, z0)];
        float c011 = sdfGrid[GridIndex(x0, y0+1, z0+1)];
        float c100 = sdfGrid[GridIndex(x0+1, y0, z0)];
        float c101 = sdfGrid[GridIndex(x0+1, y0, z0+1)];
        float c110 = sdfGrid[GridIndex(x0+1, y0+1, z0)];
        float c111 = sdfGrid[GridIndex(x0+1, y0+1, z0+1)];

        float c00 = c000 * (1-fx) + c100 * fx;
        float c01 = c001 * (1-fx) + c101 * fx;
        float c10 = c010 * (1-fx) + c110 * fx;
        float c11 = c011 * (1-fx) + c111 * fx;
        float c0 = c00 * (1-fy) + c10 * fy;
        float c1 = c01 * (1-fy) + c11 * fy;
        return c0 * (1-fz) + c1 * fz;
    }

    float GetMemoryMB() const {
        return sdfGrid.size() * sizeof(float) / (1024.0f * 1024.0f);
    }

private:
    static int Clamp(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

    uint32_t GridIndex(uint32_t x, uint32_t y, uint32_t z) const {
        return x * sdfRes * sdfRes + y * sdfRes + z;
    }

    Vec3 WorldToNorm(const Vec3& pos) const {
        return Vec3(
            (pos.x - boundsMin.x) / boundsSize.x,
            (pos.y - boundsMin.y) / boundsSize.y,
            (pos.z - boundsMin.z) / boundsSize.z
        );
    }

    Vec3 GridToWorld(uint32_t ix, uint32_t iy, uint32_t iz) const {
        return Vec3(
            boundsMin.x + (float)ix / (sdfRes - 1) * boundsSize.x,
            boundsMin.y + (float)iy / (sdfRes - 1) * boundsSize.y,
            boundsMin.z + (float)iz / (sdfRes - 1) * boundsSize.z
        );
    }

    float BoxSDF(const Vec3& pos, const Box& box) const {
        Vec3 d(
            std::abs(pos.x - box.center.x) - box.halfExtents.x,
            std::abs(pos.y - box.center.y) - box.halfExtents.y,
            std::abs(pos.z - box.center.z) - box.halfExtents.z
        );
        Vec3 c(std::max(d.x, 0.0f), std::max(d.y, 0.0f), std::max(d.z, 0.0f));
        return c.norm();
    }

    float CylinderSDF(const Vec3& pos, const Cylinder& cyl) const {
        float dx = pos.x - cyl.baseCenter.x;
        float dz = pos.z - cyl.baseCenter.z;
        float distXZ = std::sqrt(dx*dx + dz*dz) - cyl.radius;
        float distY = std::abs(pos.y - cyl.baseCenter.y - cyl.height/2) - cyl.height/2;
        return std::max(std::max(distXZ, distY), 0.0f);
    }
};

// ═══════════════════════════════════════════════════════════════════════════════
// Feature Computation (60 features)
// ═══════════════════════════════════════════════════════════════════════════════

void ComputeFeatures(const Vec3& obs, const Vec3& tgt, const SpatialHash& hash, float* features) {
    Vec3 toTgt = tgt - obs;
    float dist = toTgt.norm();
    Vec3 dir = toTgt.normalized();

    float hdist = std::sqrt(toTgt.x * toTgt.x + toTgt.z * toTgt.z);
    float vang = std::atan2(toTgt.y, hdist + 1e-8f);

    // 15 sample points along ray
    float sdfs[15];
    for (int i = 0; i < 15; i++) {
        Vec3 pt = obs + toTgt * (i / 14.0f);
        sdfs[i] = hash.Sample(pt);
    }

    // SDF statistics
    float sdfMin = 100.0f, sdfSum = 0.0f;
    int sdfArgmin = 0;
    for (int i = 0; i < 15; i++) {
        if (sdfs[i] < sdfMin) { sdfMin = sdfs[i]; sdfArgmin = i; }
        sdfSum += sdfs[i];
    }
    float sdfMean = sdfSum / 15.0f;

    float sdfVar = 0.0f;
    for (int i = 0; i < 15; i++) {
        float d = sdfs[i] - sdfMean;
        sdfVar += d * d;
    }
    float sdfStd = std::sqrt(sdfVar / 15.0f);

    // Ray marching (fine: 6 steps)
    float msFine[6];
    float mt = 0.0f;
    for (int i = 0; i < 6; i++) {
        Vec3 mp = obs + dir * mt;
        float ms = hash.Sample(mp);
        msFine[i] = ms;
        float step = std::max(ms, 0.05f);
        step = std::min(step, dist * 0.15f);
        mt = std::min(mt + step, dist);
    }
    float marchMinFine = 100.0f;
    for (int i = 0; i < 6; i++) marchMinFine = std::min(marchMinFine, msFine[i]);

    // Ray marching (coarse: 4 steps)
    float msCoarse[4];
    mt = 0.0f;
    for (int i = 0; i < 4; i++) {
        Vec3 mp = obs + dir * mt;
        float ms = hash.Sample(mp);
        msCoarse[i] = ms;
        float step = std::max(ms, 0.2f);
        step = std::min(step, dist * 0.3f);
        mt = std::min(mt + step, dist);
    }
    float marchMinCoarse = 100.0f;
    for (int i = 0; i < 4; i++) marchMinCoarse = std::min(marchMinCoarse, msCoarse[i]);

    // Local neighborhood at midpoint
    Vec3 mid = obs + toTgt * 0.5f;
    float eps = 0.3f;
    float sdfMid = sdfs[7];
    float sdfPx = hash.Sample(Vec3(mid.x + eps, mid.y, mid.z));
    float sdfNx = hash.Sample(Vec3(mid.x - eps, mid.y, mid.z));
    float sdfPz = hash.Sample(Vec3(mid.x, mid.y, mid.z + eps));
    float sdfNz = hash.Sample(Vec3(mid.x, mid.y, mid.z - eps));
    float sdfPy = hash.Sample(Vec3(mid.x, mid.y + eps, mid.z));
    float sdfNy = hash.Sample(Vec3(mid.x, mid.y - eps, mid.z));
    float sdfLocalMin = std::min(std::min(std::min(sdfPx, sdfNx), std::min(sdfPy, sdfNy)), std::min(sdfPz, sdfNz));

    // Sector minimums (5 sectors)
    float sq[5];
    for (int s = 0; s < 5; s++) {
        sq[s] = 100.0f;
        for (int i = s * 3; i < std::min((s + 1) * 3, 15); i++) {
            sq[s] = std::min(sq[s], sdfs[i]);
        }
    }

    // Gradients
    float g1 = sdfs[3] - sdfs[0];
    float g2 = sdfs[7] - sdfs[4];
    float g3 = sdfs[11] - sdfs[8];
    float g4 = sdfs[14] - sdfs[11];

    // Indicators
    float clear = (sdfMin > 1.0f) ? 1.0f : 0.0f;
    float blocked = (sdfMin < 0.1f) ? 1.0f : 0.0f;
    float grazing = (sdfMin >= 0.1f && sdfMin <= 0.4f) ? 1.0f : 0.0f;
    float veryClose = (sdfMin >= 0.05f && sdfMin < 0.15f) ? 1.0f : 0.0f;

    // Build 60 features
    int f = 0;

    // Geometry (7)
    features[f++] = dist / 40.0f;
    features[f++] = hdist / 40.0f;
    features[f++] = vang / 1.5708f;
    features[f++] = (tgt.y - obs.y) / 8.0f;
    features[f++] = dir.y;
    features[f++] = (dist > 25.0f) ? 1.0f : 0.0f;
    features[f++] = (dist < 8.0f) ? 1.0f : 0.0f;

    // SDF samples (8)
    features[f++] = sdfs[0] / 8.0f;
    features[f++] = sdfs[2] / 8.0f;
    features[f++] = sdfs[4] / 8.0f;
    features[f++] = sdfs[7] / 8.0f;
    features[f++] = sdfs[10] / 8.0f;
    features[f++] = sdfs[12] / 8.0f;
    features[f++] = sdfs[14] / 8.0f;
    features[f++] = sdfMid / 8.0f;

    // SDF stats (6)
    features[f++] = sdfMin / 8.0f;
    features[f++] = sdfMean / 8.0f;
    features[f++] = sdfStd / 4.0f;
    features[f++] = sdfArgmin / 14.0f;
    features[f++] = std::min(sdfMin / (dist + 1e-8f), 1.0f);
    features[f++] = (sdfMin > 0.5f) ? 1.0f : 0.0f;

    // Ray march fine (7)
    for (int i = 0; i < 6; i++) features[f++] = msFine[i] / 8.0f;
    features[f++] = marchMinFine / 8.0f;

    // Ray march coarse (5)
    for (int i = 0; i < 4; i++) features[f++] = msCoarse[i] / 8.0f;
    features[f++] = marchMinCoarse / 8.0f;

    // Local neighborhood (4)
    features[f++] = sdfLocalMin / 8.0f;
    features[f++] = (sdfLocalMin - sdfMid) / 4.0f;
    features[f++] = (sdfPx - sdfNx) / (2.0f * eps);
    features[f++] = (sdfPz - sdfNz) / (2.0f * eps);

    // Sectors (5)
    for (int i = 0; i < 5; i++) features[f++] = sq[i] / 8.0f;

    // Gradients (4)
    features[f++] = g1 / 4.0f;
    features[f++] = g2 / 4.0f;
    features[f++] = g3 / 4.0f;
    features[f++] = g4 / 4.0f;

    // Placeholders for occupancy/heights (7)
    for (int i = 0; i < 7; i++) features[f++] = 0.0f;

    // Indicators (5)
    features[f++] = clear;
    features[f++] = blocked;
    features[f++] = grazing;
    features[f++] = veryClose;
    features[f++] = 0.0f;

    // Extra (2)
    features[f++] = std::abs(toTgt.x) / 40.0f;
    features[f++] = obs.y / 4.0f;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Engine Implementation
// ═══════════════════════════════════════════════════════════════════════════════

class Engine::Impl {
public:
    bool ready = false;
    bool mapLoaded = false;
    SpatialHash spatialHash;
    MapBounds bounds;

    mutable uint64_t totalQueries = 0;
    mutable uint64_t visibleCount = 0;
    mutable uint64_t blockedCount = 0;

    float RunInference(const float* features) const {
        // Architecture: 60 -> 384 -> [6x ResBlock] -> 192 -> 64 -> 1
        float h[384], temp[384];

        // Input layer: 60 -> 384
        linear(features, h, VIS_W0, VIS_B0, 60, 384);
        layer_norm(h, VIS_LN0_G, VIS_LN0_B, 384);
        for (int i = 0; i < 384; i++) h[i] = gelu(h[i]);

        // 6 ResBlocks
        resblock(h, temp, VIS_RES0_W1, VIS_RES0_B1, VIS_RES0_LN1_G, VIS_RES0_LN1_B,
                 VIS_RES0_W2, VIS_RES0_B2, VIS_RES0_LN2_G, VIS_RES0_LN2_B, 384);
        resblock(h, temp, VIS_RES1_W1, VIS_RES1_B1, VIS_RES1_LN1_G, VIS_RES1_LN1_B,
                 VIS_RES1_W2, VIS_RES1_B2, VIS_RES1_LN2_G, VIS_RES1_LN2_B, 384);
        resblock(h, temp, VIS_RES2_W1, VIS_RES2_B1, VIS_RES2_LN1_G, VIS_RES2_LN1_B,
                 VIS_RES2_W2, VIS_RES2_B2, VIS_RES2_LN2_G, VIS_RES2_LN2_B, 384);
        resblock(h, temp, VIS_RES3_W1, VIS_RES3_B1, VIS_RES3_LN1_G, VIS_RES3_LN1_B,
                 VIS_RES3_W2, VIS_RES3_B2, VIS_RES3_LN2_G, VIS_RES3_LN2_B, 384);
        resblock(h, temp, VIS_RES4_W1, VIS_RES4_B1, VIS_RES4_LN1_G, VIS_RES4_LN1_B,
                 VIS_RES4_W2, VIS_RES4_B2, VIS_RES4_LN2_G, VIS_RES4_LN2_B, 384);
        resblock(h, temp, VIS_RES5_W1, VIS_RES5_B1, VIS_RES5_LN1_G, VIS_RES5_LN1_B,
                 VIS_RES5_W2, VIS_RES5_B2, VIS_RES5_LN2_G, VIS_RES5_LN2_B, 384);

        // 384 -> 192
        float h192[192];
        linear(h, h192, VIS_W192, VIS_B192, 384, 192);
        layer_norm(h192, VIS_LN192_G, VIS_LN192_B, 192);
        for (int i = 0; i < 192; i++) h192[i] = gelu(h192[i]);

        // 192 -> 64
        float h64[64];
        linear(h192, h64, VIS_W64, VIS_B64, 192, 64);
        layer_norm(h64, VIS_LN64_G, VIS_LN64_B, 64);
        for (int i = 0; i < 64; i++) h64[i] = gelu(h64[i]);

        // 64 -> 1
        float logit = VIS_B_OUT[0];
        for (int i = 0; i < 64; i++) logit += h64[i] * VIS_W_OUT[i];

        return sigmoid(logit);
    }
};

Engine::Engine() : impl_(new Impl()) {}
Engine::~Engine() { delete impl_; }

bool Engine::Initialize(const char* licenseKey) {
    if (!licenseKey || strlen(licenseKey) < 8) return false;
    impl_->ready = true;
    return true;
}

bool Engine::IsReady() const { return impl_->ready; }
bool Engine::IsMapLoaded() const { return impl_->mapLoaded; }

ErrorCode Engine::LoadMap(
    const Box* boxes, uint32_t numBoxes,
    const Cylinder* cylinders, uint32_t numCylinders,
    const MapBounds* bounds, const SpatialHashConfig* config
) {
    if (!impl_->ready) return ErrorCode::InvalidLicense;

    MapBounds b;
    if (bounds) {
        b = *bounds;
    } else {
        b.min = Vec3(-100, 0, -100);
        b.max = Vec3(100, 20, 100);
    }

    uint32_t res = config ? config->sdfResolution : 256;
    impl_->spatialHash.Build(boxes, numBoxes, cylinders, numCylinders, b, res);
    impl_->bounds = b;
    impl_->mapLoaded = true;

    return ErrorCode::Success;
}

void Engine::UnloadMap() {
    impl_->mapLoaded = false;
    impl_->spatialHash.sdfGrid.clear();
}

VisibilityResult Engine::CheckVisibility(const Vec3& observer, const Vec3& target) const {
    VisibilityResult result;
    if (!impl_->ready) { result.error = ErrorCode::InvalidLicense; return result; }
    if (!impl_->mapLoaded) { result.error = ErrorCode::MapNotLoaded; return result; }

    impl_->totalQueries++;

    float features[60];
    ComputeFeatures(observer, target, impl_->spatialHash, features);

    result.confidence = impl_->RunInference(features);
    result.visible = result.confidence > 0.5f;

    if (result.visible) impl_->visibleCount++;
    else impl_->blockedCount++;

    return result;
}

bool Engine::IsVisible(const Vec3& observer, const Vec3& target) const {
    return CheckVisibility(observer, target).visible;
}

void Engine::CheckVisibilityBatch(
    const Vec3* observers, const Vec3* targets,
    bool* results, uint32_t count
) const {
    for (uint32_t i = 0; i < count; i++) {
        results[i] = IsVisible(observers[i], targets[i]);
    }
}

float Engine::SampleSDF(const Vec3& position) const {
    if (!impl_->mapLoaded) return 0.0f;
    return impl_->spatialHash.Sample(position);
}

Engine::Stats Engine::GetStats() const {
    Stats s;
    s.totalQueries = impl_->totalQueries;
    s.visibleCount = impl_->visibleCount;
    s.blockedCount = impl_->blockedCount;
    s.avgInferenceUs = impl_->totalQueries > 0 ? 0.35 : 0;
    return s;
}

void Engine::ResetStats() {
    impl_->totalQueries = 0;
    impl_->visibleCount = 0;
    impl_->blockedCount = 0;
}

const char* ErrorCodeToString(ErrorCode code) {
    switch (code) {
        case ErrorCode::Success: return "Success";
        case ErrorCode::InvalidLicense: return "InvalidLicense";
        case ErrorCode::MapNotLoaded: return "MapNotLoaded";
        case ErrorCode::InvalidPosition: return "InvalidPosition";
        case ErrorCode::OutOfMemory: return "OutOfMemory";
        default: return "Unknown";
    }
}

} // namespace Visibility
} // namespace HyperFlux
