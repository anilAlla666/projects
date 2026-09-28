/*
 * HYPERFLUX DESTRUCTION KERNEL v1.9 - IMPLEMENTATION
 * PROPRIETARY - DO NOT DISTRIBUTE THIS FILE
 */

#include "HyperFluxDestruction.h"
#include "destruction_weights.inc"
#include <cmath>

namespace HyperFlux {

static bool g_init = false;
static float g_thresh = THRESHOLD;

static inline float silu(float x) { return x / (1.0f + expf(-x)); }
static inline float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
static inline float dist3(const Vec3& a, const Vec3& b) {
    float dx = a.x-b.x, dy = a.y-b.y, dz = a.z-b.z;
    return sqrtf(dx*dx + dy*dy + dz*dz);
}

static void linear(const float* in, const float* W, const float* b, float* out, int in_d, int out_d) {
    for (int o = 0; o < out_d; o++) {
        float s = b[o];
        for (int i = 0; i < in_d; i++) s += W[o*in_d + i] * in[i];
        out[o] = s;
    }
}

static void ln(float* x, const float* g, const float* b, int n) {
    float m = 0, v = 0;
    for (int i = 0; i < n; i++) m += x[i];
    m /= n;
    for (int i = 0; i < n; i++) { float d = x[i]-m; v += d*d; }
    float inv = 1.0f / sqrtf(v/n + 1e-5f);
    for (int i = 0; i < n; i++) x[i] = (x[i]-m)*inv*g[i] + b[i];
}

static void silu_ip(float* x, int n) { for (int i = 0; i < n; i++) x[i] = silu(x[i]); }

static void preproc(const WallState& w, const DamageEvent& d, float* f) {
    static const float MR[] = {0.6f, 1.0f, 1.4f, 2.0f};
    float res = MR[(int)w.material];
    float dist = dist3(w.position, d.impactPosition);
    float dmg = 0;
    switch (d.damageType) {
        case DamageType::BLAST_HAMMER: if (dist < 3.0f) dmg = 1000.0f/res; break;
        case DamageType::ROCKET: if (dist < d.radius) dmg = d.damage*(1.0f-dist/d.radius*0.5f)/res; break;
        case DamageType::MELEE: dmg = d.damage/res; break;
        case DamageType::EXPLOSIVE: if (dist < d.radius) dmg = d.damage/res; break;
    }
    float r = dmg/(w.currentHP+1e-6f), dr = dist/(d.radius+0.1f), lr = logf(r+0.01f);
    r = fminf(fmaxf(r,0),5); dr = fminf(fmaxf(dr,0),3); lr = fminf(fmaxf(lr,-5),5);
    int i = 0;
    f[i++]=w.position.x/10; f[i++]=w.position.y/10; f[i++]=w.position.z/10;
    f[i++]=w.dimensions.x/5; f[i++]=w.dimensions.y/5; f[i++]=w.dimensions.z/5;
    for (int m=0;m<4;m++) f[i++]=((int)w.material==m)?1.0f:0.0f;
    f[i++]=w.currentHP/3000; f[i++]=w.maxHP/3000;
    f[i++]=d.impactPosition.x/10; f[i++]=d.impactPosition.y/10; f[i++]=d.impactPosition.z/10;
    f[i++]=d.impactDirection.x; f[i++]=d.impactDirection.y; f[i++]=d.impactDirection.z;
    for (int t=0;t<4;t++) f[i++]=((int)d.damageType==t)?1.0f:0.0f;
    f[i++]=d.damage/1500; f[i++]=d.radius/10;
    f[i++]=r; f[i++]=dr; f[i++]=lr; f[i++]=r-1.0f;
}

static void fwd(const float* f, float* o) {
    float b1[256], b2[256];
    float rt[4] = {f[24],f[25],f[26],f[27]};
    
    linear(rt, destroyed_net_0_weight, destroyed_net_0_bias, b1, 4, 64);
    ln(b1, destroyed_net_1_weight, destroyed_net_1_bias, 64); silu_ip(b1, 64);
    linear(b1, destroyed_net_3_weight, destroyed_net_3_bias, b2, 64, 64);
    ln(b2, destroyed_net_4_weight, destroyed_net_4_bias, 64); silu_ip(b2, 64);
    linear(b2, destroyed_net_6_weight, destroyed_net_6_bias, b1, 64, 32);
    ln(b1, destroyed_net_7_weight, destroyed_net_7_bias, 32); silu_ip(b1, 32);
    float dl; linear(b1, destroyed_net_9_weight, destroyed_net_9_bias, &dl, 32, 1);
    o[0] = sigmoid(dl);
    
    linear(f, physics_trunk_0_weight, physics_trunk_0_bias, b1, 28, 256);
    ln(b1, physics_trunk_1_weight, physics_trunk_1_bias, 256); silu_ip(b1, 256);
    linear(b1, physics_trunk_3_weight, physics_trunk_3_bias, b2, 256, 256);
    ln(b2, physics_trunk_4_weight, physics_trunk_4_bias, 256); silu_ip(b2, 256);
    linear(b2, physics_trunk_6_weight, physics_trunk_6_bias, b1, 256, 128);
    ln(b1, physics_trunk_7_weight, physics_trunk_7_bias, 128); silu_ip(b1, 128);
    float p[10]; linear(b1, physics_head_weight, physics_head_bias, p, 128, 10);
    o[1]=sigmoid(p[0]); o[2]=sigmoid(p[1]);
    for(int j=2;j<8;j++) o[j+1]=p[j];
    o[9]=fmaxf(0,p[8]); o[10]=sigmoid(p[9]);
}

bool DestructionKernel::Initialize() { g_init=true; return true; }
void DestructionKernel::Shutdown() { g_init=false; }
bool DestructionKernel::IsInitialized() { return g_init; }

DestructionResult DestructionKernel::Predict(const WallState& w, const DamageEvent& d) {
    DestructionResult r={};
    if (!g_init) return r;
    float f[28], o[11]; preproc(w,d,f); fwd(f,o);
    r.destroyed = o[0] > g_thresh;
    r.confidence = r.destroyed ? o[0] : (1.0f-o[0]);
    r.newHP = r.destroyed ? 0 : o[1]*3000;
    r.fragmentCount = o[2]*50;
    r.debrisCenter = Vec3(o[3]*10, o[4]*10, o[5]*10);
    r.avgVelocity = Vec3(o[6]*25, o[7]*25, o[8]*25);
    r.maxVelocity = o[9]*30;
    return r;
}

void DestructionKernel::PredictBatch(const WallState* w, const DamageEvent* d, size_t n, DestructionResult* r) {
    for (size_t i=0; i<n; i++) r[i] = Predict(w[i], d[i]);
}

float DestructionKernel::GetThreshold() { return g_thresh; }
void DestructionKernel::SetThreshold(float t) { g_thresh = t; }
const char* DestructionKernel::GetVersion() { return "1.9.0"; }

}
