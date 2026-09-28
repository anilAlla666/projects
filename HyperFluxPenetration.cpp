/*
 * HyperFlux Penetration v2.0 - IMPLEMENTATION
 * CONFIDENTIAL - DO NOT DISTRIBUTE
 *
 * Architecture: 66 features -> 256 -> [4x ResBlock] -> 128 -> 5 outputs
 */

#include "HyperFluxPenetration.h"
#include "penetration_weights.inc"
#include <cmath>
#include <algorithm>
#include <cstring>

namespace HyperFlux { namespace Penetration {

// Bullet properties: pen, mass, diam, typical_vel, pen_mm_steel
static const float BULLET_PROPS[NUM_BULLET_TYPES][5] = {
    {0.25f, 0.008f, 9.0f, 360.0f, 3.0f},    // Pistol9mm
    {0.20f, 0.015f, 11.4f, 260.0f, 2.0f},   // Pistol45ACP
    {0.28f, 0.008f, 9.0f, 400.0f, 4.0f},    // SMG9mm
    {0.22f, 0.015f, 11.4f, 300.0f, 3.0f},   // SMG45ACP
    {0.55f, 0.004f, 5.56f, 940.0f, 8.0f},   // AR556
    {0.65f, 0.010f, 7.62f, 850.0f, 12.0f},  // AR762
    {0.58f, 0.004f, 5.56f, 915.0f, 9.0f},   // LMG556
    {0.68f, 0.010f, 7.62f, 820.0f, 13.0f},  // LMG762
    {0.75f, 0.012f, 7.62f, 900.0f, 15.0f},  // Sniper762
    {0.85f, 0.017f, 8.6f, 880.0f, 20.0f},   // Sniper338
    {1.00f, 0.045f, 12.7f, 900.0f, 25.0f},  // Sniper50BMG
    {0.12f, 0.003f, 8.4f, 400.0f, 1.0f},    // ShotgunBuck
    {0.35f, 0.028f, 18.5f, 450.0f, 5.0f},   // ShotgunSlug
};

// Material properties: density, hardness, deflection, resistance
static const float MAT_PROPS[NUM_MATERIAL_TYPES][4] = {
    {0.08f, 0.05f, 0.02f, 0.02f},   // Drywall
    {0.12f, 0.08f, 0.05f, 0.05f},   // Plywood
    {0.20f, 0.15f, 0.08f, 0.10f},   // WoodSolid
    {0.10f, 0.25f, 0.03f, 0.08f},   // Glass
    {0.30f, 0.40f, 0.15f, 0.25f},   // SheetMetal
    {0.25f, 0.35f, 0.12f, 0.20f},   // CarDoor
    {0.35f, 0.45f, 0.18f, 0.30f},   // CarBody
    {0.50f, 0.20f, 0.10f, 0.40f},   // Sandbags
    {0.40f, 0.12f, 0.08f, 0.25f},   // Dirt
    {0.55f, 0.60f, 0.20f, 0.50f},   // Brick
    {0.65f, 0.70f, 0.25f, 0.65f},   // ConcreteThin
    {0.80f, 0.80f, 0.30f, 0.80f},   // Concrete
    {0.45f, 0.55f, 0.05f, 0.60f},   // Kevlar
    {0.75f, 0.85f, 0.35f, 0.85f},   // SteelThin
    {0.95f, 0.95f, 0.40f, 1.00f},   // Steel
    {0.85f, 0.90f, 0.38f, 1.10f},   // Titanium
    {0.30f, 0.02f, 0.25f, 0.15f},   // Water
};

// Neural network functions
inline float gelu(float x) { return 0.5f*x*(1.0f+std::tanh(0.7978845608f*(x+0.044715f*x*x*x))); }
inline float sigmoid(float x) { return 1.0f/(1.0f+std::exp(-x)); }

void layer_norm(float* x, const float* g, const float* b, int d) {
    float m=0,v=0; for(int i=0;i<d;i++)m+=x[i]; m/=d;
    for(int i=0;i<d;i++)v+=(x[i]-m)*(x[i]-m); v/=d;
    float s=1.0f/std::sqrt(v+1e-5f);
    for(int i=0;i<d;i++)x[i]=g[i]*(x[i]-m)*s+b[i];
}

void linear(const float* in, float* out, const float* W, const float* b, int id, int od) {
    for(int o=0;o<od;o++){out[o]=b[o];for(int i=0;i<id;i++)out[o]+=in[i]*W[o*id+i];}
}

void gelu_inplace(float* x, int d) { for(int i=0;i<d;i++)x[i]=gelu(x[i]); }

void resblock(float* x, float* t, const float* W1, const float* b1, const float* g1, const float* be1,
              const float* W2, const float* b2, const float* g2, const float* be2, int d) {
    float r[256]; std::memcpy(r,x,d*sizeof(float));
    linear(x,t,W1,b1,d,d); layer_norm(t,g1,be1,d); gelu_inplace(t,d);
    linear(t,x,W2,b2,d,d); layer_norm(x,g2,be2,d);
    for(int i=0;i<d;i++)x[i]=gelu(x[i]+r[i]);
}

class Engine::Impl {
public:
    bool ready = false;
    mutable uint64_t queries = 0;
    mutable uint64_t pens = 0;
    mutable uint64_t stops = 0;
    
    void computeFeatures(BulletType bt, float vel, float ang, MaterialType mt, float thick, float* out) const {
        int b = (int)bt;
        int m = (int)mt;
        
        float bPen = BULLET_PROPS[b][0];
        float bMass = BULLET_PROPS[b][1];
        float bDiam = BULLET_PROPS[b][2];
        float bVelTyp = BULLET_PROPS[b][3];
        float bPenMm = BULLET_PROPS[b][4];
        
        float mDens = MAT_PROPS[m][0];
        float mHard = MAT_PROPS[m][1];
        float mDefl = MAT_PROPS[m][2];
        float mRes = MAT_PROPS[m][3];
        
        int f = 0;
        
        // Bullet one-hot (13)
        for(int i=0; i<NUM_BULLET_TYPES; i++) out[f++] = (i==b) ? 1.0f : 0.0f;
        
        // Bullet physics (4)
        out[f++] = bPen;
        out[f++] = bMass * 100.0f;
        out[f++] = bDiam / 15.0f;
        out[f++] = bPenMm / 25.0f;
        
        // Material one-hot (17)
        for(int i=0; i<NUM_MATERIAL_TYPES; i++) out[f++] = (i==m) ? 1.0f : 0.0f;
        
        // Material physics (4)
        out[f++] = mDens;
        out[f++] = mHard;
        out[f++] = mDefl;
        out[f++] = mRes;
        
        // Impact conditions (12)
        float velNorm = vel / 1000.0f;
        float velRatio = std::min(2.0f, vel / (bVelTyp + 1e-8f)) / 2.0f;
        float cosAng = std::cos(ang);
        float sinAng = std::sin(ang);
        float angNorm = ang / 1.5708f;
        float thickNorm = thick / 0.5f;
        float thickMm = thick * 1000.0f / 500.0f;
        float effThick = thick / std::max(0.1f, cosAng);
        float effThickNorm = std::min(2.0f, effThick / 0.5f) / 2.0f;
        
        out[f++] = velNorm;
        out[f++] = velRatio;
        out[f++] = cosAng;
        out[f++] = sinAng;
        out[f++] = angNorm;
        out[f++] = thickNorm;
        out[f++] = thickMm;
        out[f++] = effThickNorm;
        out[f++] = vel > 500.0f ? 1.0f : 0.0f;
        out[f++] = vel > 800.0f ? 1.0f : 0.0f;
        out[f++] = thick < 0.05f ? 1.0f : 0.0f;
        out[f++] = ang < 0.3f ? 1.0f : 0.0f;
        
        // Physics-derived (16)
        float effThickMm = effThick * 1000.0f;
        float steelEquiv = effThickMm * mRes;
        float velFactor = std::min(1.5f, std::max(0.3f, vel / (bVelTyp + 1e-8f)));
        float effPenMm = bPenMm * velFactor;
        float penMargin = effPenMm - steelEquiv;
        float penMarginNorm = std::max(-1.0f, std::min(1.0f, penMargin / 20.0f));
        float ke = 0.5f * bMass * vel * vel;
        float keNorm = std::min(2.0f, ke / 20000.0f);
        float mom = bMass * vel;
        float momNorm = std::min(2.0f, mom / 50.0f);
        
        out[f++] = keNorm;
        out[f++] = momNorm;
        out[f++] = bPenMm / 25.0f;
        out[f++] = steelEquiv / 50.0f;
        out[f++] = effPenMm / 25.0f;
        out[f++] = penMarginNorm;
        out[f++] = penMargin > 0 ? 1.0f : 0.0f;
        out[f++] = penMargin > 5 ? 1.0f : 0.0f;
        out[f++] = penMargin < -5 ? 1.0f : 0.0f;
        out[f++] = velFactor;
        out[f++] = mRes;
        out[f++] = bPenMm > steelEquiv ? 1.0f : 0.0f;
        out[f++] = effPenMm > steelEquiv * 1.5f ? 1.0f : 0.0f;
        out[f++] = effPenMm < steelEquiv * 0.5f ? 1.0f : 0.0f;
        out[f++] = bPen / (mHard + 0.1f);
        out[f++] = (1.0f - angNorm) * bPen;
    }
    
    void forward(const float* feat, float* out) const {
        const int H = 256;
        float t1[H], t2[H];
        
        // Input: 66 -> 256
        linear(feat, t1, input_0_weight, input_0_bias, 66, H);
        layer_norm(t1, input_1_weight, input_1_bias, H);
        gelu_inplace(t1, H);
        
        // ResBlocks
        resblock(t1, t2, blocks_0_net_0_weight, blocks_0_net_0_bias, blocks_0_net_1_weight, blocks_0_net_1_bias,
                 blocks_0_net_4_weight, blocks_0_net_4_bias, blocks_0_net_5_weight, blocks_0_net_5_bias, H);
        resblock(t1, t2, blocks_1_net_0_weight, blocks_1_net_0_bias, blocks_1_net_1_weight, blocks_1_net_1_bias,
                 blocks_1_net_4_weight, blocks_1_net_4_bias, blocks_1_net_5_weight, blocks_1_net_5_bias, H);
        resblock(t1, t2, blocks_2_net_0_weight, blocks_2_net_0_bias, blocks_2_net_1_weight, blocks_2_net_1_bias,
                 blocks_2_net_4_weight, blocks_2_net_4_bias, blocks_2_net_5_weight, blocks_2_net_5_bias, H);
        resblock(t1, t2, blocks_3_net_0_weight, blocks_3_net_0_bias, blocks_3_net_1_weight, blocks_3_net_1_bias,
                 blocks_3_net_4_weight, blocks_3_net_4_bias, blocks_3_net_5_weight, blocks_3_net_5_bias, H);
        
        // Output: 256 -> 128 -> 5
        float h128[128];
        linear(t1, h128, output_0_weight, output_0_bias, H, 128);
        layer_norm(h128, output_1_weight, output_1_bias, 128);
        gelu_inplace(h128, 128);
        
        float logits[5];
        linear(h128, logits, output_3_weight, output_3_bias, 128, 5);
        for(int i=0; i<5; i++) out[i] = sigmoid(logits[i]);
    }
};

Engine::Engine() : impl_(new Impl()) {}
Engine::~Engine() { delete impl_; }
bool Engine::Initialize(const char* k) { if(!k||strlen(k)<8)return false; impl_->ready=true; return true; }
bool Engine::IsReady() const { return impl_->ready; }

ErrorCode Engine::Calculate(BulletType bt, float vel, float ang, MaterialType mt, float thick, PenetrationResult& r) const {
    if(!impl_->ready) return ErrorCode::InvalidLicense;
    if((int)bt >= NUM_BULLET_TYPES) return ErrorCode::InvalidBulletType;
    if((int)mt >= NUM_MATERIAL_TYPES) return ErrorCode::InvalidMaterialType;
    
    impl_->queries++;
    
    float feat[66], out[5];
    impl_->computeFeatures(bt, vel, ang, mt, thick, feat);
    impl_->forward(feat, out);
    
    r.confidence = out[0];
    r.penetrated = out[0] > 0.5f;
    r.exitVelocity = out[1];
    r.damageRetained = out[2];
    r.deflection = out[3];
    r.energyDeposited = out[4];
    
    if(r.penetrated) impl_->pens++; else impl_->stops++;
    
    return ErrorCode::Success;
}

ErrorCode Engine::CalculateBatch(const BulletType* bt, const float* vel, const float* ang,
                                  const MaterialType* mt, const float* thick,
                                  PenetrationResult* results, uint32_t count) const {
    for(uint32_t i=0; i<count; i++) {
        ErrorCode e = Calculate(bt[i], vel[i], ang[i], mt[i], thick[i], results[i]);
        if(e != ErrorCode::Success) return e;
    }
    return ErrorCode::Success;
}

float Engine::GetTypicalVelocity(BulletType bt) { return BULLET_PROPS[(int)bt][3]; }
float Engine::GetMaxSteelPenetration(BulletType bt) { return BULLET_PROPS[(int)bt][4]; }
float Engine::GetMaterialResistance(MaterialType mt) { return MAT_PROPS[(int)mt][3]; }

Engine::Stats Engine::GetStats() const { return {impl_->queries, impl_->pens, impl_->stops}; }
void Engine::ResetStats() { impl_->queries = impl_->pens = impl_->stops = 0; }

const char* BulletTypeToString(BulletType bt) {
    static const char* names[] = {"Pistol9mm","Pistol45ACP","SMG9mm","SMG45ACP","AR556","AR762",
                                   "LMG556","LMG762","Sniper762","Sniper338","Sniper50BMG",
                                   "ShotgunBuck","ShotgunSlug"};
    return names[(int)bt];
}

const char* MaterialTypeToString(MaterialType mt) {
    static const char* names[] = {"Drywall","Plywood","WoodSolid","Glass","SheetMetal","CarDoor",
                                   "CarBody","Sandbags","Dirt","Brick","ConcreteThin","Concrete",
                                   "Kevlar","SteelThin","Steel","Titanium","Water"};
    return names[(int)mt];
}

const char* ErrorCodeToString(ErrorCode c) {
    switch(c) { case ErrorCode::Success:return"Success"; case ErrorCode::InvalidLicense:return"InvalidLicense";
                case ErrorCode::NotInitialized:return"NotInitialized"; case ErrorCode::InvalidBulletType:return"InvalidBulletType";
                default:return"InvalidMaterialType"; }
}

}} // namespace
