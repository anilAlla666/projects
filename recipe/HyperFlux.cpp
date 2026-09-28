/*
 * HYPERFLUX BALLISTICS SDK - IMPLEMENTATION
 * Copyright (c) 2026 HyperFlux Technologies. All Rights Reserved.
 * CONFIDENTIAL - DO NOT DISTRIBUTE
 */

#include "HyperFlux.h"
#include <cmath>
#include <algorithm>
#include <cstring>

#include "weights.inc"

namespace HyperFlux {

float Vec3::length() const { return std::sqrt(x*x + y*y + z*z); }
Vec3 Vec3::normalized() const { float l = length(); return l > 0 ? *this / l : Vec3(); }

namespace {

bool validateLicense(const char* key) {
    if (!key) return false;
    int len = 0, dashes = 0;
    for (const char* p = key; *p; ++p) {
        if (*p == '-') dashes++;
        else if ((*p >= 'A' && *p <= 'Z') || (*p >= '0' && *p <= '9')) len++;
        else return false;
    }
    return len == 16 && dashes == 3;
}

struct WeaponData {
    float muzzleVelocity, bc, bulletMass;
    uint32_t dragType;
    float effectiveRange;
    bool hasThrust;
    float tMax;
    const char* name;
};

const WeaponData WEAPON_DB[] = {
    {884, 0.151f, 0.004f, 0, 500, false, 1.5f, "M4A1"},
    {715, 0.295f, 0.008f, 0, 400, false, 1.2f, "AK_47"},
    {948, 0.151f, 0.004f, 0, 550, false, 1.6f, "M16A4"},
    {880, 0.160f, 0.0045f, 0, 600, false, 1.8f, "ACR_68"},
    {800, 0.200f, 0.0065f, 0, 500, false, 1.4f, "CR_56_AMAX"},
    {840, 0.250f, 0.0093f, 0, 600, false, 1.6f, "FN_FAL"},
    {900, 0.155f, 0.004f, 0, 550, false, 1.7f, "Grau_556"},
    {870, 0.153f, 0.004f, 0, 500, false, 1.5f, "Kilo_141"},
    {920, 0.149f, 0.004f, 0, 550, false, 1.6f, "M13"},
    {820, 0.240f, 0.0091f, 0, 600, false, 1.5f, "SCAR_H"},
    {400, 0.165f, 0.008f, 1, 200, false, 0.8f, "MP5"},
    {460, 0.150f, 0.0065f, 1, 180, false, 0.7f, "MP7"},
    {400, 0.165f, 0.008f, 1, 200, false, 0.8f, "AUG_9mm"},
    {400, 0.125f, 0.0065f, 1, 150, false, 0.6f, "Vector"},
    {400, 0.165f, 0.008f, 1, 150, false, 0.6f, "Uzi"},
    {380, 0.125f, 0.0065f, 1, 120, false, 0.5f, "MAC_10"},
    {470, 0.145f, 0.005f, 1, 200, false, 0.8f, "P90"},
    {490, 0.135f, 0.005f, 1, 200, false, 0.75f, "PPSh_41"},
    {936, 0.620f, 0.0425f, 0, 1500, false, 3.0f, "AWP"},
    {900, 0.650f, 0.045f, 0, 1500, false, 3.0f, "AX_50"},
    {900, 0.640f, 0.045f, 0, 1500, false, 3.0f, "HDR"},
    {853, 0.670f, 0.0525f, 0, 1800, false, 3.5f, "Barrett_M82"},
    {800, 0.680f, 0.055f, 0, 1800, false, 3.5f, "Rytec_AMR"},
    {880, 0.630f, 0.042f, 0, 1500, false, 3.2f, "Intervention"},
    {760, 0.530f, 0.0128f, 0, 800, false, 1.8f, "Kar98k"},
    {850, 0.550f, 0.011f, 0, 900, false, 2.0f, "M24"},
    {830, 0.430f, 0.0096f, 0, 700, false, 1.6f, "Dragunov"},
    {915, 0.175f, 0.0042f, 0, 600, false, 1.8f, "M249_SAW"},
    {735, 0.290f, 0.0096f, 0, 550, false, 1.6f, "PKM"},
    {780, 0.185f, 0.0045f, 0, 550, false, 1.6f, "Bruen_MK9"},
    {760, 0.175f, 0.004f, 0, 550, false, 1.6f, "Holger_26"},
    {850, 0.280f, 0.0095f, 0, 600, false, 1.8f, "M60"},
    {735, 0.275f, 0.0096f, 0, 500, false, 1.5f, "RPD"},
    {850, 0.430f, 0.0093f, 0, 700, false, 1.8f, "M14_EBR"},
    {550, 0.180f, 0.012f, 1, 400, false, 1.0f, "MK2_Carbine"},
    {735, 0.330f, 0.008f, 0, 500, false, 1.4f, "SKS"},
    {410, 0.015f, 0.0324f, 2, 30, false, 0.3f, "Model_680"},
    {380, 0.012f, 0.027f, 2, 25, false, 0.25f, "Origin_12"},
    {360, 0.012f, 0.027f, 2, 25, false, 0.25f, "JAK_12"},
    {400, 0.014f, 0.030f, 2, 28, false, 0.28f, "SPAS_12"},
    {410, 0.015f, 0.0324f, 2, 30, false, 0.3f, "725"},
    {375, 0.165f, 0.008f, 1, 50, false, 0.5f, "Glock_17"},
    {470, 0.172f, 0.0195f, 1, 100, false, 0.6f, "Desert_Eagle"},
    {260, 0.165f, 0.015f, 1, 50, false, 0.4f, "M1911"},
    {360, 0.165f, 0.008f, 1, 50, false, 0.45f, "M9"},
    {380, 0.170f, 0.008f, 1, 55, false, 0.5f, "Renetti"},
    {440, 0.165f, 0.0102f, 1, 75, false, 0.55f, "357_Magnum"},
    {115, 0.050f, 2.5f, 3, 300, true, 3.0f, "RPG_7"},
    {140, 0.080f, 11.8f, 3, 2000, true, 5.0f, "Javelin"},
    {290, 0.070f, 4.0f, 3, 500, true, 2.5f, "AT4"},
    {180, 0.075f, 5.5f, 3, 800, true, 4.0f, "PILA"},
    {250, 0.065f, 3.5f, 3, 400, true, 2.0f, "SMAW"},
    {22, 0.005f, 0.4f, 2, 40, false, 3.0f, "Frag_Grenade"},
    {22, 0.005f, 0.35f, 2, 40, false, 3.0f, "Semtex"},
    {18, 0.004f, 0.5f, 2, 30, false, 3.0f, "Molotov"},
    {20, 0.004f, 0.35f, 2, 35, false, 3.0f, "Thermite"},
    {22, 0.003f, 0.35f, 2, 40, false, 3.0f, "C4"},
    {0, 0.0f, 0.2f, 2, 0, false, 3.0f, "Claymore"},
    {30, 0.008f, 0.2f, 2, 25, false, 2.5f, "Throwing_Knife"},
    {20, 0.005f, 0.3f, 2, 35, false, 3.0f, "Flashbang"},
    {20, 0.005f, 0.35f, 2, 35, false, 3.0f, "Smoke_Grenade"},
    {20, 0.005f, 0.3f, 2, 35, false, 3.0f, "Stun_Grenade"},
    {22, 0.005f, 0.25f, 2, 40, false, 3.0f, "Snapshot"},
    {22, 0.005f, 0.3f, 2, 40, false, 3.0f, "Decoy_Grenade"},
    {22, 0.005f, 0.4f, 2, 40, false, 3.0f, "Gas_Grenade"},
    {0, 0.0f, 0.3f, 2, 0, false, 3.0f, "Proximity_Mine"},
};

// Neural network operations
inline float silu(float x) { return x / (1.0f + std::exp(-x)); }

void linear(const float* in, float* out, const float* w, const float* b, int inD, int outD) {
    for (int j = 0; j < outD; j++) {
        float sum = b[j];
        for (int k = 0; k < inD; k++) sum += in[k] * w[j * inD + k];
        out[j] = sum;
    }
}

void layerNorm(float* x, const float* g, const float* b, int d) {
    float mean = 0, var = 0;
    for (int i = 0; i < d; i++) mean += x[i];
    mean /= d;
    for (int i = 0; i < d; i++) { float t = x[i] - mean; var += t * t; }
    var /= d;
    float inv = 1.0f / std::sqrt(var + 1e-5f);
    for (int i = 0; i < d; i++) x[i] = (x[i] - mean) * inv * g[i] + b[i];
}

void siluInPlace(float* x, int d) { for (int i = 0; i < d; i++) x[i] = silu(x[i]); }

void forward(const float* input, float* output) {
    using namespace weights;
    alignas(32) float a[512], b[512];
    
    // Layer 0: 15 -> 512
    linear(input, a, layer0_weight, layer0_bias, 15, 512);
    layerNorm(a, layer0_ln_weight, layer0_ln_bias, 512);
    siluInPlace(a, 512);
    
    // Layer 1: 512 -> 512
    linear(a, b, layer1_weight, layer1_bias, 512, 512);
    layerNorm(b, layer1_ln_weight, layer1_ln_bias, 512);
    siluInPlace(b, 512);
    
    // Layer 2: 512 -> 512
    linear(b, a, layer2_weight, layer2_bias, 512, 512);
    layerNorm(a, layer2_ln_weight, layer2_ln_bias, 512);
    siluInPlace(a, 512);
    
    // Layer 3: 512 -> 256
    linear(a, b, layer3_weight, layer3_bias, 512, 256);
    layerNorm(b, layer3_ln_weight, layer3_ln_bias, 256);
    siluInPlace(b, 256);
    
    // Layer 4: 256 -> 256
    linear(b, a, layer4_weight, layer4_bias, 256, 256);
    layerNorm(a, layer4_ln_weight, layer4_ln_bias, 256);
    siluInPlace(a, 256);
    
    // Output: 256 -> 3
    linear(a, output, layer5_weight, layer5_bias, 256, 3);
}

} // anonymous namespace

class Engine::Impl {
public:
    LicenseStatus status;
    Impl(const char* k) : status(validateLicense(k) ? LicenseStatus::Valid : LicenseStatus::Invalid) {}
    
    Vec3 predict(const Vec3& v0, float t, int wi) const {
        if (status != LicenseStatus::Valid || wi < 0 || wi >= 66) return Vec3();
        const auto& wd = WEAPON_DB[wi];
        
        if (t < 0.01f) return Vec3(v0.x * t, v0.y * t, v0.z * t);
        float tc = std::min(std::max(t, 0.01f), wd.tMax * 0.98f);
        
        float input[15] = {
            v0.x / 1000.f, v0.y / 1000.f, v0.z / 1000.f, tc / 10.f,
            wd.muzzleVelocity / 1000.f, wd.bc / 0.5f,
            std::log10(wd.bulletMass + 1e-6f) / 2.f + 1.f,
            wd.dragType == 0 ? 1.f : 0.f, wd.dragType == 1 ? 1.f : 0.f,
            wd.dragType == 2 ? 1.f : 0.f, wd.dragType == 3 ? 1.f : 0.f,
            wd.effectiveRange / 2000.f, wd.hasThrust ? 1.f : 0.f,
            wd.tMax / 5.f, 320.f / 512.f
        };
        
        float output[3];
        forward(input, output);
        return Vec3(output[0], output[1], output[2]);
    }
};

Engine::Engine(const char* k) : m_impl(new Impl(k)) {}
Engine::~Engine() { delete m_impl; }
Engine::Engine(Engine&& o) noexcept : m_impl(o.m_impl) { o.m_impl = nullptr; }
Engine& Engine::operator=(Engine&& o) noexcept { if (this != &o) { delete m_impl; m_impl = o.m_impl; o.m_impl = nullptr; } return *this; }

bool Engine::isValid() const { return m_impl && m_impl->status == LicenseStatus::Valid; }
LicenseStatus Engine::getLicenseStatus() const { return m_impl ? m_impl->status : LicenseStatus::NotInitialized; }
uint64_t Engine::getLicenseExpiration() const { return 0; }

Vec3 Engine::predict(const Vec3& v, float t, Weapon w) const { return m_impl ? m_impl->predict(v, t, (int)w) : Vec3(); }
Vec3 Engine::predict(const Vec3& v, float t, const WeaponParams&) const { return predict(v, t, Weapon::M4A1); }
Vec3 Engine::predict(const Vec3& v, float t, Weapon w, ErrorCode* e) const { if (e) *e = ErrorCode::Success; return predict(v, t, w); }

bool Engine::predictBatch(const BatchInput& in, BatchOutput& out) const {
    if (!m_impl || !in.velocities || !in.times || !out.positions) return false;
    for (uint32_t i = 0; i < in.count; i++) {
        Weapon w = in.weapons ? in.weapons[i] : in.singleWeapon;
        out.positions[i] = predict(in.velocities[i], in.times[i], w);
    }
    out.count = in.count;
    return true;
}

const char* Engine::getWeaponName(Weapon w) { int i = (int)w; return (i >= 0 && i < 66) ? WEAPON_DB[i].name : "Unknown"; }
Weapon Engine::getWeaponByName(const char* n) { if (!n) return Weapon::COUNT; for (int i = 0; i < 66; i++) if (strcmp(WEAPON_DB[i].name, n) == 0) return (Weapon)i; return Weapon::COUNT; }
WeaponParams Engine::getWeaponDefaults(Weapon w) { WeaponParams p; int i = (int)w; if (i >= 0 && i < 66) { auto& d = WEAPON_DB[i]; p.muzzleVelocity = d.muzzleVelocity; p.ballisticCoef = d.bc; p.bulletMass = d.bulletMass; p.dragModel = d.dragType; p.effectiveRange = d.effectiveRange; p.hasThrust = d.hasThrust; p.maxTime = d.tMax; } return p; }
const char* Engine::getVersion() { return "1.0.0"; }

extern "C" {
void* HyperFlux_CreateEngine(const char* k) { return new Engine(k); }
void HyperFlux_DestroyEngine(void* e) { delete (Engine*)e; }
int HyperFlux_IsValid(void* e) { return e && ((Engine*)e)->isValid() ? 1 : 0; }
int HyperFlux_Predict(void* e, float vx, float vy, float vz, float t, uint32_t w, float* ox, float* oy, float* oz) {
    if (!e || !ox || !oy || !oz) return -1;
    Vec3 p = ((Engine*)e)->predict(Vec3(vx,vy,vz), t, (Weapon)w);
    *ox = p.x; *oy = p.y; *oz = p.z; return 0;
}
const char* HyperFlux_GetVersion() { return "1.0.0"; }
}

} // namespace HyperFlux
