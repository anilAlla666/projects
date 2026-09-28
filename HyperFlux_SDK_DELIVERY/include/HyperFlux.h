/*
 * HYPERFLUX BALLISTICS SDK
 * Version 1.0.0
 * Copyright (c) 2026 HyperFlux Technologies. All Rights Reserved.
 */

#ifndef HYPERFLUX_H
#define HYPERFLUX_H

#include <cstdint>

#if defined(_WIN32)
    #ifdef HYPERFLUX_BUILD_DLL
        #define HYPERFLUX_API __declspec(dllexport)
    #else
        #define HYPERFLUX_API __declspec(dllimport)
    #endif
#else
    #define HYPERFLUX_API __attribute__((visibility("default")))
#endif

namespace HyperFlux {

struct HYPERFLUX_API Vec3 {
    float x, y, z;
    Vec3() : x(0), y(0), z(0) {}
    Vec3(float x_, float y_, float z_) : x(x_), y(y_), z(z_) {}
    Vec3 operator+(const Vec3& o) const { return Vec3(x+o.x, y+o.y, z+o.z); }
    Vec3 operator-(const Vec3& o) const { return Vec3(x-o.x, y-o.y, z-o.z); }
    Vec3 operator*(float s) const { return Vec3(x*s, y*s, z*s); }
    Vec3 operator/(float s) const { return Vec3(x/s, y/s, z/s); }
    float dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }
    float length() const;
    Vec3 normalized() const;
};

enum class Weapon : uint32_t {
    M4A1=0, AK_47=1, M16A4=2, ACR_68=3, CR_56_AMAX=4, FN_FAL=5, Grau_556=6, Kilo_141=7, M13=8, SCAR_H=9,
    MP5=10, MP7=11, AUG_9mm=12, Vector=13, Uzi=14, MAC_10=15, P90=16, PPSh_41=17,
    AWP=18, AX_50=19, HDR=20, Barrett_M82=21, Rytec_AMR=22, Intervention=23, Kar98k=24, M24=25, Dragunov=26,
    M249_SAW=27, PKM=28, Bruen_MK9=29, Holger_26=30, M60=31, RPD=32,
    M14_EBR=33, MK2_Carbine=34, SKS=35,
    Model_680=36, Origin_12=37, JAK_12=38, SPAS_12=39, Shotgun_725=40,
    Glock_17=41, Desert_Eagle=42, M1911=43, M9=44, Renetti=45, Magnum_357=46,
    RPG_7=47, Javelin=48, AT4=49, PILA=50, SMAW=51,
    Frag_Grenade=52, Semtex=53, Molotov=54, Thermite=55, C4=56, Claymore=57, Throwing_Knife=58,
    Flashbang=59, Smoke_Grenade=60, Stun_Grenade=61, Snapshot=62, Decoy_Grenade=63, Gas_Grenade=64, Proximity_Mine=65,
    COUNT=66
};

enum class LicenseStatus : uint32_t { Valid=0, Invalid=1, Expired=2, Revoked=3, NetworkError=4, PlatformError=5, NotInitialized=6 };
enum class ErrorCode : uint32_t { Success=0, InvalidLicense=1, InvalidWeapon=2, InvalidInput=3, InternalError=4 };

struct HYPERFLUX_API WeaponParams {
    float muzzleVelocity=800, ballisticCoef=0.15f, bulletMass=0.004f, effectiveRange=500, maxTime=2;
    uint32_t dragModel=0;
    bool hasThrust=false;
};

struct HYPERFLUX_API BatchInput {
    const Vec3* velocities=nullptr;
    const float* times=nullptr;
    const Weapon* weapons=nullptr;
    Weapon singleWeapon=Weapon::M4A1;
    uint32_t count=0;
};

struct HYPERFLUX_API BatchOutput {
    Vec3* positions=nullptr;
    uint32_t count=0;
    ErrorCode error=ErrorCode::Success;
};

class HYPERFLUX_API Engine {
public:
    explicit Engine(const char* licenseKey);
    ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) noexcept;
    Engine& operator=(Engine&&) noexcept;
    
    bool isValid() const;
    LicenseStatus getLicenseStatus() const;
    uint64_t getLicenseExpiration() const;
    
    Vec3 predict(const Vec3& velocity, float time, Weapon weapon) const;
    Vec3 predict(const Vec3& velocity, float time, const WeaponParams& params) const;
    Vec3 predict(const Vec3& velocity, float time, Weapon weapon, ErrorCode* error) const;
    bool predictBatch(const BatchInput& input, BatchOutput& output) const;
    
    static const char* getWeaponName(Weapon weapon);
    static Weapon getWeaponByName(const char* name);
    static WeaponParams getWeaponDefaults(Weapon weapon);
    static const char* getVersion();
    
private:
    class Impl;
    Impl* m_impl;
};

extern "C" {
    HYPERFLUX_API void* HyperFlux_CreateEngine(const char* licenseKey);
    HYPERFLUX_API void HyperFlux_DestroyEngine(void* engine);
    HYPERFLUX_API int HyperFlux_IsValid(void* engine);
    HYPERFLUX_API int HyperFlux_Predict(void* engine, float vx, float vy, float vz, float time, uint32_t weapon, float* outX, float* outY, float* outZ);
    HYPERFLUX_API const char* HyperFlux_GetVersion();
}

} // namespace HyperFlux

#endif // HYPERFLUX_H
