#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <iomanip>
#include "RealityEngine.hpp" 

// --- THE UNIVERSAL ARSENAL MANAGER ---
class UniversalArsenal {
private:
    RealityEngine bulletEngine;
    RealityEngine grenadeEngine;
    bool bulletReady = false;
    bool grenadeReady = false;

    // --- DECODER KEYS (From Python Training) ---
    // These constants translate the "Neural Score" back to "Real Meters"
    // In a full game engine, these would be loaded from a config file.
    const float SNIPER_MEAN_DIST = 1000.0f;
    const float SNIPER_STD_DIST  = 400.0f;
    const float GRENADE_MEAN_DIST = 45.0f;
    const float GRENADE_STD_DIST  = 15.0f;

public:
    // Load all the brains into memory at startup
    void load_arsenal() {
        std::cout << "================================================\n";
        std::cout << "   🚀 UNIVERSAL BALLISTICS KERNEL (UBK-1) \n";
        std::cout << "   ARCHITECTURE: Liquid Neural Network (LNN)\n";
        std::cout << "================================================\n";
        std::cout << "[SYSTEM] Initializing Weapon Systems...\n";

        // 1. Load Bullet Brain
        if (bulletEngine.load_brain("liquid_kernel_v590.bin")) {
            bulletReady = true;
            std::cout << "  ✅ Sniper Module : ONLINE (Surgical Precision)\n";
        } else {
            std::cout << "  ❌ Sniper Module : OFFLINE (Missing liquid_kernel_v590.bin)\n";
        }

        // 2. Load Grenade Brain
        if (grenadeEngine.load_brain("grenade_kernel.bin")) {
            grenadeReady = true;
            std::cout << "  ✅ Grenade Module: ONLINE (High-Arc Physics)\n";
        } else {
            std::cout << "  ❌ Grenade Module: OFFLINE (Missing grenade_kernel.bin)\n";
        }
        std::cout << "------------------------------------------------\n";
    }

    // The Universal Fire Function
    // weaponType: 0 = Sniper, 1 = Grenade
    void fire(int weaponType, float dist_norm, float v0_norm, float angle_rad, float wind_norm, float time) {
        std::vector<float> result;
        
        // Start the Stopwatch (High-Res)
        auto start = std::chrono::high_resolution_clock::now();

        if (weaponType == 0 && bulletReady) {
            // --- FIRE SNIPER ---
            // Input: [dist, v0, angle, temp, press, wind, time]
            // We use standard Temp (0.5) and Pressure (1.0)
            result = bulletEngine.predict({dist_norm, v0_norm, angle_rad, 0.5f, 1.0f, wind_norm, time});
            
            // DECODE: Convert raw neural output to meters
            if (!result.empty()) {
                result[0] = (result[0] * SNIPER_STD_DIST) + SNIPER_MEAN_DIST;
                result[1] = result[1] * 5.0f; // Windage scale
            }
        } 
        else if (weaponType == 1 && grenadeReady) {
            // --- FIRE GRENADE ---
            result = grenadeEngine.predict({dist_norm, v0_norm, angle_rad, 0.3f, 1.0f, wind_norm, time});
            
            // DECODE: Convert raw neural output to meters
            if (!result.empty()) {
                result[0] = (result[0] * GRENADE_STD_DIST) + GRENADE_MEAN_DIST;
                result[1] = result[1] * 2.0f; // Windage scale
            }
        }
        else {
            std::cout << "⚠️ SYSTEM ERROR: Weapon system not ready or invalid ID!\n";
            return;
        }

        // Stop Stopwatch
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        // Print Report
        if (!result.empty()) {
            std::cout << std::fixed << std::setprecision(3);
            std::cout << (weaponType == 0 ? "🔫 [SNIPER] " : "💣 [GRENADE]") 
                      << " Impact X: " << std::setw(8) << result[0] << "m | "
                      << "Drift Z: " << std::setw(7) << result[1] << "m | "
                      << "Latency: " << duration.count() << "µs\n";
        }
    }
};

int main() {
    UniversalArsenal arsenal;
    arsenal.load_arsenal();

    // --- SIMULATION 1: The Sniper Shot ---
    // Scenario: Long range shot (1000m), High Velocity (Mach 2.5)
    std::cout << "\n--- SIMULATION 1: Sniper Fire (Mach 2.5) ---\n";
    // Inputs are normalized relative to training data (approx /100 or /1000)
    arsenal.fire(0, 10.0f, 8.50f, 0.026f, 1.2f, 1.2f); 

    // --- SIMULATION 2: The Grenade Arc ---
    // Scenario: Medium throw (45m), Low Velocity (25m/s), High Angle (45 deg)
    std::cout << "\n--- SIMULATION 2: Grenade Throw (Subsonic) ---\n";
    float g_dist = 0.45f;  // Normalized Distance
    float g_v0   = 0.25f;  // Normalized Velocity
    float g_ang  = 0.785f; // 45 degrees in radians
    float g_time = 3.5f;   // Flight time to check
    
    arsenal.fire(1, g_dist, g_v0, g_ang, 0.5f, g_time);
    
    // --- STRESS TEST: RAPID FIRE ---
    std::cout << "\n--- STRESS TEST: 5 Rapid Grenade Volleys ---\n";
    for(int i=0; i<5; i++) {
        arsenal.fire(1, g_dist + (i*0.01f), g_v0, g_ang, 0.5f, g_time);
    }

    return 0;
}