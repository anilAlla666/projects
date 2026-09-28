#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <random>

// --- CONFIGURATION ---
const int MAP_SIZE = 256;       // 256x256 Grid
const int N_PARTICLES = 5000;   // 5k Objects
const int N_OBSTACLES = 50;     // 50 Pillars (Congestion)
const int FRAMES = 600;         // 10 Seconds of Gameplay

// --- 1. THE MAP (SDF & OBSTACLES) ---
struct Obstacle { float x, y, r; };
std::vector<Obstacle> OBSTACLES;
float SDF_MAP[MAP_SIZE * MAP_SIZE]; // Flattened 2D Array

// Helper: Distance between two points
inline float dist_sq(float x1, float y1, float x2, float y2) {
    return (x1-x2)*(x1-x2) + (y1-y2)*(y1-y2);
}

// GENERATE WORLD (Run once at startup)
void Generate_World() {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> pos_dist(20.0f, 230.0f);
    std::uniform_real_distribution<float> rad_dist(5.0f, 15.0f);

    // 1. Create Obstacles (The Standard Engine's Nightmare)
    for(int i=0; i<N_OBSTACLES; ++i) {
        OBSTACLES.push_back({pos_dist(rng), pos_dist(rng), rad_dist(rng)});
    }

    // 2. Bake SDF (The HyperFlux Solution)
    // We pre-calculate distance to nearest wall for EVERY pixel
    for(int y=0; y<MAP_SIZE; ++y) {
        for(int x=0; x<MAP_SIZE; ++x) {
            float min_dist = 1000.0f;
            
            // Find distance to NEAREST obstacle
            for(const auto& obs : OBSTACLES) {
                float d = std::sqrt(dist_sq(x, y, obs.x, obs.y)) - obs.r;
                if (d < min_dist) min_dist = d;
            }
            
            // Store in Ghost Map (O(1) Lookup later)
            // Flip sign: < 0 is wall, > 0 is air
            SDF_MAP[y * MAP_SIZE + x] = min_dist;
        }
    }
}

// --- 2. STANDARD ENGINE (Collision Loop) ---
void Standard_Engine_Update(float* pos, float* vel, int count, float dt) {
    for(int i=0; i<count; ++i) {
        // A. Move
        pos[2*i]   += vel[2*i] * dt;
        pos[2*i+1] += vel[2*i+1] * dt;

        // B. Collision Check (THE BOTTLENECK)
        // Must check EVERY obstacle to see if we hit it
        for(const auto& obs : OBSTACLES) {
            float d2 = dist_sq(pos[2*i], pos[2*i+1], obs.x, obs.y);
            if (d2 < obs.r * obs.r) {
                // Hit! Stop.
                vel[2*i] *= -0.5f;
                vel[2*i+1] *= -0.5f;
                break; // Found one, stop checking
            }
        }
    }
}

// --- 3. HYPERFLUX ENGINE (LNN + SDF) ---
// Mock LNN: 16 Neurons (Fast Forward)
// In production, this loads weights. Here we simulate the compute cost.
inline void Run_LNN_Inference_Lite(float* vel, int i) {
    // Simulating 16x16 Matrix Multiply cost
    float temp = 0;
    for(int k=0; k<256; ++k) temp += vel[2*i] * 0.001f; 
    // (This is just to burn CPU cycles equivalent to the AI)
}

void HyperFlux_Update(float* pos, float* vel, int count, float dt) {
    for(int i=0; i<count; ++i) {
        // A. LNN Prediction (Simulated Cost)
        Run_LNN_Inference_Lite(vel, i);
        
        float next_x = pos[2*i]   + vel[2*i] * dt;
        float next_y = pos[2*i+1] + vel[2*i+1] * dt;

        // B. SDF LOOKUP (The Magic O(1))
        // Convert to grid coordinates
        int ix = std::max(0, std::min(MAP_SIZE-1, (int)next_x));
        int iy = std::max(0, std::min(MAP_SIZE-1, (int)next_y));
        
        // Fetch from Ghost Map (Instant)
        float dist = SDF_MAP[iy * MAP_SIZE + ix];

        // C. REALITY GATE
        if (dist > 0.0f) {
            // Safe (Air)
            pos[2*i] = next_x;
            pos[2*i+1] = next_y;
        } else {
            // Hit (Wall)
            vel[2*i] *= -0.5f;
            vel[2*i+1] *= -0.5f;
        }
    }
}

// --- 4. THE BENCHMARK ---
int main() {
    Generate_World();
    
    std::vector<float> p1(N_PARTICLES*2), v1(N_PARTICLES*2);
    // Init Random
    for(int i=0; i<N_PARTICLES*2; ++i) {
        p1[i] = 128.0f; // Center
        v1[i] = (rand()%20) - 10.0f;
    }
    std::vector<float> p2 = p1, v2 = v1;

    std::cout << "⚔️  C++ ARCHITECTURE BATTLE\n";
    std::cout << "    Particles: " << N_PARTICLES << "\n";
    std::cout << "    Obstacles: " << N_OBSTACLES << " (Congestion)\n\n";

    // ROUND 1: STANDARD
    auto s1 = std::chrono::high_resolution_clock::now();
    for(int f=0; f<FRAMES; ++f) Standard_Engine_Update(p1.data(), v1.data(), N_PARTICLES, 0.016f);
    auto e1 = std::chrono::high_resolution_clock::now();
    double t1 = std::chrono::duration<double, std::milli>(e1-s1).count();
    std::cout << "🔴 Standard Engine: " << t1 << " ms\n";

    // ROUND 2: HYPERFLUX
    auto s2 = std::chrono::high_resolution_clock::now();
    for(int f=0; f<FRAMES; ++f) HyperFlux_Update(p2.data(), v2.data(), N_PARTICLES, 0.016f);
    auto e2 = std::chrono::high_resolution_clock::now();
    double t2 = std::chrono::duration<double, std::milli>(e2-s2).count();
    std::cout << "🔵 HyperFlux Engine: " << t2 << " ms\n";

    std::cout << "\n🚀 SPEEDUP: " << t1/t2 << "x FASTER\n";
    return 0;
}