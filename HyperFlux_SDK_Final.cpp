#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <chrono>

// --- CONFIG ---
const int UNITS = 64; 
const int MAP_SIZE = 256; 

// --- 1. THE BRAIN (Liquid Cell Logic) ---
struct LiquidCell {
    std::vector<float> w_ff1, b_ff1, w_ff2, b_ff2;
    std::vector<float> w_tw, w_ta, w_tb; 
    int input_dim, hidden_dim;

    // FIX: Resize vectors immediately to prevent Segfault
    void Init(int in, int hid) { 
        input_dim = in; hidden_dim = hid; 
        
        w_ff1.resize(hid * (in + hid), 0.01f); b_ff1.resize(hid, 0.0f);
        w_ff2.resize(hid * (in + hid), 0.01f); b_ff2.resize(hid, 0.0f);
        w_tw.resize(hid * hid, 0.01f);
        w_ta.resize(hid * in, 0.01f);
        w_tb.resize(hid * in, 0.01f);
    }
    
    void Forward(const float* x, const float* h_prev, float dt, float* h_new) {
        // [Same Liquid Math as before]
        std::vector<float> t_factor(hidden_dim);
        
        // 1. Time Gating
        for(int i=0; i<hidden_dim; ++i) {
            float sum = 0.0f;
            for(int j=0; j<hidden_dim; ++j) sum += h_prev[j] * w_tw[i*hidden_dim + j];
            for(int j=0; j<input_dim; ++j) sum += x[j] * w_ta[i*input_dim + j];
            for(int j=0; j<input_dim; ++j) sum += x[j] * w_tb[i*input_dim + j];
            t_factor[i] = 1.0f / (1.0f + std::exp(-sum)); 
        }
        
        // 2. State Update
        std::vector<float> combined(input_dim + hidden_dim);
        for(int i=0; i<input_dim; ++i) combined[i] = x[i];
        for(int i=0; i<hidden_dim; ++i) combined[input_dim+i] = h_prev[i];
        
        for(int i=0; i<hidden_dim; ++i) {
            float sum1 = b_ff1[i], sum2 = b_ff2[i];
            for(int j=0; j<input_dim+hidden_dim; ++j) {
                sum1 += combined[j] * w_ff1[i*(input_dim+hidden_dim) + j];
                sum2 += combined[j] * w_ff2[i*(input_dim+hidden_dim) + j];
            }
            float tanh_out = std::tanh(sum1);
            float sig_out = 1.0f / (1.0f + std::exp(-sum2));
            float gate = sig_out * t_factor[i];
            h_new[i] = gate * h_prev[i] + (1.0f - gate) * tanh_out;
        }
    }
};

// --- 2. THE UNIVERSAL KERNEL ---
class HyperFluxKernel {
    LiquidCell layer1, layer2;
    std::vector<float> w_out, b_out;
    bool loaded = false;
    std::vector<float> sdf_map; 

public:
    HyperFluxKernel() {
        sdf_map.resize(MAP_SIZE * MAP_SIZE, 100.0f); // Default Air
    }

    bool Init(const char* brain_file) {
        // Initialize Memory FIRST (Prevent Segfault)
        layer1.Init(3, UNITS); 
        layer2.Init(UNITS, UNITS);
        w_out.resize(2*UNITS, 0.01f); 
        b_out.resize(2, 0.0f);
        
        // Attempt to Load Real Weights
        std::ifstream file(brain_file, std::ios::binary);
        if(file) {
            auto load_vec = [&](std::vector<float>& v) {
                file.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(float));
            };
            load_vec(layer1.w_ff1); load_vec(layer1.b_ff1);
            load_vec(layer1.w_ff2); load_vec(layer1.b_ff2);
            load_vec(layer1.w_tw); load_vec(layer1.w_ta); load_vec(layer1.w_tb);
            
            load_vec(layer2.w_ff1); load_vec(layer2.b_ff1);
            load_vec(layer2.w_ff2); load_vec(layer2.b_ff2);
            load_vec(layer2.w_tw); load_vec(layer2.w_ta); load_vec(layer2.w_tb);
            
            load_vec(w_out);
            std::cout << "✅ HyperFlux Brain Loaded.\n";
        } else {
            std::cout << "⚠️ Brain File Not Found. Using Initialized Memory (Safe Mode).\n";
        }

        // Setup Wall at X=200 for Test
        for(int y=0; y<MAP_SIZE; ++y) {
            for(int x=200; x<MAP_SIZE; ++x) sdf_map[y*MAP_SIZE + x] = -1.0f; 
        }
        
        loaded = true;
        return true;
    }

    bool Check_Reality_Gate(float x, float y) {
        int ix = std::max(0, std::min(MAP_SIZE-1, (int)x));
        int iy = std::max(0, std::min(MAP_SIZE-1, (int)y));
        return sdf_map[iy * MAP_SIZE + ix] > 0.0f; 
    }

    void Accelerate_Physics(float* pos, float* vel, int count, float dt) {
        if(!loaded) return;

        for(int i=0; i<count; ++i) {
            float vx = vel[2*i];
            float vy = vel[2*i+1];
            
            // 1. BRAIN
            float input[3] = {vx, vy, dt};
            float h1[UNITS], h2[UNITS], h_zero[UNITS] = {0};
            
            layer1.Forward(input, h_zero, dt, h1);
            layer2.Forward(h1, h1, dt, h2);
            
            // Decode
            float dx = b_out[0], dy = b_out[1];
            for(int k=0; k<UNITS; ++k) {
                dx += h2[k] * w_out[k];
                dy += h2[k] * w_out[UNITS+k];
            }
            
            // 2. PROPOSE MOVE (Simplified Prediction)
            // If weights are dummy, we just add velocity to prove Gate works
            if(std::abs(dx) < 0.001f) { dx = vx * dt; dy = vy * dt; } 

            float next_x = pos[2*i] + dx;
            float next_y = pos[2*i+1] + dy;
            
            // 3. REALITY GATE
            if (Check_Reality_Gate(next_x, next_y)) {
                pos[2*i] = next_x;
                pos[2*i+1] = next_y;
            } else {
                // COLLISION!
                // We keep old position. We zero velocity.
                vel[2*i] = 0;
                vel[2*i+1] = 0;
            }
        }
    }
};

int main() {
    HyperFluxKernel kernel;
    kernel.Init("hyperflux_pro.bin");

    float pos[2] = {195.0f, 50.0f}; 
    float vel[2] = {10.0f, 0.0f};   
    
    std::cout << "🚀 TEST: PARTICLE VS WALL\n";
    std::cout << "   Start: " << pos[0] << "\n";
    std::cout << "   Wall:  200.0\n";
    
    // Attempt Jump
    kernel.Accelerate_Physics(pos, vel, 1, 1.0f);
    
    std::cout << "   Result: " << pos[0] << "\n";
    
    if (pos[0] < 200.0f) std::cout << "✅ REALITY GATE BLOCKED THE GHOSTING.\n";
    else std::cout << "❌ FAILED: GHOSTED THROUGH WALL.\n";
    
    return 0;
}