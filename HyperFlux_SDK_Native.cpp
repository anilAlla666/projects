#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <algorithm>
#include <chrono>
#include <thread> // Standard Multi-threading
#include <future> // For managing tasks

// --- CONFIG ---
const int UNITS = 64; 
const int MAP_SIZE = 256; 

// --- 1. THE BRAIN (Liquid Cell) ---
struct LiquidCell {
    std::vector<float> w_ff1, b_ff1, w_ff2, b_ff2;
    std::vector<float> w_tw, w_ta, w_tb; 
    int input_dim, hidden_dim;

    void Init(int in, int hid) { 
        input_dim = in; hidden_dim = hid; 
        // Resize with small random values to prevent NaN if file missing
        w_ff1.resize(hid * (in + hid), 0.01f); b_ff1.resize(hid, 0.0f);
        w_ff2.resize(hid * (in + hid), 0.01f); b_ff2.resize(hid, 0.0f);
        w_tw.resize(hid * hid, 0.01f); w_ta.resize(hid * in, 0.01f); w_tb.resize(hid * in, 0.01f);
    }
    
    // Const-Correct Forward Pass (Safe for Threads)
    void Forward_Safe(const float* x, const float* h_prev, float dt, float* h_new) const {
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
    HyperFluxKernel() { sdf_map.resize(MAP_SIZE * MAP_SIZE, 100.0f); }

    bool Init(const char* brain_file) {
        // Initialize Memory
        layer1.Init(3, UNITS); layer2.Init(UNITS, UNITS);
        w_out.resize(2*UNITS, 0.01f); b_out.resize(2, 0.0f);
        
        // Load Weights
        std::ifstream file(brain_file, std::ios::binary);
        if(file) {
            auto load_vec = [&](std::vector<float>& v) { file.read((char*)v.data(), v.size()*4); };
            load_vec(layer1.w_ff1); load_vec(layer1.b_ff1); load_vec(layer1.w_ff2); load_vec(layer1.b_ff2);
            load_vec(layer1.w_tw); load_vec(layer1.w_ta); load_vec(layer1.w_tb);
            load_vec(layer2.w_ff1); load_vec(layer2.b_ff1); load_vec(layer2.w_ff2); load_vec(layer2.b_ff2);
            load_vec(layer2.w_tw); load_vec(layer2.w_ta); load_vec(layer2.w_tb);
            load_vec(w_out);
            std::cout << "✅ HyperFlux Brain Loaded.\n";
        } else { std::cout << "⚠️ Brain Not Found. Running Safe Mode.\n"; }

        // Setup Wall at X=200
        for(int y=0; y<MAP_SIZE; ++y) for(int x=200; x<MAP_SIZE; ++x) sdf_map[y*MAP_SIZE + x] = -1.0f; 
        loaded = true;
        return true;
    }

    // Thread-Safe Reality Gate (Read Only)
    bool Check_Reality_Gate(float x, float y) const {
        int ix = std::max(0, std::min(MAP_SIZE-1, (int)x));
        int iy = std::max(0, std::min(MAP_SIZE-1, (int)y));
        return sdf_map[iy * MAP_SIZE + ix] > 0.0f; 
    }

    // --- OPTIMIZED WORKER (Local Stack Memory) ---
    void Worker_Job(int start_idx, int end_idx, float* pos, float* vel, float dt) {
        // Local Stack Variables to prevent Cache Thrashing
        float h1[UNITS], h2[UNITS], h_zero[UNITS] = {0};
        float input[3];

        for(int i=start_idx; i<end_idx; ++i) {
            // 1. Read Inputs (Fast)
            float vx = vel[2*i];
            float vy = vel[2*i+1];
            float px = pos[2*i];
            float py = pos[2*i+1];
            
            input[0] = vx; input[1] = vy; input[2] = dt;
            
            // 2. Heavy Compute (Local)
            layer1.Forward_Safe(input, h_zero, dt, h1);
            layer2.Forward_Safe(h1, h1, dt, h2);
            
            float dx = b_out[0], dy = b_out[1];
            for(int k=0; k<UNITS; ++k) {
                dx += h2[k] * w_out[k];
                dy += h2[k] * w_out[UNITS+k];
            }
            // Dummy logic if uninitialized
            if(std::abs(dx) < 0.001f) { dx = vx * dt; dy = vy * dt; } 

            float next_x = px + dx;
            float next_y = py + dy;
            
            // 3. Reality Gate (Read)
            bool safe = Check_Reality_Gate(next_x, next_y);
            
            // 4. Write Back (Main Memory Access)
            if (safe) {
                pos[2*i] = next_x; pos[2*i+1] = next_y;
            } else {
                vel[2*i] = 0; vel[2*i+1] = 0;
            }
        }
    }

    // MANAGER FUNCTION
    void Accelerate_Physics_Parallel(float* pos, float* vel, int count, float dt) {
        if(!loaded) return;

        unsigned int n_threads = std::thread::hardware_concurrency();
        if(n_threads == 0) n_threads = 4;
        
        std::vector<std::thread> workers;
        int chunk_size = count / n_threads;

        for(int t=0; t<n_threads; ++t) {
            int start = t * chunk_size;
            int end = (t == n_threads-1) ? count : (start + chunk_size);
            workers.emplace_back(&HyperFluxKernel::Worker_Job, this, start, end, pos, vel, dt);
        }

        for(auto& worker : workers) worker.join();
    }
};

// --- BENCHMARK ---
int main() {
    const int N = 50000; 
    std::vector<float> pos(N*2, 0.0f), vel(N*2, 10.0f);
    
    HyperFluxKernel kernel;
    kernel.Init("hyperflux_pro.bin");

    std::cout << "🚀 STARTING OPTIMIZED PARALLEL BENCHMARK (50,000 Objects)...\n";
    std::cout << "   Hardware Threads: " << std::thread::hardware_concurrency() << "\n";

    auto start = std::chrono::high_resolution_clock::now();
    
    kernel.Accelerate_Physics_Parallel(pos.data(), vel.data(), N, 1.0f);
    
    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "✅ Optimized Time: " << ms << " ms\n";
    std::cout << "⚡ Objects per Second: " << (long)(N / (ms/1000.0)) << "\n";
    
    return 0;
}