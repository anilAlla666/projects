#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <chrono>
#include <random>
#include <Accelerate/Accelerate.h>

class RealityEngine {
private:
    std::vector<float> weights;
    // PRE-ALLOCATED BUFFERS
    std::vector<float> exp_out, h_res1, h_path, h_skip, h_combined;
    std::vector<float> tau_v, gate_v, liquid, temp_v;
    std::vector<float> gate_sig, decay_vec, softplus_tau;

public:
    RealityEngine() {
        exp_out.resize(2048);
        h_res1.resize(1024);
        h_path.resize(1024);
        h_skip.resize(1024);
        h_combined.resize(1024);
        tau_v.resize(1024);
        gate_v.resize(1024);
        liquid.resize(1024);
        temp_v.resize(2048);
        gate_sig.resize(1024);
        decay_vec.resize(1024);
        softplus_tau.resize(1024);
    }

    bool load_bin(const std::string& path) {
        std::ifstream f(path, std::ios::binary);
        if (!f.is_open()) return false;
        float val;
        while (f.read(reinterpret_cast<char*>(&val), sizeof(float))) {
            weights.push_back(val);
        }
        std::cout << "✅ Engine Primed: " << weights.size() << " parameters active." << std::endl;
        return true;
    }

    void matmul_fast(const std::vector<float>& in, std::vector<float>& out, int in_d, int out_d, size_t& ptr) {
        const float* w_ptr = &weights[ptr];
        const float* b_ptr = &weights[ptr + in_d * out_d];
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, 1, out_d, in_d, 1.0f, in.data(), in_d, w_ptr, in_d, 0.0f, out.data(), out_d);
        vDSP_vadd(out.data(), 1, b_ptr, 1, out.data(), 1, out_d);
        ptr += (in_d * out_d) + out_d;
    }

    void apply_silu_vec(float* data, int n) {
        float neg_one = -1.0f;
        vDSP_vsmul(data, 1, &neg_one, temp_v.data(), 1, n);
        vvexpf(temp_v.data(), temp_v.data(), &n);
        for (int i = 0; i < n; ++i) data[i] = data[i] / (1.0f + temp_v[i]);
    }

    void apply_ln(std::vector<float>& x, const float* w, const float* b, int n) {
        float mean = 0, var = 0;
        vDSP_meanv(x.data(), 1, &mean, n);
        float neg_mean = -mean;
        vDSP_vsadd(x.data(), 1, &neg_mean, temp_v.data(), 1, n);
        vDSP_measqv(temp_v.data(), 1, &var, n);
        float inv_std = 1.0f / std::sqrt(var + 1e-5f);
        for (int i = 0; i < n; ++i) x[i] = (x[i] - mean) * inv_std * w[i] + b[i];
    }

    std::vector<float> predict(float dist, float v0, float angle, float temp, float press, float wind, float t_flight) {
        size_t ptr = 0;
        std::vector<float> x_in = {dist/1000.f, v0/1000.f, angle, temp/50.f, press/1000.f, wind/10.f, t_flight};
        int n = 1024;

        matmul_fast(x_in, exp_out, 7, 2048, ptr);
        apply_ln(exp_out, &weights[ptr], &weights[ptr+2048], 2048); ptr += 4096;
        apply_silu_vec(exp_out.data(), 2048);

        matmul_fast(exp_out, h_res1, 2048, 1024, ptr);
        apply_ln(h_res1, &weights[ptr], &weights[ptr+1024], 1024); ptr += 2048;
        apply_silu_vec(h_res1.data(), 1024);

        matmul_fast(h_res1, h_path, 1024, 1024, ptr);
        apply_ln(h_path, &weights[ptr], &weights[ptr+1024], 1024); ptr += 2048;

        matmul_fast(exp_out, h_skip, 2048, 1024, ptr);
        vDSP_vadd(h_path.data(), 1, h_skip.data(), 1, h_combined.data(), 1, n);
        apply_silu_vec(h_combined.data(), n);

        matmul_fast(h_combined, tau_v, 1024, 1024, ptr);
        matmul_fast(h_combined, gate_v, 1024, 1024, ptr);

        float neg_one = -1.0f;
        vDSP_vsmul(gate_v.data(), 1, &neg_one, temp_v.data(), 1, n);
        vvexpf(gate_sig.data(), temp_v.data(), &n);
        for(int i=0; i<n; ++i) gate_sig[i] = 1.0f / (1.0f + gate_sig[i]);

        vvexpf(temp_v.data(), tau_v.data(), &n);
        vvlog1pf(softplus_tau.data(), temp_v.data(), &n);

        float neg_t = -t_flight;
        vDSP_vsmul(softplus_tau.data(), 1, &neg_t, temp_v.data(), 1, n);
        vvexpf(decay_vec.data(), temp_v.data(), &n);

        vDSP_vmul(h_combined.data(), 1, gate_sig.data(), 1, liquid.data(), 1, n);
        vDSP_vmul(liquid.data(), 1, decay_vec.data(), 1, liquid.data(), 1, n);

        static std::vector<float> out(2);
        matmul_fast(liquid, out, 1024, 2, ptr);
        return out;
    }
};

int main() {
    RealityEngine engine;
    if (!engine.load_bin("liquid_kernel_v590.bin")) return 1;

    // --- MONTE CARLO CONFIGURATION ---
    const int num_shots = 1000000;
    std::mt19937 gen(42); 
    
    // Realistic Distributions
    std::uniform_real_distribution<float> dist_range(100.0, 3000.0);  // 100m to 3km
    std::uniform_real_distribution<float> v0_range(750.0, 950.0);    // Velocity
    std::uniform_real_distribution<float> wind_range(-20.0, 20.0);   // Hurricane wind
    std::uniform_real_distribution<float> temp_range(-30.0, 50.0);   // Extreme weather
    std::uniform_real_distribution<float> time_range(0.1, 5.0);      // Long flight time

    std::cout << "🚀 Initiating 1,000,000 Shot Monte Carlo Stress Test..." << std::endl;

    float min_drop = 0.0f, max_drop = -99999.0f;
    float min_wind = 0.0f, max_wind = -99999.0f;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_shots; ++i) {
        float d = dist_range(gen);
        float v = v0_range(gen);
        float w = wind_range(gen);
        float t = temp_range(gen);
        float tf = time_range(gen);
        
        auto res = engine.predict(d, v, 0.12f, t, 1013.0f, w, tf);

        // Track extreme vertical drop
        if (res[0] < max_drop) max_drop = res[0]; // Most negative
        if (res[0] > min_drop) min_drop = res[0]; // Least negative

        // Track absolute max windage
        if (std::abs(res[1]) > max_wind) max_wind = std::abs(res[1]);

        if (i % 200000 == 0 && i > 0) std::cout << "  - Processed " << i << " shots..." << std::endl;
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto total_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "\n==========================================" << std::endl;
    std::cout << "        MONTE CARLO STRESS REPORT" << std::endl;
    std::cout << "==========================================" << std::endl;
    std::cout << "Total Simulation Time: " << total_ms / 1000.0 << " seconds" << std::endl;
    std::cout << "Average Latency:       " << (total_ms * 1000.0) / num_shots << " us" << std::endl;
    std::cout << "------------------------------------------" << std::endl;
    std::cout << "REALITY BOUNDS DETECTED:" << std::endl;
    std::cout << "Maximum Vertical Drop: " << max_drop << " m" << std::endl;
    std::cout << "Maximum Windage Drift: " << max_wind << " m" << std::endl;
    std::cout << "------------------------------------------" << std::endl;

    if (max_drop < -1000.0) {
        std::cout << "⚠️ WARNING: Possible Gate Leakage detected in extreme scenarios." << std::endl;
    } else {
        std::cout << "✅ SUCCESS: Reality Gate held bounds perfectly." << std::endl;
    }
    std::cout << "==========================================" << std::endl;

    return 0;
}