#ifndef REALITY_ENGINE_HPP
#define REALITY_ENGINE_HPP

#include <vector>
#include <cmath>
#include <fstream>
#include <iostream>
#include <Accelerate/Accelerate.h>

/**
 * RealityEngine: Universal Liquid Ballistics Kernel
 * Supports: Bullets, Grenades, Arrows (via Modular .bin files)
 * Architecture: 7-2048-1024-2 (Universal DNA Input)
 */
class RealityEngine {
private:
    std::vector<float> weights;
    // Architecture Constants
    const int INPUT_DIM = 7;
    const int L1_DIM = 2048;
    const int L2_DIM = 1024;
    const int OUT_DIM = 2;

public:
    // --- 1. DYNAMIC BRAIN LOADER ---
    // Replaces old 'load_weights' to support hot-swapping .bin files
    bool load_brain(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file) return false;

        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);

        // Approximate validation for the 7.3M parameter file size (~29MB)
        if (size < 5000000) return false; 

        // Allocate memory
        weights.resize(size / sizeof(float));
        if (file.read(reinterpret_cast<char*>(weights.data()), size)) {
            return true;
        }
        return false;
    }

    // --- 2. UNIVERSAL PREDICTOR ---
    // Now accepts a vector {dist, v0, angle, temp, press, wind, time}
    std::vector<float> predict(const std::vector<float>& inputs) {
        if (weights.empty()) return {0.0f, 0.0f};

        // --- LAYER 1: EXPANSION (7 -> 2048) ---
        std::vector<float> h1(L1_DIM);
        // Matrix Multiplication: Weights * Input
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    1, L1_DIM, INPUT_DIM,
                    1.0f, inputs.data(), INPUT_DIM,
                    weights.data(), INPUT_DIM,
                    0.0f, h1.data(), L1_DIM);

        // Bias & SiLU Activation
        int w_offset = INPUT_DIM * L1_DIM; // Move pointer past weights
        for (int i = 0; i < L1_DIM; i++) {
            float val = h1[i] + weights[w_offset + i]; // Add bias
            // SiLU: x * sigmoid(x)
            h1[i] = val * (1.0f / (1.0f + std::exp(-val))); 
        }
        w_offset += L1_DIM; // Move pointer past bias

        // --- LAYER 2: LIQUID STATE (2048 -> 1024) ---
        std::vector<float> h2(L2_DIM);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    1, L2_DIM, L1_DIM,
                    1.0f, h1.data(), L1_DIM,
                    &weights[w_offset], L1_DIM,
                    0.0f, h2.data(), L2_DIM);
        
        w_offset += L1_DIM * L2_DIM; // Move past Layer 2 weights
        
        // Bias, Tanh, and Liquid Time-Decay
        float time = inputs[6]; // The 't' variable is index 6
        for (int i = 0; i < L2_DIM; i++) {
            float val = std::tanh(h2[i] + weights[w_offset + i]); // Tanh activation
            // Liquid Decay Logic: e^(-t * decay_constant)
            h2[i] = val * std::exp(-time * 0.2f); 
        }
        w_offset += L2_DIM; // Move past Layer 2 bias

        // --- LAYER 3: OUTPUT MAPPING (1024 -> 2) ---
        std::vector<float> out(OUT_DIM);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    1, OUT_DIM, L2_DIM,
                    1.0f, h2.data(), L2_DIM,
                    &weights[w_offset], L2_DIM,
                    0.0f, out.data(), OUT_DIM);

        // Final Bias
        w_offset += L2_DIM * OUT_DIM;
        out[0] += weights[w_offset];
        out[1] += weights[w_offset + 1];

        return out;
    }
};

#endif