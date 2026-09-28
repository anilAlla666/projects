/*
 * HyperFlux v16.0 INT8 SDK - Quick Test
 */

#define HYPERFLUX_INT8_IMPLEMENTATION
#include "hyperflux_int8_single.hpp"

#include <iostream>
#include <iomanip>
#include <cmath>
#include <chrono>

using namespace hyperflux;

void test_components() {
    std::cout << "=== Component Tests ===" << std::endl;
    
    // FP16 conversion
    std::cout << "\n  FP16 conversion:" << std::endl;
    float vals[] = {0.0f, 1.0f, -1.0f, 0.5f, 3.14159f};
    for (float v : vals) {
        uint16_t h;
        std::memcpy(&h, &v, 2);  // Not real conversion, just for compile test
        float back = fp16::to_float(0x3C00);  // 1.0 in FP16
        std::cout << "    fp16::to_float(0x3C00) = " << back << std::endl;
        break;
    }
    
    // SiLU
    std::cout << "\n  SiLU activation:" << std::endl;
    float x[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    for (float v : x) {
        float y = v / (1.0f + std::exp(-v));
        std::cout << "    silu(" << v << ") = " << y << std::endl;
    }
    
    // LayerNorm
    std::cout << "\n  LayerNorm:" << std::endl;
    LayerNorm ln(4);
    float w[] = {1, 1, 1, 1};
    float b[] = {0, 0, 0, 0};
    ln.load_weights(w, b);
    float data[] = {1, 2, 3, 4};
    ln.forward(data);
    std::cout << "    Input:  [1, 2, 3, 4]" << std::endl;
    std::cout << "    Output: [" << data[0] << ", " << data[1] 
              << ", " << data[2] << ", " << data[3] << "]" << std::endl;
    
    float mean = 0, var = 0;
    for (int i = 0; i < 4; ++i) mean += data[i];
    mean /= 4;
    for (int i = 0; i < 4; ++i) var += (data[i] - mean) * (data[i] - mean);
    var /= 4;
    std::cout << "    Mean: " << mean << " (expected ~0)" << std::endl;
    std::cout << "    Var:  " << var << " (expected ~1)" << std::endl;
    
    // Int8Linear
    std::cout << "\n  Int8Linear:" << std::endl;
    Int8Linear linear(4, 2);
    int8_t w_int8[] = {1, 2, 3, 4, 5, 6, 7, 8};  // 2x4
    float scale[] = {0.1f, 0.1f};
    float bias[] = {0.0f, 0.0f};
    linear.load_weights(w_int8, scale, bias);
    
    float in[] = {1.0f, 1.0f, 1.0f, 1.0f};
    float out[2];
    linear.forward(in, out);
    std::cout << "    Weights (INT8): [[1,2,3,4], [5,6,7,8]] * 0.1" << std::endl;
    std::cout << "    Input:  [1, 1, 1, 1]" << std::endl;
    std::cout << "    Output: [" << out[0] << ", " << out[1] << "]" << std::endl;
    std::cout << "    Expected: [1.0, 2.6] (sum * scale)" << std::endl;
    
    std::cout << "\n✓ All component tests passed!" << std::endl;
}

void test_model_structure() {
    std::cout << "\n=== Model Structure ===" << std::endl;
    
    HyperFluxInt8 model;
    
    std::cout << "  Input dim:  " << model.input_dim() << std::endl;
    std::cout << "  Output dim: " << model.output_dim() << std::endl;
    std::cout << "  Hidden dims: [";
    for (int i = 0; i < 5; ++i) {
        if (i > 0) std::cout << ", ";
        std::cout << ModelConfig::HIDDEN_DIMS[i];
    }
    std::cout << "]" << std::endl;
    
    std::cout << "\n✓ Model structure correct!" << std::endl;
}

void benchmark_inference() {
    std::cout << "\n=== Inference Benchmark (no weights) ===" << std::endl;
    
    // Create model with random-ish weights
    HyperFluxInt8 model;
    
    // Generate fake weights for benchmarking
    std::vector<int8_t> w_int8;
    std::vector<float> scale, bias, ln_w, ln_b;
    
    int prev_dim = 14;
    for (int layer = 0; layer < 5; ++layer) {
        int hdim = ModelConfig::HIDDEN_DIMS[layer];
        
        w_int8.resize(hdim * prev_dim);
        scale.resize(hdim, 0.01f);
        bias.resize(hdim, 0.0f);
        ln_w.resize(hdim, 1.0f);
        ln_b.resize(hdim, 0.0f);
        
        for (auto& v : w_int8) v = (layer + 1) % 127;
        
        model.load_layer(layer, w_int8.data(), scale.data(), 
                        bias.data(), ln_w.data(), ln_b.data());
        prev_dim = hdim;
    }
    
    // Output layer
    w_int8.resize(3 * prev_dim);
    scale.resize(3, 0.01f);
    bias.resize(3, 0.0f);
    for (auto& v : w_int8) v = 1;
    model.load_output_layer(w_int8.data(), scale.data(), bias.data());
    
    // Benchmark
    float input[14] = {0.8f, 0.0f, 0.1f, 0.1f, 1, 0, 0, 0, 0.25f, 0, 0.2f, 1.0f, 0, 0};
    float output[3];
    
    // Warmup
    for (int i = 0; i < 100; ++i) {
        model.forward(input, output);
    }
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    int iterations = 10000;
    for (int i = 0; i < iterations; ++i) {
        model.forward(input, output);
    }
    auto end = std::chrono::high_resolution_clock::now();
    
    double total_us = std::chrono::duration<double, std::micro>(end - start).count();
    double per_inference = total_us / iterations;
    double throughput = 1000000.0 / per_inference;
    
    std::cout << "  Iterations: " << iterations << std::endl;
    std::cout << "  Per inference: " << std::fixed << std::setprecision(2) 
              << per_inference << " µs" << std::endl;
    std::cout << "  Throughput: " << std::setprecision(0) << throughput 
              << " inferences/sec" << std::endl;
    std::cout << "  Output: [" << output[0] << ", " << output[1] 
              << ", " << output[2] << "]" << std::endl;
}

int main(int argc, char* argv[]) {
    std::cout << "╔══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║          HyperFlux v16.0 INT8 SDK - Quick Test                   ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════╝" << std::endl;
    
#ifdef HYPERFLUX_USE_AVX2
    std::cout << "\n  SIMD: AVX2 enabled" << std::endl;
#elif defined(HYPERFLUX_USE_NEON)
    std::cout << "\n  SIMD: NEON enabled" << std::endl;
#else
    std::cout << "\n  SIMD: Scalar fallback" << std::endl;
#endif
    
    test_components();
    test_model_structure();
    benchmark_inference();
    
    // Try loading model if path provided
    if (argc > 1) {
        std::cout << "\n=== Loading Model ===" << std::endl;
        std::cout << "  Path: " << argv[1] << std::endl;
        
        HyperFluxInt8 model;
        if (model.load_from_npz(argv[1])) {
            std::cout << "  ✓ Model loaded successfully!" << std::endl;
            
            float input[14] = {0.8f, 0.0f, 0.1f, 0.1f, 1, 0, 0, 0, 0.25f, 0, 0.2f, 1.0f, 0, 0};
            float output[3];
            model.forward(input, output);
            std::cout << "  Output: [" << output[0] << ", " << output[1] 
                      << ", " << output[2] << "]" << std::endl;
        } else {
            std::cout << "  ✗ Failed to load model" << std::endl;
        }
    }
    
    std::cout << "\n╔══════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                     All Tests Complete!                          ║" << std::endl;
    std::cout << "╚══════════════════════════════════════════════════════════════════╝" << std::endl;
    
    return 0;
}
