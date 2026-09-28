#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <onnxruntime_cxx_api.h> // The Microsoft Engine

// CONFIGURATION
const int BENCHMARK_FRAMES = 5000; 

int main() {
    std::cout << "--- HYPERFLUX C++ KERNEL INITIALIZING ---" << std::endl;

    // 1. INITIALIZE ENVIRONMENT
    // We create the runtime environment. In a game, this happens once at startup.
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "HyperFlux");
    Ort::SessionOptions session_options;
    
    // OPTIMIZATION: Enable Graph Optimization (Fuse math operations)
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

    // 2. LOAD THE BRAIN
    // This reads the .onnx file and maps the neural pathways to memory.
    std::cout << "Loading Neural Surrogate..." << std::endl;
#ifdef _WIN32
    const wchar_t* model_path = L"hyperflux_engine.onnx"; // Windows needs wide strings
#else
    const char* model_path = "hyperflux_engine.onnx";
#endif

    Ort::Session session(env, model_path, session_options);

    // 3. DEFINE I/O SHAPES
    // We know our input is [Batch=1, Seq=1, Features=3] -> (Time, Sin, Cos)
    std::vector<int64_t> input_shape = {1, 1, 3};
    
    // 4. THE BENCHMARK LOOP
    std::cout << "Starting High-Frequency Simulation (" << BENCHMARK_FRAMES << " frames)..." << std::endl;
    
    // Start the stopwatch
    auto start_time = std::chrono::high_resolution_clock::now();

    // MEMORY ALLOCATION (Doing this outside the loop is faster)
    std::vector<float> input_values(3); 
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const char* input_names[] = {"input_state"};
    const char* output_names[] = {"physics_out"};

    for (int i = 0; i < BENCHMARK_FRAMES; i++) {
        float t = i * 0.01f;

        // STEP A: PREPARE INPUTS (The "Sensors")
        // Instead of just 't', we feed the encoded vector [t, sin(t), cos(t)]
        input_values[0] = t;
        input_values[1] = std::sin(t);
        input_values[2] = std::cos(t);

        // Wrap data in an ONNX Tensor object
        Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
            memory_info, input_values.data(), 3, input_shape.data(), 3
        );

        // STEP B: INFERENCE (The "Magic")
        // Run the neural network
        auto output_tensors = session.Run(
            Ort::RunOptions{nullptr}, 
            input_names, &input_tensor, 1, 
            output_names, 1
        );

        // STEP C: READ OUTPUT
        // In a real game, we would apply this value to the object's position here.
        float* result = output_tensors.front().GetTensorMutableData<float>();
        
        // Optional: Print first few to prove it works
        if (i < 5) {
            std::cout << "Frame " << i << " | t=" << t << " | Physics=" << result[0] << std::endl;
        }
    }

    // Stop the stopwatch
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    // 5. REPORT CARD
    double fps = BENCHMARK_FRAMES / elapsed.count();
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "COMPLETED IN: " << elapsed.count() << " seconds" << std::endl;
    std::cout << "SPEED:        " << fps << " FPS" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    return 0;
}