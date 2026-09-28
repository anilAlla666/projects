/*
 * SOMA v7 — Unitree SDK2 Integration
 *
 * Reference implementation for Unitree G1/H1 humanoid robots.
 * Uses unitree_sdk2 C++ API to read joint states and write torques.
 *
 * The robot's Jetson Orin NX runs this process, communicating with
 * the locomotion computer via CycloneDDS at 1kHz.
 *
 * Build:
 *   g++ -O2 -std=c++17 soma_unitree.cpp -lm -lunitree_sdk2 -lcyclonedds -o soma_unitree
 *
 * Run (on G1 dev computer at 192.168.123.99):
 *   ./soma_unitree ./weights
 */

/*
 * ============================================================
 * PSEUDOCODE — Unitree G1 integration via unitree_sdk2
 * ============================================================
 *
 * #include "unitree/robot/channel/channel_publisher.hpp"
 * #include "unitree/robot/channel/channel_subscriber.hpp"
 * #include "unitree/idl/hg/LowCmd_.hpp"
 * #include "unitree/idl/hg/LowState_.hpp"
 *
 * // Include SOMA — zero dependencies
 * #define SOMA_IMPLEMENTATION
 * #include "soma.h"
 *
 * // G1 has 29 joints. SOMA trains per-robot, so the weights
 * // loaded here must match the G1 training run.
 * // For G1: INPUT_DIM = 4*29 = 116, OUTPUT_DIM = 29
 * // (Rebuild soma.h with adjusted #defines, or use dynamic version)
 *
 * class SomaUnitreeController {
 * private:
 *     SomaHandle* soma_ = nullptr;
 *     
 *     // DDS channels
 *     unitree::robot::ChannelPublisher<unitree_hg::msg::LowCmd_> cmd_pub_;
 *     unitree::robot::ChannelSubscriber<unitree_hg::msg::LowState_> state_sub_;
 *     
 *     // Latest state from robot
 *     unitree_hg::msg::LowState_ latest_state_;
 *     std::mutex state_mutex_;
 *     
 *     // Reference trajectory (from VLA/planner)
 *     std::vector<float> q_ref_;
 *     int n_joints_ = 29;  // G1 main body
 *     float dt_ = 0.001f;  // 1kHz
 *
 * public:
 *     bool init(const char* weights_dir) {
 *         // Load SOMA weights
 *         soma_ = soma_init(weights_dir);
 *         if (!soma_) return false;
 *         
 *         // Init DDS channels
 *         unitree::robot::ChannelFactory::Instance()->Init(0, "enp131s0");
 *         cmd_pub_.InitChannel("rt/lowcmd");
 *         state_sub_.InitChannel("rt/lowstate");
 *         state_sub_.SetRecvCallback(
 *             [this](const void* msg) { this->on_state(msg); }
 *         );
 *         
 *         q_ref_.resize(n_joints_, 0.0f);
 *         return true;
 *     }
 *     
 *     void on_state(const void* msg) {
 *         // Called by DDS when new state arrives (~500Hz from locomotion computer)
 *         std::lock_guard<std::mutex> lock(state_mutex_);
 *         latest_state_ = *(const unitree_hg::msg::LowState_*)msg;
 *     }
 *     
 *     void control_loop() {
 *         // Real-time loop at 1kHz
 *         while (running_) {
 *             auto t0 = std::chrono::steady_clock::now();
 *             
 *             // 1. Read latest joint state
 *             unitree_hg::msg::LowState_ state;
 *             {
 *                 std::lock_guard<std::mutex> lock(state_mutex_);
 *                 state = latest_state_;
 *             }
 *             
 *             // 2. Build SOMA input: [q_error, q_vel, grav_comp, q_ref]
 *             float soma_input[4 * 29];  // 116 for G1
 *             for (int i = 0; i < n_joints_; i++) {
 *                 float q = state.motor_state()[i].q();
 *                 float qd = state.motor_state()[i].dq();
 *                 soma_input[i]              = q_ref_[i] - q;   // q_error
 *                 soma_input[n_joints_ + i]  = qd;               // q_vel
 *                 soma_input[2*n_joints_+i]  = 0.0f;             // grav_comp
 *                 soma_input[3*n_joints_+i]  = q_ref_[i];        // q_ref
 *             }
 *             
 *             // 3. SOMA inference — THE ENTIRE NEURAL NETWORK
 *             float torque[29];
 *             soma_step(soma_, soma_input, dt_, torque);
 *             
 *             // 4. Send torque commands to robot
 *             unitree_hg::msg::LowCmd_ cmd{};
 *             for (int i = 0; i < n_joints_; i++) {
 *                 cmd.motor_cmd()[i].mode() = 0x01;  // torque mode
 *                 cmd.motor_cmd()[i].tau() = torque[i];
 *                 cmd.motor_cmd()[i].kp() = 0.0f;    // no PD — SOMA handles everything
 *                 cmd.motor_cmd()[i].kd() = 0.0f;
 *             }
 *             cmd_pub_.Write(cmd);
 *             
 *             // 5. Sleep to maintain 1kHz
 *             auto t1 = std::chrono::steady_clock::now();
 *             auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
 *             auto sleep_us = std::chrono::microseconds(1000) - elapsed;
 *             if (sleep_us.count() > 0) {
 *                 std::this_thread::sleep_for(sleep_us);
 *             }
 *         }
 *     }
 *     
 *     void set_reference(const std::vector<float>& q_ref) {
 *         q_ref_ = q_ref;
 *     }
 *     
 *     ~SomaUnitreeController() {
 *         if (soma_) soma_free(soma_);
 *     }
 * };
 *
 * // Main — the entire deployment
 * int main(int argc, char** argv) {
 *     SomaUnitreeController ctrl;
 *     ctrl.init(argv[1]);
 *     
 *     // Set home position as initial reference
 *     ctrl.set_reference(G1_HOME_POSITION);
 *     
 *     // Run real-time control
 *     ctrl.control_loop();
 *     return 0;
 * }
 *
 * ============================================================
 */
