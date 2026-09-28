/*
 * SOMA v7 — ros2_control Plugin Wrapper
 *
 * Thin wrapper that integrates SOMA into the ros2_control framework.
 * The controller reads joint states, calls soma_step(), and writes
 * torque commands.
 *
 * Build as part of a ROS2 package:
 *   add_library(soma_controller SHARED soma_ros2_controller.cpp)
 *   target_link_libraries(soma_controller m)  # only -lm needed
 *
 * Register in plugin.xml:
 *   <class name="soma_controller/SomaController"
 *          type="soma_controller::SomaController"
 *          base_class_type="controller_interface::ControllerInterface">
 *   </class>
 */

#ifndef SOMA_ROS2_CONTROLLER_HPP
#define SOMA_ROS2_CONTROLLER_HPP

/*
 * NOTE: This is a reference implementation showing the integration pattern.
 * Adjust includes and namespaces to match your ros2_control version.
 *
 * The key insight: SOMA's entire inference is a single C function call.
 * All the ROS2 code below is just plumbing to get joint states in
 * and torque commands out.
 */

/*
 * ============================================================
 * PSEUDOCODE — ros2_control integration
 * ============================================================
 *
 * #include "controller_interface/controller_interface.hpp"
 * #include "soma.h"   // The C SDK header
 *
 * class SomaController : public controller_interface::ControllerInterface {
 * private:
 *     SomaHandle* soma_ = nullptr;
 *     int n_joints_ = 7;  // configured from YAML
 *     double dt_ = 0.002; // control period
 *     
 *     // Reference trajectory (from higher-level planner)
 *     std::vector<double> q_ref_;
 *
 * public:
 *     // Called once at startup
 *     CallbackReturn on_init() override {
 *         // Read weights directory from parameter
 *         std::string weights_dir = get_node()->get_parameter("weights_dir").as_string();
 *         n_joints_ = get_node()->get_parameter("n_joints").as_int();
 *         dt_ = get_node()->get_parameter("dt").as_double();
 *         
 *         soma_ = soma_init(weights_dir.c_str());
 *         if (!soma_) return CallbackReturn::ERROR;
 *         
 *         q_ref_.resize(n_joints_, 0.0);
 *         return CallbackReturn::SUCCESS;
 *     }
 *
 *     // Called every control cycle (1kHz)
 *     controller_interface::return_type update(
 *         const rclcpp::Time& time, const rclcpp::Duration& period) override
 *     {
 *         // 1. Read joint states from hardware interface
 *         float state[SOMA_INPUT_DIM];  // [q_err, q_vel, grav_comp, q_ref]
 *         for (int i = 0; i < n_joints_; i++) {
 *             double q = state_interfaces_[i].get_value();           // position
 *             double qd = state_interfaces_[n_joints_ + i].get_value(); // velocity
 *             state[i] = (float)(q_ref_[i] - q);          // q_error
 *             state[n_joints_ + i] = (float)qd;            // q_vel
 *             state[2*n_joints_ + i] = 0.0f;               // gravity_comp (from model)
 *             state[3*n_joints_ + i] = (float)q_ref_[i];   // q_ref
 *         }
 *
 *         // 2. Run SOMA inference — THIS IS THE ENTIRE NEURAL NETWORK
 *         float torque[SOMA_OUTPUT_DIM];
 *         soma_step(soma_, state, (float)dt_, torque);
 *
 *         // 3. Write torque commands to hardware
 *         for (int i = 0; i < n_joints_; i++) {
 *             command_interfaces_[i].set_value((double)torque[i]);
 *         }
 *
 *         return controller_interface::return_type::OK;
 *     }
 *
 *     // Cleanup
 *     CallbackReturn on_deactivate(const rclcpp_lifecycle::State&) override {
 *         soma_free(soma_);
 *         soma_ = nullptr;
 *         return CallbackReturn::SUCCESS;
 *     }
 * };
 *
 * ============================================================
 * YAML configuration (soma_controller.yaml):
 * ============================================================
 *
 * soma_controller:
 *   ros__parameters:
 *     weights_dir: "/opt/soma/weights"
 *     n_joints: 7
 *     dt: 0.002
 *     joints:
 *       - joint1
 *       - joint2
 *       - joint3
 *       - joint4
 *       - joint5
 *       - joint6
 *       - joint7
 *     command_interfaces:
 *       - effort
 *     state_interfaces:
 *       - position
 *       - velocity
 *
 * ============================================================
 */

#endif /* SOMA_ROS2_CONTROLLER_HPP */
