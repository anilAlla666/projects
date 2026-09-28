import React from 'react';
import { motion } from 'framer-motion';
import ComparisonTable from './ComparisonTable';
import CapabilitiesGrid from './CapabilitiesGrid';
import { Cpu, Activity, Network } from 'lucide-react';

const RoboticsSection = ({ id }) => {
  return (
    <section className="py-32 relative overflow-hidden" id={id}>
      <div className="absolute inset-0 bg-navy-950/80 pointer-events-none" />
      <div className="absolute top-0 right-0 w-[800px] h-[800px] bg-red-900/10 rounded-full blur-[120px] pointer-events-none" />
      
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <motion.div 
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="text-center mb-20"
        >
          <div className="inline-flex items-center justify-center p-3 bg-red-500/10 rounded-xl mb-6 text-red-400 border border-red-500/20">
            <Cpu className="w-6 h-6 mr-2" /> Robotics Intelligence
          </div>
          <h2 className="text-4xl md:text-6xl font-bold text-white mb-6">
            Project <span className="text-transparent bg-clip-text bg-gradient-to-r from-red-500 to-gold-400">TIRUSOMA</span>
          </h2>
          <p className="text-xl text-gray-400 max-w-3xl mx-auto">
            The world's first O(1) motor cortex for humanoid and industrial robotics, redefining physical intelligence.
          </p>
        </motion.div>

        {/* The Problem & What We're Building */}
        <div className="grid md:grid-cols-2 gap-12 mb-20">
          <motion.div 
            initial={{ opacity: 0, x: -30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="p-8 rounded-2xl bg-navy-900/30 border border-white/5"
          >
            <h3 className="text-2xl font-bold text-white mb-4 flex items-center gap-3">
              <Activity className="text-red-400" /> The Problem
            </h3>
            <p className="text-gray-400 leading-relaxed">
              Modern robotics relies on Model Predictive Control (MPC) and complex kinematic equations that scale poorly. Computing balance, trajectory, and multi-joint actuation requires intense processing power, resulting in high latency, jittery movement, and the inability to adapt instantly to unforeseen physical perturbations.
            </p>
          </motion.div>

          <motion.div 
            initial={{ opacity: 0, x: 30 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="p-8 rounded-2xl bg-navy-900/30 border border-white/5"
          >
            <h3 className="text-2xl font-bold text-white mb-4 flex items-center gap-3">
              <Network className="text-gold-400" /> What We're Building
            </h3>
            <p className="text-gray-400 leading-relaxed">
              TIRUSOMA is a neural motor cortex that utilizes phase-space encoding to calculate exact inverse kinematics and dynamics in O(1) time. By mapping the entire physical possibility space into a queryable latent structure, TIRUSOMA provides instant, fluid, and biologically-accurate motor commands to any robotic rig.
            </p>
          </motion.div>
        </div>

        {/* Comparison Table */}
        <div className="mb-24">
          <h3 className="text-3xl font-bold text-white text-center mb-8">How This Compares</h3>
          <ComparisonTable />
        </div>

        {/* Capabilities Grid */}
        <div className="mb-24">
          <div className="text-center mb-12">
            <h3 className="text-3xl font-bold text-white mb-4">21 Core Motor Behaviors</h3>
            <p className="text-gray-400">Natively supported and executed in sub-millisecond latency.</p>
          </div>
          <CapabilitiesGrid />
        </div>

        {/* Roadmap & Market Context */}
        <div className="grid md:grid-cols-2 gap-12">
          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            className="glass-card-gold p-8 rounded-2xl"
          >
            <h3 className="text-2xl font-bold text-white mb-6">Where We Are Today</h3>
            <ul className="space-y-4 text-gray-300">
              <li className="flex items-start gap-3">
                <span className="w-2 h-2 rounded-full bg-gold-500 mt-2" />
                <span>Simulated bipeds walking natively without traditional IK solvers.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="w-2 h-2 rounded-full bg-gold-500 mt-2" />
                <span>Latency verified at 0.02ms on standard edge hardware.</span>
              </li>
              <li className="flex items-start gap-3">
                <span className="w-2 h-2 rounded-full bg-gold-500 mt-2" />
                <span>Integration APIs established for ROS2 and custom firmware.</span>
              </li>
            </ul>
          </motion.div>

          <motion.div 
            initial={{ opacity: 0, y: 30 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true }}
            transition={{ delay: 0.2 }}
            className="bg-navy-900 border border-white/10 p-8 rounded-2xl"
          >
            <h3 className="text-2xl font-bold text-white mb-6">Market Context</h3>
            <p className="text-gray-400 mb-4">
              The humanoid robotics market is projected to reach $38B by 2035. The primary bottleneck is not hardware, but software intelligence capable of fluid real-world operation.
            </p>
            <p className="text-gray-400">
              TIRUSOMA acts as the foundational OS for next-generation hardware, licensing the "cerebellum" to hardware manufacturers.
            </p>
          </motion.div>
        </div>

      </div>
    </section>
  );
};

export default RoboticsSection;